"""
Pricing a bo3 series from the Bayesian series model's posterior, and blending it with the pre-map model.

The model (fitted in models/cs2/bayes) gives each player a round-logit skill. For a map between lineups A and B:

    D = mean(theta_A) - mean(theta_B) + beta[A, map] - beta[B, map] + rho[home A] - rho[home B] + eta_map + eps_series

and A wins each round with probability sigmoid(D). eta is per map (rounds within a map are correlated), eps is
shared by every map of the series (when a matchup is misjudged, every map is off the same way). Pricing lives here,
base-only, so a live session prices from the exported numbers without installing pymc.

Before the veto the maps are unknown: the team-map term is averaged over the two teams' map pool, weighted by how
often each played the map in the fit window, and every map slot gets the same expected term. Given eps, the three
slots are then independent with the same map probability p, so P(series) = p^2 (3 - 2p).
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win

REGIONS = ("europe", "north america", "south america", "asia", "oceania", "cis")
# Index 0 is "unknown" and index 1 Europe, the reference; both are fixed at zero in the model.
UNKNOWN_REGION = 0
UNRANKED = 300.0
RANK_MAX_AGE_DAYS = 120
LINEUP_SIZE = 5

_OUTER_X, _OUTER_W = np.polynomial.hermite_e.hermegauss(21)
_OUTER_W = _OUTER_W / _OUTER_W.sum()
_INNER_X, _INNER_W = np.polynomial.hermite_e.hermegauss(21)
_INNER_W = _INNER_W / _INNER_W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)


def event_region(event_name: str | None) -> int:
    """Region code of an event from its name (1 Europe .. 6 CIS), or UNKNOWN_REGION for international events."""
    n = (event_name or "").lower()
    for i, r in enumerate(REGIONS):
        if re.search(rf"\b{r}\b", n) or (r == "north america" and re.search(r"\bna\b", n)) \
                or (r == "south america" and re.search(r"\bsa\b|latam", n)):
            return i + 1
    return UNKNOWN_REGION


def home_regions(series: pd.DataFrame) -> dict[int, int]:
    """Each team's most frequent regional-event region among `series` (event_name, team_a_id, team_b_id)."""
    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for name, a, b in zip(series.event_name, series.team_a_id, series.team_b_id):
        region = event_region(name)
        if region != UNKNOWN_REGION:
            counts[int(a)][region] += 1
            counts[int(b)][region] += 1
    return {t: c.most_common(1)[0][0] for t, c in counts.items()}


class RankStrength:
    """-log of a team's latest HLTV rank on or before a day; older than RANK_MAX_AGE_DAYS counts as unranked."""

    def __init__(self, rankings: pd.DataFrame):
        r = rankings.assign(date=_naive(pd.to_datetime(rankings["date"]))).sort_values("date")
        self._hist = {int(t): (g.date.to_numpy(), g["rank"].to_numpy(float)) for t, g in r.groupby("team_id")}

    def __call__(self, team: int, day: pd.Timestamp) -> float:
        day = pd.Timestamp(day)
        day = day.tz_convert(None) if day.tz is not None else day
        h = self._hist.get(int(team))
        if h is None:
            return -np.log(UNRANKED)
        i = np.searchsorted(h[0], np.datetime64(day), side="right") - 1
        if i < 0 or (np.datetime64(day) - h[0][i]) > np.timedelta64(RANK_MAX_AGE_DAYS, "D"):
            return -np.log(UNRANKED)
        return -np.log(h[1][i])


def _naive(dates: pd.Series) -> pd.Series:
    """UTC without a zone: the datasets store naive UTC, extracts and live rows may carry the zone."""
    return dates.dt.tz_convert(None) if dates.dt.tz is not None else dates


def _bo3(p: np.ndarray) -> np.ndarray:
    return p * p * (3 - 2 * p)


@dataclass(frozen=True, eq=False)
class SeriesPosterior:
    """Posterior draws (S of them) needed to price a series, as plain arrays so they outlive library versions."""

    player_ids: np.ndarray      # (P,)
    theta: np.ndarray           # (S, P) skill in the last fitted month
    tau: float                  # monthly skill drift, carried one month forward when pricing
    b: np.ndarray               # (S,) newcomer prior: theta ~ N(b * standardized strength, sigma0)
    sigma0: np.ndarray          # (S,)
    strength_mu: float
    strength_sd: float
    rho: np.ndarray             # (S, len(REGIONS) + 1) by region code
    sigma_m: np.ndarray         # (S,) per-map noise
    sigma_s: np.ndarray         # (S,) per-series noise
    home_team_ids: np.ndarray   # (H,)
    home_region: np.ndarray     # (H,)
    tm_team_ids: np.ndarray     # (K,) team-map effects
    tm_maps: np.ndarray         # (K,)
    beta: np.ndarray            # (S, K)
    plays_team_ids: np.ndarray  # (M,) maps each team played in the window, for the pre-veto map pool
    plays_maps: np.ndarray      # (M,)
    plays_counts: np.ndarray    # (M,)
    # Debutants (no map anywhere in the data before the fit's end) start from their own prior: the regular one is
    # learned mostly from established players and rates an unknown newcomer on a good team too highly. Without the
    # debut prior, a_new is 0 and sigma_new is sigma0, so every unseen player takes the regular prior.
    a_new: np.ndarray           # (S,)
    sigma_new: np.ndarray       # (S,)
    career_ids: np.ndarray      # players with any map before the fit's end
    as_of: str                  # the fit window's end (ISO date): the posterior knows nothing after it

    def __post_init__(self):
        object.__setattr__(self, "_pidx", {int(p): i for i, p in enumerate(self.player_ids)})
        object.__setattr__(self, "_home", dict(zip(self.home_team_ids.tolist(), self.home_region.tolist())))
        object.__setattr__(self, "_tmidx", {(int(t), str(m)): i for i, (t, m) in enumerate(zip(self.tm_team_ids, self.tm_maps))})
        plays: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        for t, m, c in zip(self.plays_team_ids, self.plays_maps, self.plays_counts):
            plays[int(t)][str(m)] += int(c)
        object.__setattr__(self, "_plays", plays)
        object.__setattr__(self, "_career", set(self.career_ids.tolist()))
        # The map noise is integrated on a fixed grid at its posterior mean: its posterior spread (~0.01 on 0.37)
        # moves a price by far less than the draws' sampling noise, and the grid makes pricing a lookup.
        noise = float(self.sigma_m.mean())
        table = regulation_map_win(_GRID[:, None] + noise * _INNER_X[None, :]) @ _INNER_W
        object.__setattr__(self, "_map_table", table)

    @property
    def n_draws(self) -> int:
        return int(self.b.shape[0])

    def age_days(self, now: pd.Timestamp) -> float:
        return (pd.Timestamp(now).tz_localize(None).normalize() - pd.Timestamp(self.as_of)).days

    def thinned(self, n_draws: int, seed: int = 0) -> SeriesPosterior:
        """n_draws of the draws, kept at random, with every per-draw array as float32: a few MB a live session can load.
        Prices move by far less than the draws' own sampling noise."""
        keep = np.sort(np.random.default_rng(seed).choice(self.n_draws, size=min(n_draws, self.n_draws), replace=False))
        fields = {k: getattr(self, k) for k in SeriesPosterior.__dataclass_fields__}
        for k, v in fields.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == self.n_draws and v.dtype.kind == "f":
                fields[k] = v[keep].astype(np.float32)
        return SeriesPosterior(**fields)

    def _lineup(self, lineup: list[int], strength: float) -> tuple[np.ndarray, np.ndarray]:
        """Mean skill per draw, and the variance per draw that drift and newcomer draws add to it."""
        prior = self.b * (strength - self.strength_mu) / self.strength_sd
        mean, var = np.zeros(self.n_draws), np.zeros(self.n_draws)
        for p in lineup:
            i = self._pidx.get(int(p))
            if i is None and int(p) in self._career:
                mean += prior
                var += self.sigma0 ** 2
            elif i is None:
                mean += prior + self.a_new
                var += self.sigma_new ** 2
            else:
                mean += self.theta[:, i]
                var += self.tau ** 2
        return mean / len(lineup), var / len(lineup) ** 2

    def _map_term(self, team_a: int, team_b: int) -> np.ndarray:
        pool = self._plays[int(team_a)] + self._plays[int(team_b)]
        effect = np.zeros(self.n_draws)
        total = sum(pool.values())
        for m, c in pool.items():
            for team, sign in ((team_a, 1.0), (team_b, -1.0)):
                i = self._tmidx.get((int(team), m))
                if i is not None:
                    effect += sign * (c / total) * self.beta[:, i]
        return effect

    def gap(self, team_a: int, lineup_a: list[int], strength_a: float,
            team_b: int, lineup_b: list[int], strength_b: float) -> tuple[np.ndarray, np.ndarray]:
        """Per draw: the expected round-logit gap of a pre-veto map, and the variance shared by every map."""
        mean_a, var_a = self._lineup(lineup_a, strength_a)
        mean_b, var_b = self._lineup(lineup_b, strength_b)
        rho_a = self.rho[:, self._home.get(int(team_a), UNKNOWN_REGION)]
        rho_b = self.rho[:, self._home.get(int(team_b), UNKNOWN_REGION)]
        mean = mean_a - mean_b + self._map_term(team_a, team_b) + rho_a - rho_b
        return mean, var_a + var_b + self.sigma_s ** 2

    def price(self, team_a: int, lineup_a: list[int], strength_a: float,
              team_b: int, lineup_b: list[int], strength_b: float) -> float:
        """P(team A wins the bo3), averaged over the posterior draws."""
        mean, shared_var = self.gap(team_a, lineup_a, strength_a, team_b, lineup_b, strength_b)
        d = mean[:, None] + np.sqrt(shared_var)[:, None] * _OUTER_X[None, :]
        p = np.interp(d, _GRID, self._map_table)
        return float((_bo3(p) @ _OUTER_W).mean())

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{k: np.asarray(getattr(self, k)) for k in SeriesPosterior.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> SeriesPosterior:
        with np.load(path, allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        for k in ("tau", "strength_mu", "strength_sd"):
            d[k] = float(d[k])
        d["as_of"] = str(d["as_of"])
        return cls(**d)


@dataclass(frozen=True)
class SeriesBlend:
    """sigmoid(w_bayes * logit(q_bayes) + w_xgb * logit(q_xgb)); no intercept, so it stays symmetric in A and B."""

    w_bayes: float
    w_xgb: float

    @classmethod
    def fit(cls, q_bayes: np.ndarray, q_xgb: np.ndarray, y: np.ndarray) -> SeriesBlend:
        X = np.column_stack([_logit(q_bayes), _logit(q_xgb)])
        y = np.asarray(y, dtype=float)
        w = np.zeros(2)
        for _ in range(100):
            p = 1 / (1 + np.exp(-X @ w))
            step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
            w += step
            if np.abs(step).max() < 1e-10:
                break
        return cls(float(w[0]), float(w[1]))

    def predict(self, q_bayes: np.ndarray, q_xgb: np.ndarray) -> np.ndarray:
        z = self.w_bayes * _logit(q_bayes) + self.w_xgb * _logit(q_xgb)
        return 1 / (1 + np.exp(-z))

    def to_dict(self) -> dict:
        return {"w_bayes": self.w_bayes, "w_xgb": self.w_xgb}

    @classmethod
    def from_dict(cls, d: dict) -> SeriesBlend:
        return cls(float(d["w_bayes"]), float(d["w_xgb"]))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))
