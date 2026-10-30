"""
Per-player Gaussian skill filter over HLTV and FACEIT maps.

Copied from models/cs2/hltv/player_skill.py and changed in three ways: every observation carries a source, and each
source has its own beta-binomial overdispersion and likelihood weight (FACEIT league maps are played with less at
stake than HLTV-covered series, so they are allowed to say less per round); a player who first appears without an
HLTV team starts from a prior set by the division of that competition rather than from ranking points; and the
filter keeps each team's most recent five, which is the only lineup a pre-match prediction can know when the
announced lineup is not in the data yet.
"""
from __future__ import annotations

import bisect
import collections
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import TIE_TOLERANCE, regulation_map_win

HLTV, FACEIT = "hltv", "faceit"

# Round-logit prior means by ESEA division (and the qualifier/other buckets) for players with no HLTV team at
# their debut; the HLTV prior for an unranked team is `a` (about -0.11), ESEA Open sits well below that.
DIVISION_PRIOR = {"entry": -0.55, "open": -0.50, "intermediate": -0.35, "main": -0.20, "advanced": -0.05,
                  "ecl": 0.0, "qualifier": -0.10, "other": -0.20}

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)


@dataclass(frozen=True)
class FilterConfig:
    tau: float = 0.009319191322975635
    sigma0: float = 0.2536091079975324
    a: float = -0.1081114476105491
    b: float = 0.14696262712175345
    lam: float = 0.0761331162651603
    kappa_max: float = 0.03328732428013525
    c0: float = 7.134081474707468
    rho: float = 0.017231894894383412
    chem_days: int = 180
    faceit_weight: float = 1.0
    faceit_rho: float = 0.017231894894383412
    faceit_sigma0: float = 0.2536091079975324
    faceit_prior_shift: float = 0.0
    # Pricing: map-level and series-level noise added to the filter's own gap variance.
    sigma_map: float = 0.0
    sigma_series: float = 0.0


def division(competition_name: str | None) -> str:
    n = (competition_name or "").lower()
    if "qualifier" in n:
        return "qualifier"
    for key in ("entry", "open", "intermediate", "main", "advanced", "ecl", "challenger"):
        if key in n:
            return "ecl" if key == "challenger" else key
    return "other"


@dataclass
class Lineup:
    players: list[int]
    mus: list[float]
    variances: list[float]
    weights: list[float]
    strength: float
    kappa: float
    total: float


class SkillFilter:
    def __init__(self, cfg: FilterConfig, rankings: pd.DataFrame | None):
        self.cfg = cfg
        self._mu: dict[int, float] = {}
        self._var: dict[int, float] = {}
        self._seen: dict[int, pd.Timestamp] = {}
        self._cov: dict[tuple[int, int], float] = {}
        self.games: collections.Counter = collections.Counter()
        self._five_days: dict[frozenset, collections.deque] = collections.defaultdict(collections.deque)
        self.last_five: dict[int, tuple[int, ...]] = {}
        self._rank_dates: list[pd.Timestamp] = []
        self._rank_points: list[dict[int, float]] = []
        if rankings is not None and len(rankings):
            r = rankings.assign(date=pd.to_datetime(rankings["date"]))
            for day, g in r.groupby("date", sort=True):
                self._rank_dates.append(pd.Timestamp(day))
                self._rank_points.append(dict(zip(g["team_id"].astype(int), g["points"].astype(float))))

    def _points(self, team_id: int, day: pd.Timestamp) -> float:
        i = bisect.bisect_left(self._rank_dates, day) - 1
        return self._rank_points[i].get(team_id, 0.0) if i >= 0 else 0.0

    def prior(self, hltv_team: int | None, day: pd.Timestamp, div: str) -> tuple[float, float]:
        if hltv_team is not None:
            return self.cfg.a + self.cfg.b * math.log1p(self._points(hltv_team, day)), self.cfg.sigma0 ** 2
        return DIVISION_PRIOR[div] + self.cfg.faceit_prior_shift, self.cfg.faceit_sigma0 ** 2

    def _state(self, player: int, hltv_team: int | None, day: pd.Timestamp, div: str) -> tuple[float, float]:
        if player not in self._mu:
            return self.prior(hltv_team, day, div)
        idle = max((day - self._seen[player]).days, 0)
        return self._mu[player], self._var[player] + self.cfg.tau ** 2 * idle

    def _five_maps(self, five: tuple[int, ...], day: pd.Timestamp) -> float:
        q = self._five_days.get(frozenset(five))
        if not q:
            return 0.0
        cutoff = day - pd.Timedelta(days=self.cfg.chem_days)
        return float(sum(1 for d in q if d >= cutoff))

    def _weights(self, mus: list[float]) -> list[float]:
        n = len(mus)
        top = max(mus)
        best = next(i for i, m in enumerate(mus) if m >= top - TIE_TOLERANCE)
        return [1 / n + self.cfg.lam * ((1.0 if i == best else 0.0) - 1 / n) for i in range(n)]

    def _block(self, players: list[int], variances: list[float]) -> np.ndarray:
        cov = np.diag(variances)
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p, q = sorted((players[i], players[j]))
                cov[i, j] = cov[j, i] = self._cov.get((p, q), 0.0)
        return cov

    def lineup(self, players: tuple[int, ...], hltv_team: int | None, day: pd.Timestamp, div: str = "other") -> Lineup:
        states = [self._state(p, hltv_team, day, div) for p in players]
        mus = [s[0] for s in states]
        weights = self._weights(mus)
        five_maps = self._five_maps(players, day) if len(players) == 5 else 0.0
        kappa = self.cfg.kappa_max * (1 - math.exp(-five_maps / self.cfg.c0))
        strength = sum(w * m for w, m in zip(weights, mus))
        return Lineup(list(players), mus, [s[1] for s in states], weights, strength, kappa, strength + kappa)

    def gap(self, a: Lineup, b: Lineup) -> tuple[float, float, np.ndarray, np.ndarray]:
        w = np.array(a.weights + [-x for x in b.weights])
        cov = self._block(a.players + b.players, a.variances + b.variances)
        return a.total - b.total, float(w @ cov @ w), cov, w

    def observe_map(self, day: pd.Timestamp, key_a: int, key_b: int, hltv_a: int | None, hltv_b: int | None,
                    players_a: tuple[int, ...], players_b: tuple[int, ...], rounds_a: int, rounds_b: int,
                    source: str, div: str) -> None:
        if not players_a or not players_b:
            return
        if len(players_a) == 5:
            self.last_five[key_a] = players_a
        if len(players_b) == 5:
            self.last_five[key_b] = players_b
        a = self.lineup(players_a, hltv_a, day, div)
        b = self.lineup(players_b, hltv_b, day, div)
        players = a.players + b.players
        if len(set(players)) != len(players):
            return
        d, s2, cov, w = self.gap(a, b)
        n = rounds_a + rounds_b
        pi = 1 / (1 + math.exp(-d))
        rho, weight = (self.cfg.faceit_rho, self.cfg.faceit_weight) if source == FACEIT else (self.cfg.rho, 1.0)
        inflation = (1 + (n - 1) * rho) / weight
        grad, curv = (rounds_a - n * pi) / inflation, n * pi * (1 - pi) / inflation
        cw = cov @ w
        denom = 1 + curv * s2
        mus = np.array(a.mus + b.mus) + cw * grad / denom
        cov = cov - np.outer(cw, cw) * curv / denom
        for i, p in enumerate(players):
            self._mu[p] = float(mus[i])
            self._var[p] = max(1e-6, float(cov[i, i]))
            self._seen[p] = day
            self.games[p] += 1
            for j in range(i + 1, len(players)):
                self._cov[tuple(sorted((p, players[j])))] = float(cov[i, j])
        cutoff = day - pd.Timedelta(days=self.cfg.chem_days)
        for five in (players_a, players_b):
            if len(five) == 5:
                q = self._five_days[frozenset(five)]
                q.append(day)
                while q and q[0] < cutoff:
                    q.popleft()


class SeriesPricer:
    """Round-logit gap ~ N(d, s2) to P(bo3): series-shared uncertainty integrated outside the bo3 polynomial,
    map noise inside it, on a grid so pricing is a lookup."""

    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        table = regulation_map_win(_GRID[:, None] + cfg.sigma_map * _GH_X[None, :]) @ _GH_W
        self._table = table

    def series(self, d: float, s2: float) -> float:
        sd = math.sqrt(max(s2, 0.0) + self.cfg.sigma_series ** 2)
        p = np.interp(d + sd * _GH_X, _GRID, self._table)
        return float((p * p * (3 - 2 * p)) @ _GH_W)


def placeholder_five(team_key: int) -> tuple[int, ...]:
    """Five ids no player ever has, so a team with no map in the data prices from its prior alone."""
    return tuple(-(10 ** 15) - abs(int(team_key)) * 10 - k for k in range(5))
