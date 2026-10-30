"""Pricing a bo3 series before the veto from a fitted skill model, and the announced lineups it needs.

The announced lineup is not in a point-in-time bundle, so each team is assumed to field the players it fielded most
recently: those of its last LINEUP_SERIES series, the latest counting most."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.e05.data import TEAM_SIZE
from gnomepy_research.sessions.cs2_comp.e05 import kernel as K
from gnomepy_research.sessions.cs2_comp.e05.kernel import PAIR_SHIFT
from gnomepy_research.sessions.cs2_comp.e05.stream import Filter, map_table, ranking_points

UNKNOWN = -1
MIN_LINKED = 30
MIN_RESIDUAL = 0.005
UNINFORMATIVE = 1e6
LINEUP_SERIES = 5
LINEUP_DECAY = 0.6
DAYS_PER_MONTH = 30.44
HORIZON_DAYS = 3.5
_X, _W = np.polynomial.hermite_e.hermegauss(21)
_W = _W / _W.sum()
_GRID = K.GRID_LO + K.GRID_STEP * np.arange(int(round(-2 * K.GRID_LO / K.GRID_STEP)) + 2)


@dataclass(frozen=True)
class PriceConfig:
    sigma_series: float = 0.1
    scale: float = 1.0
    var_mult: float = 1.0
    lineup_series: int = 3
    lineup_decay: float = 0.5
    use_moves: bool = True
    faceit_lineups: bool = False
    standin_weight: float = 1.0
    coh_window: int = 60
    coh_sigma: float = 0.0
    coh_n0: float = 3.0
    link_inflate: float = 0.0


def roster_changes(moves: pd.DataFrame) -> pd.DataFrame:
    """Player moves that put a player on or off a team's active five: (date, team, player, +1 joined / -1 left)."""
    m = moves[moves.person_role == "player"]
    leave = m[m.from_team_id.gt(0) & m.from_status.isin(["active", "starter"]) & (m.from_team_id != m.to_team_id) | (m.kind == "benched")]
    join = m[m.to_team_id.gt(0) & m.to_status.isin(["active", "starter"]) & (m.from_team_id != m.to_team_id) | (m.kind == "activated")]
    return pd.concat([
        pd.DataFrame({"date": pd.to_datetime(leave.event_date), "team": leave.from_team_id, "player": leave.person_id, "delta": -1}),
        pd.DataFrame({"date": pd.to_datetime(join.event_date), "team": join.to_team_id, "player": join.person_id, "delta": 1}),
    ], ignore_index=True)


def apply_roster_changes(lineup: list[tuple[int, float]], changes: pd.DataFrame, since: pd.Timestamp) -> list[tuple[int, float]]:
    """Drop players who left after the team's last series and add those who joined, then rescale to five expected players."""
    weight = dict(lineup)
    for c in changes[changes.date > since].sort_values("date", kind="stable").itertuples():
        if c.delta < 0:
            weight.pop(int(c.player), None)
        else:
            weight[int(c.player)] = 1.0
    total = sum(weight.values())
    if total > TEAM_SIZE:
        weight = {p: w * TEAM_SIZE / total for p, w in weight.items()}
    elif total < TEAM_SIZE - 1e-9:
        weight[UNKNOWN] = TEAM_SIZE - total
    return sorted(weight.items(), key=lambda pw: (-pw[1], pw[0]))


def infer_lineups(history: pd.DataFrame, teams: set[int] | None = None, series: int = LINEUP_SERIES,
                  decay: float = LINEUP_DECAY, changes: pd.DataFrame | None = None,
                  extra: pd.DataFrame | None = None, standins: frozenset | None = None,
                  standin_weight: float = 1.0) -> dict[int, list[tuple[int, float]]]:
    """team id -> (player, probability of playing): the share of the team's maps in its last series each player took part in,
    recent series counting more. The probabilities of a team sum to five."""
    h = history[~showmatch_mask(history)]
    if teams is not None:
        h = h[h.team_a_id.isin(teams) | h.team_b_id.isin(teams)]
    h = h[h.team_a_player_ids.map(lambda x: x is not None and len(x) == TEAM_SIZE)
          & h.team_b_player_ids.map(lambda x: x is not None and len(x) == TEAM_SIZE)]
    h = h.assign(time=pd.to_datetime(h.match_time.fillna(h.match_date), utc=True))
    rows = pd.concat([
        pd.DataFrame({"team": h.team_a_id, "match": h.match_id, "time": h.time, "players": h.team_a_player_ids}),
        pd.DataFrame({"team": h.team_b_id, "match": h.match_id, "time": h.time, "players": h.team_b_player_ids}),
    ]).dropna(subset=["team"])
    if extra is not None:
        rows = pd.concat([rows, extra[extra.team.isin(teams)] if teams is not None else extra], ignore_index=True)
    if teams is not None:
        rows = rows[rows.team.isin(teams)]
    rows = rows.sort_values(["team", "time", "match"], ascending=[True, False, False])
    out: dict[int, list[tuple[int, float]]] = {}
    for team, g in rows.groupby("team", sort=False):
        weight: dict[int, float] = {}
        order = {m: i for i, m in enumerate(dict.fromkeys(g.match))}
        counts = g.groupby("match").size()
        for m, players in zip(g.match, g.players):
            if order[m] >= series:
                continue
            for p in players:
                share = standin_weight if standins is not None and (m, int(p)) in standins else 1.0
                weight[int(p)] = weight.get(int(p), 0.0) + share * decay ** order[m] / counts[m]
        total = sum(decay ** i for i in range(min(series, len(order))))
        lineup = sorted(((p, w / total) for p, w in weight.items()), key=lambda pw: (-pw[1], pw[0]))
        if changes is not None:
            lineup = apply_roster_changes(lineup, changes[changes.team == team], g.time.max().tz_convert(None))
        out[int(team)] = lineup
    return out


def calibrate_link(flt: Filter) -> tuple[float, float, float]:
    """Affine map from a player's FACEIT skill to their HLTV skill and its residual variance, read off the current state of
    the players with the most settled HLTV skills. A weak relation comes out as a large residual and so a small weight."""
    link = flt.stream.link
    hltv = np.flatnonzero(link >= 0)
    face = link[hltv]
    ok = (flt.seen[hltv] >= 0) & (flt.seen[face] >= 0)
    hltv, face = hltv[ok], face[ok]
    if len(hltv) < MIN_LINKED:
        return 0.0, 0.0, UNINFORMATIVE
    settled = flt.var[hltv] <= np.quantile(flt.var[hltv], 0.5)
    hltv, face = hltv[settled], face[settled]
    x, y = flt.mu[face], flt.mu[hltv]
    beta = max(0.0, np.cov(x, y)[0, 1] / x.var(ddof=1))
    alpha = y.mean() - beta * x.mean()
    residual = ((y - alpha - beta * x) ** 2).mean() - flt.var[hltv].mean() - beta ** 2 * flt.var[face].mean()
    return float(alpha), float(beta), float(max(residual, MIN_RESIDUAL))


class SeriesPricer:
    """Prices series from the state of a Filter that has advanced to the week's cutoff."""

    def __init__(self, flt: Filter, rankings: pd.DataFrame, cutoff: pd.Timestamp, cfg: PriceConfig):
        self.flt, self.rankings, self.cutoff, self.cfg = flt, rankings, cutoff, cfg
        self.table = map_table(flt.cfg.sigma_map.get("hltv", flt.cfg.sigma_map_default))
        self.now = (np.datetime64(cutoff.date()) - np.datetime64("1970-01-01")).astype(np.int64) + HORIZON_DAYS
        self.hltv_lam = float(flt.src_lam[0])
        self.alpha, self.beta, self.residual = calibrate_link(flt) if cfg.link_inflate > 0.0 else (0.0, 0.0, UNINFORMATIVE)
        day = flt.stream.day
        lo, hi = np.searchsorted(day, [self.now - HORIZON_DAYS - cfg.coh_window, self.now - HORIZON_DAYS])
        self.window_a, self.window_b = flt.stream.side_a[lo:hi], flt.stream.side_b[lo:hi]

    def _with_faceit(self, i: int, mean: float, var: float) -> tuple[float, float]:
        """Condition one HLTV skill on the same person's FACEIT skill, read as a noisy measurement of it."""
        f = self.flt
        q = int(f.stream.link[i])
        if self.beta <= 0.0 or q < 0 or f.seen[q] < 0:
            return mean, var
        m2 = self.alpha + self.beta * f.mu[q]
        v2 = self.cfg.link_inflate * self.residual + self.beta ** 2 * (f.var[q] + f.cfg.tau ** 2 * max(0.0, self.now - float(f.seen[q])))
        gain = var / (var + v2)
        return mean + gain * (m2 - mean), var * (1.0 - gain)

    def _state(self, players: list[int], team: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, c = self.flt, self.flt.cfg
        keys = f.stream.keys
        mean, var = np.zeros(len(players)), np.zeros(len(players))
        points = float(ranking_points(self.rankings, np.array([team]), np.array([self.now - HORIZON_DAYS], dtype=np.int64))[0])
        compact = []
        for k, p in enumerate(players):
            i = int(np.searchsorted(keys, p))
            found = i < len(keys) and keys[i] == p
            if found and f.seen[i] >= 0:
                mean[k] = f.mu[i]
                var[k] = f.var[i] + c.tau ** 2 * max(0.0, self.now - float(f.seen[i]))
                mean[k], var[k] = self._with_faceit(i, mean[k], var[k])
                compact.append(i)
                continue
            mean[k], var[k] = c.hltv_a + c.hltv_b * np.log1p(points), c.hltv_sigma0 ** 2
            if found:
                mean[k], var[k] = self._with_faceit(i, mean[k], var[k])
            compact.append(-1)
        block = np.diag(var)
        for j in range(len(players)):
            for k in range(j + 1, len(players)):
                if compact[j] >= 0 and compact[k] >= 0:
                    block[j, k] = block[k, j] = f.cov.get(_pair_key(compact[j], compact[k]), 0.0)
        return mean, block, np.asarray(compact)

    def lineup(self, candidates: list[tuple[int, float]], team: int) -> tuple[float, float]:
        """Mean and variance of a lineup strength when each candidate plays with its probability."""
        if not candidates:
            candidates = [(UNKNOWN, 1.0)] * TEAM_SIZE
        players = [p for p, _ in candidates]
        presence = np.array([q for _, q in candidates])
        mean, block, _ = self._state(players, team)
        order = np.argsort(-mean, kind="stable")
        absent = np.cumprod(np.concatenate([[1.0], 1.0 - presence[order]]))[:-1]
        best = np.empty(len(players))
        best[order] = presence[order] * absent
        best = best / best.sum()
        w = presence / presence.sum() + self.flt.cfg.lam_best * (best - presence / presence.sum())
        return float(w @ mean), float(w @ block @ w)

    def maps_together(self, candidates: list[tuple[int, float]]) -> int:
        """Maps of the last coh_window days (either source) in which at least four of the expected five were on one side."""
        keys = self.flt.stream.keys
        top = [p for p, _ in sorted(candidates, key=lambda c: -c[1])[:TEAM_SIZE]]
        index = np.searchsorted(keys, top)
        known = index[(index < len(keys)) & (keys[np.minimum(index, len(keys) - 1)] == top)]
        if len(known) < 4 or not len(self.window_a):
            return 0
        a = np.isin(self.window_a, known).sum(axis=1) >= 4
        b = np.isin(self.window_b, known).sum(axis=1) >= 4
        return int((a | b).sum())

    def components(self, team_a: int, lineup_a: list[tuple[int, float]], team_b: int, lineup_b: list[tuple[int, float]]) -> np.ndarray:
        """Everything a series price needs from the fitted state: both lineups' mean and variance and their maps together."""
        ma, va = self.lineup(lineup_a, team_a)
        mb, vb = self.lineup(lineup_b, team_b)
        return np.array([ma, va, mb, vb, self.maps_together(lineup_a), self.maps_together(lineup_b)])

    def price(self, team_a: int, lineup_a: list[tuple[int, float]], team_b: int, lineup_b: list[tuple[int, float]]) -> float:
        return float(series_probability(self.table, self.hltv_lam, self.components(team_a, lineup_a, team_b, lineup_b)[None, :], self.cfg)[0])


def series_probability(table: np.ndarray, lam: float, parts: np.ndarray, cfg: PriceConfig) -> np.ndarray:
    """Bo3 series win probability for rows of (mean_a, var_a, mean_b, var_b, together_a, together_b)."""
    mu = cfg.scale * lam * (parts[:, 0] - parts[:, 2])
    extra = cfg.coh_sigma ** 2 * (1.0 / (1.0 + parts[:, 4] / cfg.coh_n0) + 1.0 / (1.0 + parts[:, 5] / cfg.coh_n0))
    sd = np.sqrt(cfg.var_mult * (parts[:, 1] + parts[:, 3]) * (cfg.scale * lam) ** 2 + cfg.sigma_series ** 2 + extra)
    p = np.interp(mu[:, None] + sd[:, None] * _X[None, :], _GRID, table)
    return (p * p * (3 - 2 * p)) @ _W


def _pair_key(p: int, q: int) -> int:
    return (p << PAIR_SHIFT) | q if p < q else (q << PAIR_SHIFT) | p
