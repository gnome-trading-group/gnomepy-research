"""
Pricing a bo3 series from the skill filter's state at a week's cutoff.

Targets carry no lineups (a series' own page is not known before it starts), so each team's five is inferred from
its recent maps and the roster moves since: the last five who played, minus players who left, plus joiners, padded
from the recent maps. The more a player is a guess, the more variance he adds.

The round-logit gap of a pre-veto map is Normal around the lineups' skill gap, with the variance the filter carries
plus a series-level noise shared by every map; each map adds its own noise. Given the shared part the three maps are
independent, so P(series) = p^2 (3 - 2p) integrated over the shared part.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e08.data import FIVE, HLTV
from gnomepy_research.sessions.cs2_comp.e08.filter import SkillFilter

RECENT_MAPS = 10
PLACEHOLDER_BASE = -1
LEAVING = {"leave", "transfer", "benched", "to_coach", "to_substitute", "loan"}
JOINING = {"join", "transfer", "activated", "loan_return"}

_X, _W = np.polynomial.hermite_e.hermegauss(21)
_W = _W / _W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)


@dataclass(frozen=True)
class SeriesParams:
    sigma_m: float = 0.30
    sigma_s: float = 0.10
    sigma_lu: float = 0.5
    stale_days: float = 1e9
    lineup_shift: float = 0.0
    slope: float = 1.0


class MapTable:
    def __init__(self, sigma_m: float):
        self.table = regulation_map_win(_GRID[:, None] + sigma_m * _X[None, :]) @ _W

    def bo3(self, mean: float, var: float) -> float:
        d = mean + np.sqrt(max(var, 0.0)) * _X
        p = np.interp(d, _GRID, self.table)
        return float((p * p * (3 - 2 * p)) @ _W)


class Moves:
    """Roster moves of players by team, for the lineup inference."""

    def __init__(self, moves: pd.DataFrame):
        m = moves[(moves.person_role == "player") & moves.event_date.notna()].copy()
        m["event_date"] = pd.to_datetime(m.event_date)
        self.leaving: dict[int, list] = collections.defaultdict(list)
        self.joining: dict[int, list] = collections.defaultdict(list)
        for r in m.itertuples():
            if r.kind in LEAVING and r.from_status == "active" and r.from_team_id:
                self.leaving[int(r.from_team_id)].append((r.event_date, int(r.person_id)))
            if r.kind in JOINING and r.to_status == "active" and r.to_team_id:
                self.joining[int(r.to_team_id)].append((r.event_date, int(r.person_id)))

    def left(self, team: int, after: pd.Timestamp, before: pd.Timestamp) -> set[int]:
        return {p for d, p in self.leaving.get(team, ()) if after < d < before}

    def joined(self, team: int, after: pd.Timestamp, before: pd.Timestamp) -> list[int]:
        return [p for d, p in sorted(self.joining.get(team, ()), reverse=True) if after < d < before]


def infer_lineup(team: int, recent: dict[int, collections.deque], moves: Moves, cutoff: pd.Timestamp,
                 stale_days: float = 1e9) -> tuple[list[int], list[float]]:
    """The five and, per player, the probability that he plays (1.0 for a regular, lower for a guess)."""
    hist = recent.get(team)
    if not hist:
        return [PLACEHOLDER_BASE - i for i in range(FIVE)], [0.0] * FIVE
    last_day, last_five = hist[-1]
    played = collections.Counter(p for _, five in hist for p in five)
    left = moves.left(team, last_day, cutoff)
    five = [p for p in last_five if p not in left]
    prob = {p: played[p] / len(hist) for p in five}
    for p in moves.joined(team, last_day, cutoff):
        if len(five) < FIVE and p not in five and p not in left:
            five.append(p)
            prob[p] = 0.5
    if len(five) < FIVE:
        for p, _ in sorted(played.items(), key=lambda kv: -kv[1]):
            if len(five) < FIVE and p not in five and p not in left:
                five.append(p)
                prob[p] = played[p] / len(hist) * 0.5
    while len(five) < FIVE:
        placeholder = PLACEHOLDER_BASE - len(five)
        five.append(placeholder)
        prob[placeholder] = 0.0
    fresh = math.exp(-max((cutoff - last_day).days, 0) / stale_days)
    return five, [prob[p] * fresh for p in five]


def price_series(f: SkillFilter, table: MapTable, sp: SeriesParams, t: int, five_a: list[int], prob_a: list[float],
                 feat_a: float, five_b: list[int], prob_b: list[float], feat_b: float) -> tuple[float, float, float]:
    """(q, mean gap, variance of the gap) for team A against team B, as of time t."""
    mu_a, ca = f.block(five_a, [feat_a] * FIVE, HLTV, 0, t)
    mu_b, cb = f.block(five_b, [feat_b] * FIVE, HLTV, 0, t)
    mu_a = mu_a - sp.lineup_shift * (1.0 - np.array(prob_a))
    mu_b = mu_b - sp.lineup_shift * (1.0 - np.array(prob_b))
    wa, wb = f.weights(mu_a), f.weights(mu_b)
    mean = float(wa @ mu_a - wb @ mu_b)
    var = float(wa @ ca @ wa + wb @ cb @ wb)
    guess = sum(1.0 - p for p in prob_a) + sum(1.0 - p for p in prob_b)
    var += sp.sigma_lu ** 2 * guess / FIVE ** 2 + sp.sigma_s ** 2
    return table.bo3(mean, var), mean, var


def calibrate(q: float, slope: float) -> float:
    q = min(max(q, 1e-4), 1 - 1e-4)
    return float(1 / (1 + np.exp(-slope * np.log(q / (1 - q)))))
