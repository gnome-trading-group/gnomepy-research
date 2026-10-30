"""Who will play: each team's expected five before the lineup is announced.

The bundle cut at a week's cutoff does not hold the series being predicted, so its lineup is unknown. For each team the
candidates are the players of its last HISTORY series plus anyone a roster move has just made a member. A logistic model
gives each candidate's probability of being in the five of a series that week, from how often and how recently he played
for the team, stand-in flags and roster moves. Prices are then averaged over lineups drawn from those probabilities; a
slot nobody fills becomes an unknown newcomer rated from the team's ranking."""
from __future__ import annotations

import bisect
import collections
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e09.maps import EPOCH

HISTORY = 12
JOIN_WINDOW_DAYS = 150
LINEUP_SIZE = 5
UNKNOWN_BASE = 10 ** 9
FEATURES = ("in_last1", "frac3", "frac6", "frac12", "log_team_gap", "log_since_played", "standin_last", "mv_member",
            "mv_bench", "mv_gone", "joined_no_games", "log_hist_len", "stability")

# Fitted on series from April 2024 to March 2026 (scripts in the competition scratch folder reproduce them).
COEF = np.array([1.0961, 0.7858, 0.0969, 0.4732, 0.6115, -0.9638, -0.2667, 3.3161, -1.7575, -1.3355, 2.4591, -0.2758, 0.4029])
INTERCEPT = -0.7671


@dataclass
class Entry:
    day: float
    match_id: str
    five: tuple[int, ...]
    standins: frozenset


class Moves:
    """Roster moves by player, newest last, for point-in-time lookups."""

    def __init__(self, moves: pd.DataFrame):
        m = moves[moves.person_role == "player"].copy()
        m["day"] = (pd.to_datetime(m.event_date) - EPOCH).dt.total_seconds() / 86400.0
        m = m.sort_values(["person_id", "day"], kind="stable")
        self.by_player: dict[int, tuple[list[float], list[tuple]]] = {}
        for pid, g in m.groupby("person_id"):
            self.by_player[int(pid)] = (g.day.tolist(), list(zip(g.kind, g.from_team_id.fillna(0).astype(int), g.to_team_id.fillna(0).astype(int),
                                                                   g.from_status, g.to_status, g.day)))
        self.members: dict[int, list[tuple[float, int]]] = collections.defaultdict(list)
        for pid, (days, evs) in self.by_player.items():
            for ev in evs:
                if ev[2] and ev[4] == "active":
                    self.members[ev[2]].append((ev[5], pid))
        for lst in self.members.values():
            lst.sort()

    def latest(self, player: int, before_day: float):
        rec = self.by_player.get(player)
        if rec is None:
            return None
        i = bisect.bisect_left(rec[0], before_day) - 1
        return rec[1][i] if i >= 0 else None

    def recent_joiners(self, team: int, before_day: float) -> list[int]:
        lst = self.members.get(team, [])
        lo = bisect.bisect_left(lst, (before_day - JOIN_WINDOW_DAYS, -1))
        hi = bisect.bisect_left(lst, (before_day, -1))
        return sorted({pid for _, pid in lst[lo:hi]})


def candidate_features(team: int, cutoff: float, hist: list[Entry], moves: Moves | None) -> tuple[list[int], np.ndarray]:
    """Candidates and their feature rows; `hist` is the team's series before `cutoff`, oldest first."""
    recent = hist[-HISTORY:][::-1]
    n = len(recent)
    seen: dict[int, list[int]] = {}
    for i, e in enumerate(recent):
        for p in e.five:
            seen.setdefault(p, []).append(i)
    joiners = [p for p in (moves.recent_joiners(team, cutoff) if moves else []) if p not in seen]
    cands = list(seen) + joiners
    if not cands:
        return [], np.zeros((0, len(FEATURES)))
    team_gap = math.log1p(max(cutoff - recent[0].day, 0.0)) if n else 0.0
    stab = 0.0
    if n >= 2:
        k = min(5, n - 1)
        stab = sum(1 for i in range(k) if set(recent[i].five) == set(recent[i + 1].five)) / k
    rows = []
    for p in cands:
        idx = seen.get(p, [])
        last_app = recent[idx[0]].day if idx else None
        ev = moves.latest(p, cutoff) if moves else None
        member = bench = gone = 0.0
        if ev is not None:
            kind, f_team, t_team, f_status, t_status, d = ev
            member = float(t_team == team and t_status == "active")
            bench = float(t_team == team and t_status in ("benched", "substitute"))
            gone = float((f_team == team and t_team != team and f_status == "active") or (t_team != team and last_app is not None and d > last_app))
        rows.append([
            float(0 in idx), sum(i < 3 for i in idx) / 3, sum(i < 6 for i in idx) / 6, len(idx) / max(n, 1),
            team_gap, math.log1p(max(cutoff - last_app, 0.0)) if last_app is not None else math.log1p(JOIN_WINDOW_DAYS),
            float(bool(idx) and p in recent[idx[0]].standins), member, bench, gone, float(not idx), math.log1p(n), stab])
    return cands, np.array(rows)


def play_probabilities(x: np.ndarray, coef: np.ndarray = COEF, intercept: float = INTERCEPT) -> np.ndarray:
    return 1 / (1 + np.exp(-(x @ coef + intercept)))


def sample_fives(team: int, cands: list[int], probs: np.ndarray, rng: np.random.Generator, n: int) -> list[tuple[int, ...]]:
    """n lineups: each candidate plays with his probability, extras beyond five drop the least likely, gaps are newcomers."""
    out = []
    unknown = [UNKNOWN_BASE + team * 5 + i for i in range(LINEUP_SIZE)]
    probs = np.asarray(probs, float)
    for _ in range(n):
        u = rng.random(len(cands))
        chosen = [(probs[i] + 1e-6 * rng.random(), cands[i]) for i in range(len(cands)) if u[i] < probs[i]]
        chosen.sort(reverse=True)
        five = [p for _, p in chosen[:LINEUP_SIZE]]
        five += unknown[: LINEUP_SIZE - len(five)]
        out.append(tuple(five))
    return out
