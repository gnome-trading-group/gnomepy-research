"""
Sequential rating engines in round-logit units, read at weekly cutoffs.

Each engine consumes decided maps in time order and prices a series from the fives announced for it. Three engines
see the same maps from different angles:

- `TeamMargin`: one rating per team id, moved by the round share of each map (margin Elo). It knows the team as an
  organisation: coaching, map pool, chemistry, whoever is in the five.
- `PlayerMargin`: one rating per player, a five is the mean of its players; a newcomer starts at the mean of the
  teammates who are known, else at a prior from the team's ranking points. It follows players across rosters.
- `PlayerKalman`: the repo's fitted extended Kalman filter (`PlayerSkillFilter`), which also carries uncertainty and
  covariance between teammates, so a five's gap has a variance.

A snapshot at a cutoff only ever reads state built from maps before it; the walk-forward driver in `features.py`
guarantees the order.
"""
from __future__ import annotations

import bisect
import collections
import math

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import PlayerSkillConfig, PlayerSkillFilter, map_win_prob, regulation_map_win

UNRANKED = 300.0
LOG_UNRANKED = math.log(UNRANKED)
ROUNDS = 24.0


def bo3(p_map: np.ndarray | float) -> np.ndarray | float:
    return p_map * p_map * (3 - 2 * p_map)


def series_prob(gap: float, var: float = 0.0) -> float:
    p = map_win_prob(gap, var) if var > 0 else float(regulation_map_win(np.array([gap]))[0])
    return float(bo3(p))


class Rankings:
    """HLTV ranking points and rank of a team as of a day, from rankings published before it."""

    def __init__(self, rankings: pd.DataFrame):
        self.dates: list[pd.Timestamp] = []
        self.points: list[dict[int, float]] = []
        self.rank: list[dict[int, float]] = []
        for day, g in rankings.groupby("date", sort=True):
            self.dates.append(pd.Timestamp(day))
            self.points.append(dict(zip(g.team_id.astype(int), g.points.astype(float))))
            self.rank.append(dict(zip(g.team_id.astype(int), g["rank"].astype(float))))

    def _idx(self, day: pd.Timestamp) -> int:
        return bisect.bisect_left(self.dates, day) - 1

    def points_of(self, team: int, day: pd.Timestamp) -> float:
        i = self._idx(day)
        return self.points[i].get(team, 0.0) if i >= 0 else 0.0

    def rank_of(self, team: int, day: pd.Timestamp, max_age_days: int = 120) -> float:
        i = self._idx(day)
        if i < 0 or (day - self.dates[i]).days > max_age_days:
            return UNRANKED
        return self.rank[i].get(team, UNRANKED)


class TeamMargin:
    def __init__(self, rankings: Rankings, k: float = 0.25, k_new: float = 1.0, new_maps: float = 10.0,
                 a: float = -0.11, b: float = 0.147):
        self.rk = rankings
        self.k, self.k_new, self.new_maps, self.a, self.b = k, k_new, new_maps, a, b
        self.r: dict[int, float] = {}
        self.n: collections.Counter = collections.Counter()

    def rating(self, team: int, day: pd.Timestamp) -> float:
        if team in self.r:
            return self.r[team]
        return self.a + self.b * math.log1p(self.rk.points_of(team, day))

    def gap(self, team_a: int, team_b: int, day: pd.Timestamp) -> float:
        return self.rating(team_a, day) - self.rating(team_b, day)

    def observe(self, day: pd.Timestamp, team_a: int, team_b: int, rounds_a: int, rounds_b: int) -> None:
        ra, rb = self.rating(team_a, day), self.rating(team_b, day)
        n = rounds_a + rounds_b
        p = 1 / (1 + math.exp(-(ra - rb)))
        g = (rounds_a - n * p) / ROUNDS
        ka = self.k * (1 + self.k_new * math.exp(-self.n[team_a] / self.new_maps))
        kb = self.k * (1 + self.k_new * math.exp(-self.n[team_b] / self.new_maps))
        self.r[team_a] = ra + ka * g
        self.r[team_b] = rb - kb * g
        self.n[team_a] += 1
        self.n[team_b] += 1


class PlayerMargin:
    def __init__(self, rankings: Rankings, k: float = 0.25, k_new: float = 1.0, new_maps: float = 10.0,
                 a: float = -0.11, b: float = 0.147):
        self.rk = rankings
        self.k, self.k_new, self.new_maps, self.a, self.b = k, k_new, new_maps, a, b
        self.r: dict[int, float] = {}
        self.n: collections.Counter = collections.Counter()

    def five(self, players: tuple[int, ...], team: int, day: pd.Timestamp) -> tuple[list[float], int]:
        known = [self.r[p] for p in players if p in self.r]
        prior = float(np.mean(known)) if len(known) >= 2 else self.a + self.b * math.log1p(self.rk.points_of(team, day))
        return [self.r.get(p, prior) for p in players], len(players) - len(known)

    def strength(self, players: tuple[int, ...], team: int, day: pd.Timestamp) -> float:
        return float(np.mean(self.five(players, team, day)[0]))

    def observe(self, day: pd.Timestamp, team_a: int, team_b: int, five_a: tuple[int, ...], five_b: tuple[int, ...],
                rounds_a: int, rounds_b: int) -> None:
        ma, _ = self.five(five_a, team_a, day)
        mb, _ = self.five(five_b, team_b, day)
        n = rounds_a + rounds_b
        p = 1 / (1 + math.exp(-(float(np.mean(ma)) - float(np.mean(mb)))))
        g = (rounds_a - n * p) / ROUNDS
        for players, mus, sign in ((five_a, ma, 1.0), (five_b, mb, -1.0)):
            for pl, mu in zip(players, mus):
                k = self.k * (1 + self.k_new * math.exp(-self.n[pl] / self.new_maps))
                self.r[pl] = mu + sign * k * g
                self.n[pl] += 1


class PlayerKalman:
    def __init__(self, rankings: pd.DataFrame, cfg: PlayerSkillConfig | None = None):
        self.f = PlayerSkillFilter(cfg, rankings)

    def observe(self, day: pd.Timestamp, team_a: int, team_b: int, five_a: tuple[int, ...], five_b: tuple[int, ...],
                rounds_a: int, rounds_b: int) -> None:
        self.f.observe_map(day, team_a, team_b, five_a, five_b, rounds_a, rounds_b)

    def price(self, five_a: tuple[int, ...], five_b: tuple[int, ...], team_a: int, team_b: int,
              standins_a: float, standins_b: float, day: pd.Timestamp) -> dict:
        a = self.f.lineup(five_a, team_a, standins_a, day)
        b = self.f.lineup(five_b, team_b, standins_b, day)
        d, s2, _, _ = self.f.gap(a, b)
        return {"gap": d, "sd": math.sqrt(max(s2, 0.0)), "chem_a": a.kappa, "chem_b": b.kappa,
                "sd_a": self.f.lineup_sd(a), "sd_b": self.f.lineup_sd(b),
                "best_a": max(a.mus), "best_b": max(b.mus), "worst_a": min(a.mus), "worst_b": min(b.mus),
                "games_a": [self.f.games[p] for p in five_a], "games_b": [self.f.games[p] for p in five_b]}
