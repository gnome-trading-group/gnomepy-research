"""
Per-player Gaussian skill filter over the merged HLTV + FACEIT map stream.

This is models/cs2/hltv/player_skill.PlayerSkillFilter (an extended Kalman filter on round-logit skill, with
covariances between players who shared a map) with days as integers, plus what a second source needs:

- FACEIT maps are their own measurement: their binomial gradient and curvature are scaled by w_faceit, i.e. extra
  overdispersion for rounds played in online leagues, scrim-like qualifiers and lower divisions.
- A player first seen on FACEIT starts from his own prior: offset a_faceit, plus slope s_faceit on his five's mean
  FACEIT skill level below 10, plus the HLTV ranking prior when FACEIT linked his team to an HLTV team.
"""
from __future__ import annotations

import bisect
import collections
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import PlayerSkillConfig, map_win_prob
from gnomepy_research.sessions.cs2_comp.e11.stream import FACEIT, MapStream, day_number

TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FilterConfig:
    base: PlayerSkillConfig = PlayerSkillConfig()
    w_faceit: float = 1.0
    a_faceit: float = 0.0
    s_faceit: float = 0.0
    sigma_series: float = 0.0
    # Which FACEIT fives may stand for a team's latest five: 0 none, 1 only fully linked ones, 2 any.
    faceit_lineups: int = 1


def bo3(p: float) -> float:
    return p * p * (3 - 2 * p)


class SkillFilter:
    def __init__(self, cfg: FilterConfig, rankings: pd.DataFrame):
        self.cfg = cfg
        self.c = cfg.base
        self._mu: dict[int, float] = {}
        self._var: dict[int, float] = {}
        self._seen: dict[int, int] = {}
        self._cov: dict[tuple[int, int], float] = {}
        self.games: collections.Counter = collections.Counter()
        self._five_days: dict[frozenset, collections.deque] = collections.defaultdict(collections.deque)
        self._rank_days: list[int] = []
        self._rank_points: list[dict[int, float]] = []
        r = rankings.assign(date=pd.to_datetime(rankings["date"]))
        for day, g in r.groupby("date", sort=True):
            self._rank_days.append(day_number(day))
            self._rank_points.append(dict(zip(g["team_id"].astype(int), g["points"].astype(float))))

    def _points(self, team_id: int, day: int) -> float:
        i = bisect.bisect_left(self._rank_days, day) - 1
        return self._rank_points[i].get(team_id, 0.0) if i >= 0 else 0.0

    def _prior(self, team_id: int, day: int, source: int, skill: float) -> float:
        mu = self.c.a + self.c.b * math.log1p(self._points(team_id, day)) if team_id >= 0 else self.c.a
        if source == FACEIT:
            mu += self.cfg.a_faceit - (self.cfg.s_faceit * (10.0 - skill) if not math.isnan(skill) else 0.0)
        return mu

    def _state(self, player: int, team_id: int, day: int, source: int, skill: float) -> tuple[float, float]:
        if player not in self._mu:
            return self._prior(team_id, day, source, skill), self.c.sigma0 ** 2
        idle = max(day - self._seen[player], 0)
        return self._mu[player], self._var[player] + self.c.tau ** 2 * idle

    def _five_maps(self, five: tuple[int, ...], day: int) -> float:
        q = self._five_days.get(frozenset(five))
        if not q:
            return 0.0
        cutoff = day - self.c.chem_days
        return float(sum(1 for d in q if d >= cutoff))

    def _weights(self, mus: list[float]) -> list[float]:
        n = len(mus)
        # Near-ties go to the first listed player: a raw argmax lets machine-dependent rounding bits pick the leader.
        top = max(mus)
        best = next(i for i, m in enumerate(mus) if m >= top - TIE_TOLERANCE)
        return [1 / n + self.c.lam * ((1.0 if i == best else 0.0) - 1 / n) for i in range(n)]

    def _block(self, players: list[int], variances: list[float]) -> np.ndarray:
        cov = np.diag(variances)
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p, q = (players[i], players[j]) if players[i] < players[j] else (players[j], players[i])
                cov[i, j] = cov[j, i] = self._cov.get((p, q), 0.0)
        return cov

    def _lineup(self, players: tuple[int, ...], team_id: int, day: int, source: int, skill: float):
        states = [self._state(p, team_id, day, source, skill) for p in players]
        mus = [s[0] for s in states]
        weights = self._weights(mus)
        five_maps = self._five_maps(players, day) if len(players) == 5 else 0.0
        kappa = self.c.kappa_max * (1 - math.exp(-five_maps / self.c.c0))
        total = sum(w * m for w, m in zip(weights, mus)) + kappa
        return list(players), mus, [s[1] for s in states], weights, total

    def gap(self, players_a, team_a, players_b, team_b, day, source=0, skill_a=math.nan, skill_b=math.nan):
        pa, mua, va, wa, ta = self._lineup(players_a, team_a, day, source, skill_a)
        pb, mub, vb, wb, tb = self._lineup(players_b, team_b, day, source, skill_b)
        w = np.array(wa + [-x for x in wb])
        cov = self._block(pa + pb, va + vb)
        return ta - tb, float(w @ cov @ w), cov, w, pa + pb, mua + mub

    def price(self, players_a, team_a, players_b, team_b, day) -> tuple[float, float]:
        """P(A wins a bo3) and the round-logit gap, pre-veto, from the current state."""
        d, s2, *_ = self.gap(players_a, team_a, players_b, team_b, day)
        p_map = map_win_prob(d, s2 + self.cfg.sigma_series ** 2)
        return bo3(p_map), d

    def observe(self, day: int, team_a: int, team_b: int, players_a, players_b, rounds_a: int, rounds_b: int,
                source: int, skill_a: float, skill_b: float) -> float | None:
        """Update on one map; returns the pre-map probability that A wins the map (None if skipped)."""
        if not players_a or not players_b:
            return None
        d, s2, cov, w, players, mus = self.gap(players_a, team_a, players_b, team_b, day, source, skill_a, skill_b)
        if len(set(players)) != len(players):
            return None
        p_pre = map_win_prob(d, s2)
        n = rounds_a + rounds_b
        pi = 1 / (1 + math.exp(-d))
        inflation = (1 + (n - 1) * self.c.rho) / (self.cfg.w_faceit if source == FACEIT else 1.0)
        grad, curv = (rounds_a - n * pi) / inflation, n * pi * (1 - pi) / inflation
        cw = cov @ w
        denom = 1 + curv * s2
        new_mus = np.array(mus) + cw * grad / denom
        cov = cov - np.outer(cw, cw) * curv / denom
        for i, p in enumerate(players):
            self._mu[p] = float(new_mus[i])
            self._var[p] = max(1e-6, float(cov[i, i]))
            self._seen[p] = day
            self.games[p] += 1
            for j in range(i + 1, len(players)):
                q = players[j]
                self._cov[(p, q) if p < q else (q, p)] = float(cov[i, j])
        cutoff = day - self.c.chem_days
        for five in (players_a, players_b):
            if len(five) == 5:
                dq = self._five_days[frozenset(five)]
                dq.append(day)
                while dq and dq[0] < cutoff:
                    dq.popleft()
        return p_pre


def run_stream(f: SkillFilter, stream: MapStream, start: int, stop_time: np.datetime64,
               lineups: dict[int, tuple[np.datetime64, tuple[int, ...]]], lineup_sources: tuple[int, ...],
               log: list | None = None) -> int:
    """Advance the filter through stream[start:] while maps are available before stop_time; returns the next index.
    Each team's latest five (by availability) is kept in `lineups`, from the sources in lineup_sources."""
    i = start
    n = len(stream)
    while i < n and stream.avail[i] < stop_time:
        src = int(stream.source[i])
        pa, pb = stream.players_a[i], stream.players_b[i]
        p = f.observe(int(stream.day[i]), int(stream.team_a[i]), int(stream.team_b[i]), pa, pb,
                      int(stream.rounds_a[i]), int(stream.rounds_b[i]), src, float(stream.skill_a[i]),
                      float(stream.skill_b[i]))
        if log is not None and p is not None:
            log.append((i, p))
        if src in lineup_sources:
            for team, five in ((int(stream.team_a[i]), pa), (int(stream.team_b[i]), pb)):
                if team >= 0 and len(five) == 5 and (src != FACEIT or f.cfg.faceit_lineups == 2 or min(five) > 0):
                    lineups[team] = (stream.avail[i], five)
        i += 1
    return i
