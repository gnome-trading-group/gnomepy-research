"""Per-player Gaussian skill filter over maps from two sources.

A copy of models/cs2/hltv/player_skill.PlayerSkillFilter with three changes: a map's gradient and curvature are scaled
by a per-source weight, a player first seen on FACEIT starts from the FACEIT prior (its own offset and spread), and the
rank-points lookup takes its own date so a prediction can read rankings only up to the week's cutoff.
"""
from __future__ import annotations

import bisect
import collections
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import TIE_TOLERANCE, PlayerSkillConfig, map_win_prob

HLTV, FACEIT = 0, 1
MIN_EIGENVALUE = 1e-6


@dataclass(frozen=True)
class SourceConfig:
    """FACEIT measurement settings on top of the fitted HLTV filter settings."""

    faceit_weight: float = 1.0
    faceit_prior_offset: float = -0.5
    faceit_sigma0: float = 0.5


@dataclass
class Lineup:
    players: list[int]
    mus: list[float]
    variances: list[float]
    weights: list[float]
    strength: float
    total: float


class SkillFilter:
    def __init__(self, cfg: PlayerSkillConfig, src: SourceConfig, rankings: pd.DataFrame | None, use_faceit: bool):
        self.cfg, self.src, self.use_faceit = cfg, src, use_faceit
        self._mu: dict[int, float] = {}
        self._var: dict[int, float] = {}
        self._seen: dict[int, pd.Timestamp] = {}
        self._cov: dict[tuple[int, int], float] = {}
        self.games: collections.Counter = collections.Counter()
        self.faceit_games: collections.Counter = collections.Counter()
        self._five_days: dict[frozenset, collections.deque] = collections.defaultdict(collections.deque)
        self._rank_dates: list[pd.Timestamp] = []
        self._rank_points: list[dict[int, float]] = []
        if rankings is not None and len(rankings):
            r = rankings.assign(date=pd.to_datetime(rankings["date"]))
            for day, g in r.groupby("date", sort=True):
                self._rank_dates.append(pd.Timestamp(day))
                self._rank_points.append(dict(zip(g["team_id"].astype(int), g["points"].astype(float))))

    def points(self, team_id: int | None, day: pd.Timestamp) -> float:
        """HLTV ranking points from the latest ranking dated before `day`, 0 when unranked or not an HLTV team."""
        if team_id is None:
            return 0.0
        i = bisect.bisect_left(self._rank_dates, day) - 1
        return self._rank_points[i].get(team_id, 0.0) if i >= 0 else 0.0

    def _state(self, player: int, team_id: int | None, day: pd.Timestamp, rank_day: pd.Timestamp, source: int) -> tuple[float, float]:
        if player not in self._mu:
            mean = self.cfg.a + self.cfg.b * math.log1p(self.points(team_id, rank_day))
            if source == FACEIT:
                return mean + self.src.faceit_prior_offset, self.src.faceit_sigma0 ** 2
            return mean, self.cfg.sigma0 ** 2
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
        # Pairwise covariances are updated locally and at different times, so the assembled block need not be
        # positive semi-definite; an indefinite block makes the gap variance negative and the update explode.
        vals, vecs = np.linalg.eigh(cov)
        if vals[0] < MIN_EIGENVALUE:
            cov = (vecs * np.maximum(vals, MIN_EIGENVALUE)) @ vecs.T
        return cov

    def lineup(self, players: tuple[int, ...], team_id: int | None, day: pd.Timestamp, rank_day: pd.Timestamp,
               source: int = HLTV) -> Lineup:
        states = [self._state(p, team_id, day, rank_day, source) for p in players]
        mus = [s[0] for s in states]
        weights = self._weights(mus)
        five_maps = self._five_maps(players, day) if len(players) == 5 else 0.0
        kappa = self.cfg.kappa_max * (1 - math.exp(-five_maps / self.cfg.c0))
        strength = sum(w * m for w, m in zip(weights, mus))
        return Lineup(list(players), mus, [s[1] for s in states], weights, strength, strength + kappa)

    def gap(self, a: Lineup, b: Lineup) -> tuple[float, float, np.ndarray, np.ndarray]:
        w = np.array(a.weights + [-x for x in b.weights])
        cov = self._block(a.players + b.players, a.variances + b.variances)
        return a.total - b.total, float(w @ cov @ w), cov, w

    def series_price(self, a: Lineup, b: Lineup) -> tuple[float, float, float]:
        """Round-logit gap mean and variance, and P(A wins a bo3) from them."""
        d, s2, _, _ = self.gap(a, b)
        p = map_win_prob(d, s2)
        return d, s2, p * p * (3 - 2 * p)

    def observe_map(self, day: pd.Timestamp, source: int, team_a: int | None, team_b: int | None,
                    players_a: tuple[int, ...], players_b: tuple[int, ...], rounds_a: int, rounds_b: int) -> None:
        if source == FACEIT and not self.use_faceit:
            return
        if not players_a or not players_b:
            return
        a = self.lineup(players_a, team_a, day, day, source)
        b = self.lineup(players_b, team_b, day, day, source)
        players = a.players + b.players
        if len(set(players)) != len(players):
            return
        d, s2, cov, w = self.gap(a, b)
        n = rounds_a + rounds_b
        pi = 1 / (1 + math.exp(-d))
        weight = self.src.faceit_weight if source == FACEIT else 1.0
        inflation = (1 + (n - 1) * self.cfg.rho) / weight
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
            if source == FACEIT:
                self.faceit_games[p] += 1
            for j in range(i + 1, len(players)):
                self._cov[tuple(sorted((p, players[j])))] = float(cov[i, j])
        cutoff = day - pd.Timedelta(days=self.cfg.chem_days)
        for five in (players_a, players_b):
            if len(five) == 5:
                q = self._five_days[frozenset(five)]
                q.append(day)
                while q and q[0] < cutoff:
                    q.popleft()
