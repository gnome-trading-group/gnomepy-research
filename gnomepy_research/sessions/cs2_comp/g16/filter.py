"""
The player-skill Kalman filter with two additions.

HLTV map ratings as a second measurement. Round results only ever see a five's mean, so a fixed five reveals its sum
and nothing about who carries it. Each player's HLTV rating relative to the five's mean on that map is a
linear-Gaussian reading of exactly the part the rounds cannot see:

    rating_p - mean(rating_five) ~ N(gamma * (s_p - mean(s_five)), sd^2)

Division-dependent newcomer priors. Each player gets a fifth of a map's information, so a five's level stays close to
its newcomers' prior for many maps. FACEIT brings most newcomers, from ESEA Entry to Advanced, and the competition's
division is the best guess at their level before any map is played.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import PlayerSkillConfig, PlayerSkillFilter

FIVE = 5
_CENTER = np.eye(FIVE) - np.full((FIVE, FIVE), 1.0 / FIVE)


class SkillFilter(PlayerSkillFilter):
    def __init__(self, cfg: PlayerSkillConfig, rankings: pd.DataFrame, rating_gamma: float = 0.0, rating_sd: float = 0.31,
                 division_offsets: dict[str, float] | None = None):
        super().__init__(cfg, rankings)
        self.rating_gamma = rating_gamma
        self.rating_sd = rating_sd
        self.division_offsets = division_offsets or {}
        self._division: str | None = None

    def _state(self, player: int, team_id: int, day: pd.Timestamp) -> tuple[float, float]:
        if player not in self._mu and self._division is not None:
            offset = self.division_offsets.get(self._division, 0.0)
            return self.cfg.a + self.cfg.b * math.log1p(self._points(team_id, day)) + offset, self.cfg.sigma0 ** 2
        return super()._state(player, team_id, day)

    def observe(self, day: pd.Timestamp, team_a: int, team_b: int, five_a: tuple[int, ...], five_b: tuple[int, ...],
                rounds_a: float, rounds_b: float, division: str | None) -> None:
        self._division = division
        try:
            self.observe_map(day, team_a, team_b, five_a, five_b, rounds_a, rounds_b)
        finally:
            self._division = None

    def observe_ratings(self, day: pd.Timestamp, team_id: int, players: tuple[int, ...], ratings: np.ndarray) -> None:
        if not self.rating_gamma or len(players) != FIVE or len(set(players)) != FIVE:
            return
        states = [self._state(p, team_id, day) for p in players]
        mu = np.array([s[0] for s in states])
        cov = self._block(list(players), [s[1] for s in states])
        H = self.rating_gamma * _CENTER
        y = ratings - ratings.mean()
        S = H @ cov @ H.T + self.rating_sd ** 2 * np.eye(FIVE)
        K = np.linalg.solve(S, H @ cov).T
        mu = mu + K @ (y - H @ mu)
        cov = cov - K @ H @ cov
        for i, p in enumerate(players):
            self._mu[p] = float(mu[i])
            self._var[p] = max(1e-6, float(cov[i, i]))
            self._seen[p] = day
            for j in range(i + 1, FIVE):
                self._cov[tuple(sorted((p, players[j])))] = float(cov[i, j])
