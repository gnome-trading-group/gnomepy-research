"""
The player-skill Kalman filter with HLTV map ratings as a second measurement.

Round results only ever see a five's mean, so a fixed five reveals its sum and nothing about who carries it. Each
player's HLTV rating relative to the five's mean on that map is a linear-Gaussian reading of exactly the part the
rounds cannot see:

    rating_p - mean(rating_five) ~ N(gamma * (s_p - mean(s_five)), sd^2)

which matters whenever the five a team fields next is not the five it was rated with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import PlayerSkillConfig, PlayerSkillFilter

FIVE = 5
_CENTER = np.eye(FIVE) - np.full((FIVE, FIVE), 1.0 / FIVE)


class SkillFilter(PlayerSkillFilter):
    def __init__(self, cfg: PlayerSkillConfig, rankings: pd.DataFrame, rating_gamma: float = 0.6, rating_sd: float = 0.31):
        super().__init__(cfg, rankings)
        self.rating_gamma = rating_gamma
        self.rating_sd = rating_sd

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
