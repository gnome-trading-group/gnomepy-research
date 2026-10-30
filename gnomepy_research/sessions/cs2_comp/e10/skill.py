"""Player skill as a Gaussian state-space filter.

Skill is in round-logit units: a lineup gap D gives each round to team A with probability sigmoid(D). A player is one
Gaussian whose variance grows tau^2 per idle day. A lineup is the mean of its players plus lam on its best one.
Only teammates carry covariance: a fixed five reveals only its sum, so teammates' errors turn negatively correlated and
each player's own variance falls slowly.

Two measurement channels feed one filter:
  - the rounds of a map, observed through the lineup gap, with an overdispersion term;
  - a performance rating relative to the player's teammates on the map, observed through gamma * (skill - lineup mean).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NEWCOMER = -1
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SkillConfig:
    tau: float = 0.009319191322975635
    sigma0: float = 0.2536091079975324
    a: float = -0.1081114476105491
    b: float = 0.14696262712175345
    lam: float = 0.0761331162651603
    rho: float = 0.017231894894383412
    scale: float = 1.0
    level_slope: float = 0.0
    gamma: float = 0.0
    sigma_r: float = 0.25


@dataclass
class SkillState:
    n: int
    sigma0: float
    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)
    seen: np.ndarray = field(init=False)
    games: np.ndarray = field(init=False)
    cov: dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        self.mu = np.zeros(self.n)
        self.var = np.full(self.n, self.sigma0 ** 2)
        self.seen = np.full(self.n, NEWCOMER, dtype=np.int64)
        self.games = np.zeros(self.n, dtype=np.int64)


def _psd(cov: np.ndarray) -> np.ndarray:
    """Covariances of a pair are only refreshed when the pair plays together, so a stale one can exceed what the
    players' current variances allow; the block is then clipped back to positive semi-definite."""
    w, v = np.linalg.eigh(cov)
    if w[0] >= 1e-9:
        return cov
    return (v * np.maximum(w, 1e-9)) @ v.T


class SkillFilter:
    def __init__(self, n_players: int, cfg: SkillConfig):
        self.cfg = cfg
        self.n = n_players
        self.s = SkillState(n_players, cfg.sigma0)

    def _key(self, p: int, q: int) -> int:
        return p * self.n + q if p < q else q * self.n + p

    def _block(self, idx: np.ndarray, var: np.ndarray) -> np.ndarray:
        k = len(idx)
        cov = np.diag(var)
        stored = self.s.cov
        for i in range(k):
            pi = int(idx[i])
            for j in range(i + 1, k):
                v = stored.get(self._key(pi, int(idx[j])))
                if v is not None:
                    cov[i, j] = cov[j, i] = v
        return _psd(cov)

    def _store(self, idx: np.ndarray, cov: np.ndarray) -> None:
        stored = self.s.cov
        k = len(idx)
        for i in range(k):
            pi = int(idx[i])
            for j in range(i + 1, k):
                stored[self._key(pi, int(idx[j]))] = float(cov[i, j])

    def introduce(self, idx: int, prior_mean: float, day: int) -> None:
        self.s.seen[idx] = day
        self.s.mu[idx] = prior_mean

    def known(self, idx: int) -> bool:
        return self.s.seen[idx] != NEWCOMER

    def current(self, idx: np.ndarray, day: int) -> tuple[np.ndarray, np.ndarray]:
        st = self.s
        idle = np.maximum(day - st.seen[idx], 0)
        return st.mu[idx], st.var[idx] + self.cfg.tau ** 2 * idle

    def weights(self, mus: np.ndarray) -> np.ndarray:
        n = len(mus)
        w = np.full(n, (1 - self.cfg.lam) / n)
        # Near-ties go to the first listed player: teammates who debut together differ only in rounding bits.
        w[int(np.argmax(mus >= mus.max() - TIE_TOLERANCE))] += self.cfg.lam
        return w

    def gap(self, a: np.ndarray, b: np.ndarray, day: int, mu: tuple | None = None, var: tuple | None = None) -> tuple[float, float]:
        """Mean and variance of the lineup gap; mu/var override the filter's own player estimates (a, b order)."""
        mu_a, var_a = self.current(a, day) if mu is None else (mu[0], var[0])
        mu_b, var_b = self.current(b, day) if mu is None else (mu[1], var[1])
        wa, wb = self.weights(mu_a), self.weights(mu_b)
        cov_a, cov_b = self._block(a, var_a), self._block(b, var_b)
        return float(wa @ mu_a - wb @ mu_b), float(wa @ cov_a @ wa + wb @ cov_b @ wb)

    def observe_map(self, day: int, a: np.ndarray, b: np.ndarray, rounds_a: int, rounds_b: int) -> None:
        cfg = self.cfg
        mu_a, var_a = self.current(a, day)
        mu_b, var_b = self.current(b, day)
        wa, wb = self.weights(mu_a) * cfg.scale, self.weights(mu_b) * cfg.scale
        cov_a, cov_b = self._block(a, var_a), self._block(b, var_b)
        d = float(wa @ mu_a - wb @ mu_b)
        s2 = float(wa @ cov_a @ wa + wb @ cov_b @ wb)
        n = rounds_a + rounds_b
        pi = 1 / (1 + math.exp(-max(min(d, 30.0), -30.0)))
        inflation = 1 + (n - 1) * cfg.rho
        grad, curv = (rounds_a - n * pi) / inflation, n * pi * (1 - pi) / inflation
        ca, cb = cov_a @ wa, cov_b @ wb
        denom = 1 + curv * s2
        self._write(a, mu_a + ca * grad / denom, cov_a - np.outer(ca, ca) * curv / denom, day)
        self._write(b, mu_b - cb * grad / denom, cov_b - np.outer(cb, cb) * curv / denom, day)

    def _write(self, idx: np.ndarray, mu: np.ndarray, cov: np.ndarray, day: int) -> None:
        st = self.s
        st.mu[idx] = mu
        st.var[idx] = np.maximum(np.diag(cov), 1e-6)
        st.seen[idx] = day
        st.games[idx] += 1
        self._store(idx, cov)

    def observe_rating(self, day: int, idx: np.ndarray, rating_dev: np.ndarray) -> None:
        g = self.cfg.gamma
        if g == 0.0 or len(idx) < 2 or not np.isfinite(rating_dev).all():
            return
        k = len(idx)
        mu, var = self.current(idx, day)
        cov = self._block(idx, var)
        h = g * (np.eye(k) - 1.0 / k)
        s_mat = h @ cov @ h.T + self.cfg.sigma_r ** 2 * np.eye(k)
        gain = cov @ h.T @ np.linalg.inv(s_mat)
        mu = mu + gain @ (rating_dev - h @ mu)
        cov = cov - gain @ h @ cov
        st = self.s
        st.mu[idx] = mu
        st.var[idx] = np.maximum(np.diag(cov), 1e-6)
        self._store(idx, cov)
