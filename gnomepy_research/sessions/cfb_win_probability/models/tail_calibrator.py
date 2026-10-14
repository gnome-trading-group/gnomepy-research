"""Post-hoc tail calibrator: logit(p') = a z + b z|z| + c post z, with z = logit(p).

b > 0 stretches confident predictions further out (under-confident tails); c shrinks or stretches
postseason games separately. No intercept, so swapping home and away (z -> -z) gives 1 - p'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

CLIP = 1e-4
START = np.array([1.0, 0.0, 0.0])


def _z(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def _design(p: np.ndarray, post: np.ndarray) -> np.ndarray:
    z = _z(p)
    return np.column_stack([z, z * np.abs(z), np.asarray(post, float) * z])


@dataclass(frozen=True)
class TailCalibrator:
    a: float
    b: float
    c: float

    def apply(self, p: np.ndarray, post: np.ndarray) -> np.ndarray:
        eta = _design(p, post) @ np.array([self.a, self.b, self.c])
        return 1 / (1 + np.exp(-eta))


def _nll(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    eta = X @ theta
    loss = np.logaddexp(0, eta) - y * eta
    grad = X.T @ (1 / (1 + np.exp(-eta)) - y)
    return float(loss.mean()), grad / len(y)


def fit_tail_calibrator(p: np.ndarray, y: np.ndarray, post: np.ndarray) -> TailCalibrator:
    X, y = _design(p, post), np.asarray(y, float)
    res = minimize(_nll, START, args=(X, y), jac=True, method="BFGS")
    return TailCalibrator(*map(float, res.x))


def bootstrap_ci(p: np.ndarray, y: np.ndarray, post: np.ndarray, n: int = 500, seed: int = 0,
                 level: float = 0.95) -> dict[str, tuple[float, float]]:
    p, y, post = (np.asarray(v, float) for v in (p, y, post))
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        cal = fit_tail_calibrator(p[idx], y[idx], post[idx])
        fits.append((cal.a, cal.b, cal.c))
    q = np.quantile(np.array(fits), [(1 - level) / 2, (1 + level) / 2], axis=0)
    return {k: (float(q[0, i]), float(q[1, i])) for i, k in enumerate("abc")}
