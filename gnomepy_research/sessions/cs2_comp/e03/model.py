"""Ridge-penalised logistic regression fitted by Newton steps, so a weekly refit is deterministic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEWTON_STEPS = 50
TOLERANCE = 1e-9


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Logit:
    mu: np.ndarray
    sd: np.ndarray
    beta: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self._design(x) @ self.beta)

    def _design(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(x)), (x - self.mu) / self.sd])


def fit(x: np.ndarray, y: np.ndarray, ridge: float) -> Logit:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    design = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    penalty = np.diag([0.0] + [ridge] * x.shape[1])
    beta = np.zeros(design.shape[1])
    for _ in range(NEWTON_STEPS):
        p = _sigmoid(design @ beta)
        grad = design.T @ (y - p) - penalty @ beta
        hess = design.T @ (design * (p * (1.0 - p))[:, None]) + penalty
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < TOLERANCE:
            break
    return Logit(mu=mu, sd=sd, beta=beta)
