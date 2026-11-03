"""
Stacking the raters: a logistic regression of the bo3 outcome on the raters' A-minus-B skill gaps, fitted on the bo3
series of the trailing year that finished before the cutoff, each priced by the raters as they stood at its own week's
cutoff. No intercept: the gaps are antisymmetric in A and B, so the price is too.

The market is deliberately not an input here. Pooled with the kickoff price (as the harness does), a model-only price
did better than any stack that also took logit(m0), its path or market-fitted segment weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Newton's method with an L2 penalty (deterministic)."""
    k = X.shape[1]
    b = np.zeros(k)
    pen = l2 * np.eye(k)
    for _ in range(100):
        p = sigmoid(X @ b)
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + pen + 1e-9 * np.eye(k), X.T @ (y - p) - l2 * b)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


@dataclass
class Stack:
    cols: tuple
    l2: float = 3.0
    window_days: int = 365

    def fit(self, hist: pd.DataFrame, cutoff: pd.Timestamp) -> "Stack":
        h = hist[(hist.known < cutoff) & hist.y.notna() & (hist.bo_type == 3)
                 & (hist.week_cutoff >= cutoff - pd.Timedelta(days=self.window_days))]
        self.b = fit_logistic(h[list(self.cols)].to_numpy(float), h.y.to_numpy(float), self.l2)
        self.n_train = len(h)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return sigmoid(d[list(self.cols)].to_numpy(float) @ self.b)
