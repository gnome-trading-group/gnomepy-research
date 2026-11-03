"""Development scoring that mirrors the harness: pool and stack log-loss differences against the market and the e12
pool, each pool refitted with each month held out, with a series-bootstrap interval."""
from __future__ import annotations

import numpy as np
import pandas as pd


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return w


def pool(cols: list, y: np.ndarray, months: np.ndarray) -> np.ndarray:
    X = np.column_stack([logit(c) for c in cols] + [np.ones(len(y))])
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ _fit(X[tr], y[tr])))
    return out


def ci(x: np.ndarray, seed: int = 0) -> str:
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (1000, len(x)))].mean(1)
    return f"{x.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}]"


class Scorer:
    """frame: match_id, y, m0, q_e12, month."""

    def __init__(self, frame: pd.DataFrame):
        self.f = frame.reset_index(drop=True)
        self.y = self.f.y.to_numpy(float)
        self.mo = self.f.month.to_numpy()
        self.lm = ll(self.f.m0, self.y)
        self.bp = ll(pool([self.f.m0, self.f.q_e12], self.y, self.mo), self.y)

    def __call__(self, q: np.ndarray, label: str) -> str:
        q = np.asarray(q, float)
        pm = ll(pool([self.f.m0, q], self.y, self.mo), self.y) - self.lm
        st = ll(pool([self.f.m0, q, self.f.q_e12], self.y, self.mo), self.y) - self.bp
        return f"{label:28s} alone {ci(ll(q, self.y) - self.lm)} pool-mkt {ci(pm)} stack {ci(st)}"
