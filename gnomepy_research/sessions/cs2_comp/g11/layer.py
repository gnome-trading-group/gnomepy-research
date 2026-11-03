"""
The market layer: a ridge logistic regression of the series outcome on the market's kickoff price and the graph
price, refitted for each target week on the market series that finished before its cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The graph price the layer uses: the five-vs-five gap shifted by each team's pool-average map offset.
PRICE = "q_uni_10"


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def features(r: pd.DataFrame) -> pd.DataFrame:
    """Antisymmetric in A and B except the intercept. A series without a kickoff print uses its last print, and one
    with no print at all stands its graph price in for the market's."""
    f = pd.DataFrame(index=r.index)
    zr = logit(r[PRICE].to_numpy(float))
    m = r.m0.fillna(r.p_last)
    f["zm"] = np.where(m.notna(), logit(m.fillna(0.5)), zr)
    f["zr"] = zr
    f["one"] = 1.0
    return f


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ w))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1]), X.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w
