"""
The market layer: a ridge logistic regression of the series outcome on the market's kickoff price and the graph
price, refitted for each target week on the market series that finished before its cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g11.graph import series_price


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def features(r: pd.DataFrame) -> pd.DataFrame:
    """Antisymmetric in A and B except the intercept. A series without a kickoff print uses its last print, and one
    with no print at all sits at even money with the graph price alone deciding (zm = 0, has_market = 0)."""
    f = pd.DataFrame(index=r.index)
    m = r.m0.fillna(r.p_last)
    has = m.notna().to_numpy(float)
    f["zm"] = np.where(has > 0, logit(m.fillna(0.5)), 0.0)
    zr = logit(series_price(r.gap.to_numpy(float), r.gap_var.to_numpy(float)))
    f["zr"] = zr
    f["zr_nomkt"] = zr * (1 - has)
    f["zr_sd"] = zr * np.sqrt(r.gap_var.to_numpy(float))
    f["zr_cross"] = zr * r.cross.to_numpy(float)
    f["zr_thin"] = zr * (r.thin_a + r.thin_b).to_numpy(float)
    f["standin"] = (r.standin_a - r.standin_b).to_numpy(float)
    f["idle"] = np.log1p(r.idle_a.to_numpy(float)) - np.log1p(r.idle_b.to_numpy(float))
    f["sub"] = (r.sub_a - r.sub_b).to_numpy(float)
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
