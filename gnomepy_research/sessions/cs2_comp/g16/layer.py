"""
The combination layer: a logistic regression of the outcome on the market reading and the skill prior, with a
Gaussian prior on its weights so the first weeks, which see only a month of market history, stay close to
"trust the market" instead of chasing noise.

The design is antisymmetric in the two teams except for the intercept, so swapping A and B flips the price.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit

LOGIT_CLIP = 1e-4


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class Spec:
    """Which columns enter the layer and the prior on their weights."""
    columns: tuple[str, ...] = ("zm", "zs")
    prior_mean: dict = field(default_factory=lambda: {"zm": 1.0})
    prior_precision: float = 4.0
    intercept: bool = False


def design(rows: pd.DataFrame, spec: Spec) -> np.ndarray:
    zm = logit(rows.twap)
    zl = logit(rows.m_last)
    zs = logit(rows.p_series)
    liq = np.log1p(rows.n_active.to_numpy(float)) - np.log1p(600.0)
    gs = rows.gap_sd.to_numpy(float)
    cols = {"zm": zm, "zs": zs, "dz": zl - zm, "zm_liq": zm * liq, "zs_liq": zs * liq, "zs_sd": zs * gs,
            "zm_sd": zm * gs, "new": (rows.new_a - rows.new_b).to_numpy(float), "zs_new": zs * (rows.new_a + rows.new_b).to_numpy(float),
            "dz_liq": (zl - zm) * liq, "zl": zl}
    X = np.column_stack([cols[c] for c in spec.columns])
    if spec.intercept:
        X = np.column_stack([X, np.ones(len(rows))])
    return X


def fit(X: np.ndarray, y: np.ndarray, spec: Spec) -> np.ndarray:
    k = X.shape[1]
    m = np.array([spec.prior_mean.get(c, 0.0) for c in spec.columns] + ([0.0] if spec.intercept else []))
    lam = spec.prior_precision
    w = m.copy()
    for _ in range(100):
        p = expit(X @ w)
        g = X.T @ (y - p) - lam * (w - m)
        H = X.T @ (X * (p * (1 - p))[:, None]) + lam * np.eye(k)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return expit(X @ w)
