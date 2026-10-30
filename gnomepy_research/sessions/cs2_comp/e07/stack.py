"""
The last stage: a logistic regression without an intercept (so a series stays symmetric in A and B) on features that
are each antisymmetric in the two teams, with weights fitted on HLTV series of Oct 2025 - Mar 2026 and frozen in
final_config.py. Every tune target is after that window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Q_LIMIT = 1e-4


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), Q_LIMIT, 1 - Q_LIMIT)
    return np.log(p / (1 - p))


def design(d: pd.DataFrame, faceit: bool) -> tuple[np.ndarray, list[str]]:
    cols = [logit(d.q_h.to_numpy())]
    names = ["z_hltv"]
    if faceit:
        cols.append((d.f_s_a - d.f_s_b).to_numpy(float))
        names.append("faceit_skill_diff")
    return np.column_stack(cols), names


def fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-x @ w))
        hess = x.T @ (x * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(hess, x.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def apply(d: pd.DataFrame, weights: dict, faceit: bool) -> np.ndarray:
    x, names = design(d, faceit)
    w = np.array([weights[n] for n in names])
    return 1 / (1 + np.exp(-x @ w))
