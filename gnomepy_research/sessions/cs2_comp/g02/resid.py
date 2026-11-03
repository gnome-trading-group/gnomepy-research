"""Persistent market error by player: ridge logistic of past outcomes with the kickoff market price as an offset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize


def fit_residuals(train: pd.DataFrame, lam: float, by_team: bool = False) -> dict[int, float]:
    keys = (lambda r: ([r.team_a_id], [r.team_b_id])) if by_team else (lambda r: (list(r.team_a_lineup_ids), list(r.team_b_lineup_ids)))
    pairs = [keys(r) for r in train.itertuples()]
    ids = sorted({p for a, b in pairs for p in a + b})
    idx = {p: i for i, p in enumerate(ids)}
    rows, cols, vals = [], [], []
    for r, (a, b) in enumerate(pairs):
        for p in a:
            rows.append(r), cols.append(idx[p]), vals.append(1.0)
        for p in b:
            rows.append(r), cols.append(idx[p]), vals.append(-1.0)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(train), len(ids)))
    off = np.log(np.clip(train.m0.to_numpy(), 1e-4, 1 - 1e-4) / (1 - np.clip(train.m0.to_numpy(), 1e-4, 1 - 1e-4)))
    y = train.y.to_numpy(float)

    def f(u):
        z = off + X @ u
        p = 1 / (1 + np.exp(-z))
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * lam * u @ u, X.T @ (p - y) + lam * u

    u = minimize(f, np.zeros(len(ids)), jac=True, method="L-BFGS-B").x
    return dict(zip(ids, u))
