"""The final layer: a Bayesian logistic regression of the series result on the market and the filter.

    logit P(A) = b . x,    b ~ N(b0, diag(prior_sd^2))

x holds logit(m0), the filter's logit price, the filter price scaled down by its uncertainty and by how thin the
market is, the late moves of the market path, and an intercept. The prior centres the market weight on one and every
other weight on zero, so with little training data the layer stays at the market. The posterior mode is the price.

A series without a kickoff print is priced by the same layer with the market terms set to zero (a separate fit on the
same rows with the market columns left out)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(d: pd.DataFrame, market: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Columns, prior means and prior sds."""
    zq = logit(d.q_skill)
    cols, b0, sd = [zq], [0.0], [1.0]
    if market:
        z0 = logit(d.m0)
        cols += [z0, logit(d.p1) - z0, logit(d.mean6) - z0]
        b0 += [1.0, 0.0, 0.0]
        sd += [1.0, 0.5, 0.5]
    cols.append(np.ones(len(d)))
    b0.append(0.0)
    sd.append(0.5)
    return np.column_stack(cols), np.array(b0), np.array(sd)


def map_fit(X: np.ndarray, y: np.ndarray, b0: np.ndarray, sd: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    prec = 1.0 / sd ** 2
    w = np.ones(len(y)) if w is None else w
    b = b0.copy()
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (w * (y - p)) - prec * (b - b0)
        H = X.T @ (X * (w * p * (1 - p))[:, None]) + np.diag(prec)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def fit_and_price(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    has = rows.m0.notna().to_numpy() & rows.has_skill.astype(bool).to_numpy() | rows.m0.notna().to_numpy()
    q = np.full(len(rows), np.nan)
    tr = train[train.m0.notna()] if len(train) else train
    y = tr.y.to_numpy(float) if len(tr) else np.zeros(0)
    for market, sel in ((True, has), (False, ~has)):
        if not sel.any():
            continue
        X, b0, sd = design(tr, market) if len(tr) else (np.zeros((0, len(design(rows[sel], market)[1]))), *design(rows[sel], market)[1:])
        b = map_fit(X, y, b0, sd) if len(tr) else b0
        Xr = design(rows[sel], market)[0]
        q[sel] = 1 / (1 + np.exp(-Xr @ b))
    return q
