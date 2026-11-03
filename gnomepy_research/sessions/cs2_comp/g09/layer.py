"""
The g09 layer: a ridge logistic regression of the series result on the power rating and two team adjustments,
fitted on every bo3 series of the last TRAIN_DAYS before the cutoff.

  zavg    mean over the rating variants of the bo3 price logit implied by each map-level rating gap
  form    A-minus-B recency-weighted residual against the main rating (team.py)
  mapadv  A's expected edge over the maps a greedy veto leaves (team.py)

The intercept carries HLTV's team-A ordering. The market is left out on purpose: the scorer pools q with the kickoff
price itself, and every market feature tried (smoothed path, market-implied team ratings) added nothing once the
ratings were in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g09.core import bo3, logit

FEATURES = ("zavg", "form", "mapadv")
TRAIN_DAYS = 180.0
L2 = 1.0


def _price_logit(gap: np.ndarray) -> np.ndarray:
    return logit(bo3(1 / (1 + np.exp(-gap))))


def design(rows: pd.DataFrame, features=FEATURES) -> np.ndarray:
    gaps = [rows.elo_gap.to_numpy(float)] + [rows[c].to_numpy(float) for c in rows.columns if c.startswith("elo_gap_")]
    cols = {"zavg": np.mean([_price_logit(g) for g in gaps], axis=0)}
    return np.column_stack([cols[f] if f in cols else rows[f].to_numpy(float) for f in features] + [np.ones(len(rows))])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = L2) -> np.ndarray:
    w = np.zeros(X.shape[1])
    pen = l2 * np.ones(X.shape[1])
    pen[-1] = 1e-6
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen), X.T @ (y - p) - pen * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def fit_predict(train: pd.DataFrame, targets: pd.DataFrame, features=FEATURES, l2: float = L2) -> np.ndarray:
    cutoff = targets.snap.max()
    train = train[train.kickoff >= cutoff - pd.Timedelta(days=TRAIN_DAYS)]
    w = fit_logistic(design(train, features), train.y.to_numpy(float), l2)
    return 1 / (1 + np.exp(-design(targets, features) @ w))
