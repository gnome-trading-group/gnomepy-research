"""
A walk-forward ridge logistic layer over the e12 skill price and the series' market path: the candidate `q` this
entry evaluated and did not submit.

Refit at every weekly cutoff on every priced bo3 series known before it. The default spec anchors on e12's logit
price (fixed coefficient 1) and learns an intercept and a thin-market gate (few prints, or a price that never
moved); other specs re-learn the market/e12 mix or add context columns. On dev every spec sat at or below the
baseline pool and the gate's weights flipped sign from week to week, so `predict.py` carries e12's price alone.
Every column is antisymmetric in A and B except the intercept.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

THIN_PRINTS = 50


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class LayerSpec:
    price: str = "m_last2h"
    columns: tuple[str, ...] = ("lm_thin", "z_thin")
    l2: float = 1.0
    # A column added to the logit with a fixed coefficient of 1, so the layer only learns what to add to it.
    offset: str | None = "z"
    # Rows the layer may learn from: at least this many weeks of priced history, else the fallback price is used.
    min_rows: int = 200


def thin_market(rows: pd.DataFrame) -> np.ndarray:
    return ((rows.n_prints.fillna(0) < THIN_PRINTS) | (rows.flat.fillna(1) > 0)).to_numpy(float)


def design(rows: pd.DataFrame, spec: LayerSpec) -> np.ndarray:
    """One column per name in `spec.columns`, then the intercept, then the offset (or zeros). Missing path features
    count as a thin market at a flat price."""
    m = rows[spec.price].fillna(0.5)
    lm = logit(m)
    z = rows.z_e12.to_numpy(float)
    thin = thin_market(rows)
    cols = {
        "lm": lm, "z": z, "lm_thin": lm * thin, "z_thin": z * thin,
        "zh": rows.zh.to_numpy(float), "zf": rows.zf.to_numpy(float),
        "drift6": lm - logit(rows.m_at6h.fillna(m)), "drift24": lm - logit(rows.m_at24h.fillna(m)),
        "lm0": logit(rows.m0.fillna(0.5)),
        "thin": thin,
        "z_lan": z * rows.is_lan.fillna(0).to_numpy(float), "lm_lan": lm * rows.is_lan.fillna(0).to_numpy(float),
        "standin": rows.standin.to_numpy(float), "idle": rows.idle.to_numpy(float),
        "log_rank": rows.log_rank.to_numpy(float), "z_unk": z * rows.thin.to_numpy(float),
        "z_new": z * rows.new.to_numpy(float),
    }
    if "z_ekf" in rows:
        # Experiment columns (experiments/ekf.py); a series the filter could not price takes e12's price.
        ze = rows.z_ekf.fillna(rows.z_e12).to_numpy(float)
        sd = rows.ekf_sd.fillna(0.0).to_numpy(float)
        cols.update({"z_ekf": ze, "zekf_thin": ze * thin, "z_sd": z * sd, "lm_sd": lm * sd,
                     "ekf_chem": rows.ekf_chem.fillna(0.0).to_numpy(float)})
    offset = cols[spec.offset] if spec.offset else np.zeros(len(rows))
    return np.column_stack([cols[c] for c in spec.columns] + [np.ones(len(rows)), offset])


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Newton steps on the L2-penalised log loss (the intercept is not penalised); the last column is the offset."""
    X, offset = X[:, :-1], X[:, -1]
    pen = np.full(X.shape[1], l2)
    pen[-1] = 0.0
    w = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ w + offset)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - pen * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


@dataclass
class Layer:
    spec: LayerSpec
    weights: np.ndarray | None
    n_train: int = 0
    info: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, train: pd.DataFrame, spec: LayerSpec) -> Layer:
        train = train[train.y.notna() & train.m0.notna()]
        if len(train) < spec.min_rows:
            return cls(spec, None, len(train))
        X = design(train, spec)
        w = fit_ridge(X, train.y.to_numpy(float), spec.l2)
        return cls(spec, w, len(train), dict(zip(list(spec.columns) + ["intercept"], map(float, w))))

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            return fallback(rows, self.spec)
        X = design(rows, self.spec)
        return 1 / (1 + np.exp(-(X[:, :-1] @ self.weights + X[:, -1])))


def fallback(rows: pd.DataFrame, spec: LayerSpec) -> np.ndarray:
    """Before the layer has history to learn from: the smoothed market where it exists, else e12's price."""
    m = rows[spec.price].to_numpy(float)
    z = 1 / (1 + np.exp(-rows.z_e12.to_numpy(float)))
    thin = thin_market(rows) > 0
    return np.where(np.isnan(m) | thin, z, m)
