"""
The learned layers on top of the rating engine's series rows.

Model layer: a ridge logistic regression of the series outcome on the engine's features, fitted on every HLTV bo3
series known before the cutoff (recency-weighted), giving z_model = logit of the model's own price.

Market layer: on the series that had a Polymarket price, the outcome regressed on logit(m0) plus z_model and a few
interactions and price-path features, ridge-shrunk toward the market alone, so with little market history the
answer is the market and the model's weight grows as the evidence does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.data import LOG_UNRANKED, logit

MODEL_FEATURES = ("z_fit", "z_fit_thin", "off_fit", "spread_fit", "z_elo", "z_series", "veto_edge", "z_thin", "form", "churn", "new",
                  "idle", "together", "since_move", "sub", "rank", "z_lan", "new_lan")
MARKET_FEATURES = ("z_model", "z_model_sd", "z_model_thin", "z_model_liq", "drift24", "drift6", "drift1", "liq", "flat")


@dataclass(frozen=True)
class LayerConfig:
    window_days: float = 540.0
    half_life_days: float = 365.0
    l2_model: float = 2.0
    l2_market: float = 2.0
    market_features: tuple[str, ...] = ("z_model", "drift24", "drift1")
    model_features: tuple[str, ...] = MODEL_FEATURES
    min_market_rows: int = 150


def fit_ridge_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, offset: np.ndarray | None = None,
                       penalize: np.ndarray | None = None, iters: int = 40) -> np.ndarray:
    """Weighted ridge logistic regression by Newton's method; `offset` is a fixed part of the linear predictor,
    `penalize` masks which coefficients the ridge applies to (default all but the last, the intercept)."""
    n, k = X.shape
    if penalize is None:
        penalize = np.ones(k)
        penalize[-1] = 0.0
    off = np.zeros(n) if offset is None else offset
    beta = np.zeros(k)
    R = l2 * np.diag(penalize)
    for _ in range(iters):
        eta = off + X @ beta
        p = 1 / (1 + np.exp(-eta))
        g = X.T @ (w * (y - p)) - R @ beta
        H = X.T @ (X * (w * p * (1 - p))[:, None]) + R + 1e-8 * np.eye(k)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    return beta


def model_column(rows: pd.DataFrame, name: str) -> np.ndarray:
    if name == "intercept":
        return np.ones(len(rows))
    return _columns(rows)[name]


def model_design(rows: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    cols = _columns(rows)
    return np.column_stack([cols[f] for f in features] + [np.ones(len(rows))])


def _columns(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    z = rows.z_series.to_numpy(float)
    ze = rows.z_elo.to_numpy(float)
    cols = {}
    if "z_fit" in rows:
        zf = rows.z_fit.to_numpy(float)
        cols.update({"z_fit": zf, "z_fit_thin": zf * (rows.thin_fit_a + rows.thin_fit_b).to_numpy(float),
                     "off_fit": rows.off_fit.to_numpy(float), "spread_fit": (rows.spread_fit_a - rows.spread_fit_b).to_numpy(float)})
    cols.update({
        "z_elo": ze,
        "z_elo_sd": ze * (rows.sd_elo.to_numpy(float) - 0.2),
        "adj": (rows.adj_a - rows.adj_b).to_numpy(float),
        "z_series": z,
        "veto_edge": (rows.z_veto - rows.z_series).to_numpy(float),
        "d_team": rows.d_team.to_numpy(float),
        "z_sd": z * rows.sd_d.to_numpy(float),
        "z_thin": z * (rows.thin_a + rows.thin_b).to_numpy(float),
        "form": (rows.form_a - rows.form_b).to_numpy(float),
        "churn": (rows.churn_a - rows.churn_b).to_numpy(float),
        "new": (rows.new_a - rows.new_b).to_numpy(float),
        "idle": np.log1p(rows.idle_a.to_numpy(float)) - np.log1p(rows.idle_b.to_numpy(float)),
        "together": np.log1p(rows.together_a.to_numpy(float)) - np.log1p(rows.together_b.to_numpy(float)),
        "since_move": np.log1p(rows.since_move_a.to_numpy(float)) - np.log1p(rows.since_move_b.to_numpy(float)),
        "sub": (rows.sub_a - rows.sub_b).to_numpy(float),
        "rank": np.log(rows.rank_b.to_numpy(float)) - np.log(rows.rank_a.to_numpy(float)),
        "z_lan": z * rows.is_lan.to_numpy(float),
        "new_lan": (rows.new_a - rows.new_b).to_numpy(float) * rows.is_lan.to_numpy(float),
    })
    return cols


def market_design(rows: pd.DataFrame, z_model: np.ndarray, features: tuple[str, ...]) -> np.ndarray:
    liq = np.log1p(rows.n_prints.to_numpy(float)) - 6.5
    thin = (rows.thin_a + rows.thin_b).to_numpy(float)
    cols = {
        "z_model": z_model,
        "z_model_sd": z_model * (rows.sd_d.to_numpy(float) - 0.15),
        "z_model_thin": z_model * thin,
        "z_model_liq": z_model * liq,
        "drift24": rows.drift24.to_numpy(float),
        "drift6": rows.drift6.to_numpy(float),
        "drift1": rows.drift1.to_numpy(float),
        "liq": liq,
        "flat": rows.flat_half.to_numpy(float),
    }
    return np.column_stack([cols[f] for f in features] + [np.ones(len(rows))])


class Layers:
    """Both layers fitted at one cutoff on the series rows known before it."""

    def __init__(self, rows: pd.DataFrame, cutoff: pd.Timestamp, cfg: LayerConfig):
        self.cfg = cfg
        known = pd.to_datetime(rows.known)
        age = (cutoff - pd.to_datetime(rows.time)).dt.total_seconds().to_numpy() / 86400.0
        keep = (known < cutoff).to_numpy() & (age <= cfg.window_days) & (age >= 0)
        tr = rows[keep]
        w = 0.5 ** (age[keep] / cfg.half_life_days)
        X = model_design(tr, cfg.model_features)
        self.scale = np.maximum(X.std(axis=0), 1e-6)
        self.scale[-1] = 1.0
        self.beta_model = fit_ridge_logistic(X / self.scale, tr.y.to_numpy(float), w, cfg.l2_model) / self.scale
        self.n_model = int(keep.sum())
        z = X @ self.beta_model
        has = tr.m0.notna().to_numpy()
        self.n_market = int(has.sum())
        self.beta_market = None
        if self.n_market >= cfg.min_market_rows:
            tm = tr[has]
            Xm = market_design(tm, z[has], cfg.market_features)
            self.mscale = np.maximum(Xm.std(axis=0), 1e-6)
            self.mscale[-1] = 1.0
            self.beta_market = fit_ridge_logistic(Xm / self.mscale, tm.y.to_numpy(float), w[has], cfg.l2_market,
                                                  offset=logit(tm.m0.to_numpy())) / self.mscale

    def z_model(self, rows: pd.DataFrame) -> np.ndarray:
        return model_design(rows, self.cfg.model_features) @ self.beta_model

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """q_model (the model alone) and q (market-adjusted where the series has a kickoff price)."""
        z = self.z_model(rows)
        q_model = 1 / (1 + np.exp(-z))
        q = q_model.copy()
        has = rows.m0.notna().to_numpy() if "m0" in rows else np.zeros(len(rows), bool)
        if self.beta_market is not None and has.any():
            Xm = market_design(rows[has], z[has], self.cfg.market_features)
            eta = logit(rows.m0.to_numpy()[has]) + Xm @ self.beta_market
            q[has] = 1 / (1 + np.exp(-eta))
        return pd.DataFrame({"match_id": rows.match_id.to_numpy(), "q_model": q_model, "q": q, "z_model": z})
