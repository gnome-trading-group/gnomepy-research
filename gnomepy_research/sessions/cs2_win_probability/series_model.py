"""
Direct series-winner model — the benchmark the composed DP must justify itself against.

One row per series, labelled by who won it. This is the estimator DP composition
has to beat or match; if it wins decisively, the DP is anchored to it rather than
replaced by it, so the two markets stay mutually consistent.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.sessions.cs2_win_probability.pre_map_features import PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import _matrix
from gnomepy_research.sessions.cs2_win_probability.symmetry import (
    augment,
    build_swap_plan,
    symmetric_predict_proba,
)

logger = logging.getLogger(__name__)


def load_series_frame(
    priors: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Map-1 priors per series, labelled with the series outcome.

    Frames may be passed in so a locally rebuilt corpus can be evaluated without
    publishing it first.
    """
    ds = DatasetStore()
    priors = ds.load("cs2_match_priors") if priors is None else priors
    history = ds.load("cs2_match_history") if history is None else history

    outcome = history.groupby("match_id").agg(
        n_maps=("team_a_won", "size"),
        a_maps=("team_a_won", "sum"),
        bo_type=("bo_type", "first"),
        match_date=("match_date", "first"),
    ).reset_index()
    outcome["team_a_won_series"] = (outcome.a_maps > outcome.n_maps - outcome.a_maps).astype(int)

    first_map = history[history.map_position_in_series == 1][["match_id", "map_name"]]
    df = outcome.merge(first_map, on="match_id", how="inner")
    df = df.merge(priors, on=["match_id", "map_name"], how="inner", suffixes=("", "_p"))
    return df.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def train_series_model(
    df: pd.DataFrame,
    train_pos: np.ndarray,
    es_pos: np.ndarray,
    cal_pos: np.ndarray,
    n_estimators: int = 1000,
    max_depth: int = 4,
) -> tuple[XGBClassifier, IsotonicRegression, object]:
    """Same symmetry and calibration discipline as Layer 1, on the series label."""
    plan = build_swap_plan(PRE_MAP_FEATURE_NAMES)
    X = _matrix(df, PRE_MAP_FEATURE_NAMES)
    y = df["team_a_won_series"].to_numpy().astype(int)

    X_tr, y_tr = augment(X[train_pos], y[train_pos], plan)
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        early_stopping_rounds=50, tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X[es_pos], y[es_pos])], verbose=False)

    raw = symmetric_predict_proba(model, X[cal_pos], plan)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.concatenate([raw, 1.0 - raw]), np.concatenate([y[cal_pos], 1 - y[cal_pos]]))
    logger.info("series model: best iteration %d", model.best_iteration)
    return model, iso, plan
