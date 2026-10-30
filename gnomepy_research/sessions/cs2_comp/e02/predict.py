"""Predict pre-match series probabilities with the e02 model.

    python -m gnomepy_research.sessions.cs2_comp.e02.predict --data <bundle> --targets <parquet> --out <parquet> [--no-faceit]

Each target week is predicted by models fitted only on series dated before that week's cutoff, with features cut at
the same cutoff. Orientation is symmetric: q averages each model over both team orders.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from gnomepy_research.sessions.cs2_comp.e02.features import Queries, build_queries, feature_columns

BEST_OF = 3
SEED = 0
GBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 40,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": SEED,
}


@dataclass(frozen=True)
class Config:
    models: tuple[str, ...] = ("gbm",)
    window_days: int | None = None
    gbm: dict = field(default_factory=lambda: dict(GBM_PARAMS))


def make_model(name: str, gbm: dict):
    if name == "gbm":
        return XGBClassifier(**gbm)
    if name == "lr":
        return make_pipeline(SimpleImputer(strategy="mean", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000))
    raise ValueError(name)


def training_rows(queries: Queries, week: pd.Timestamp, window_days: int | None) -> np.ndarray:
    meta = queries.meta
    keep = (meta.week < week) & ~meta.is_target & (meta.bo == BEST_OF)
    if window_days is not None:
        keep &= meta.week >= week - pd.Timedelta(days=window_days)
    return keep.to_numpy()


def fit(queries: Queries, week: pd.Timestamp, faceit: bool, config: Config) -> list:
    idx = training_rows(queries, week, config.window_days)
    cols = feature_columns(faceit)
    X = pd.concat([queries.ab.loc[idx, cols], queries.ba.loc[idx, cols]], ignore_index=True)
    y_base = queries.meta.y.to_numpy()[idx]
    y = np.concatenate([y_base, 1 - y_base]).astype(int)
    models = []
    for name in config.models:
        model = make_model(name, config.gbm)
        model.fit(X, y)
        models.append(model)
    return models


def predict_week(models: list, queries: Queries, mask: np.ndarray, faceit: bool) -> np.ndarray:
    cols = feature_columns(faceit)
    ps = []
    for model in models:
        p_ab = model.predict_proba(queries.ab.loc[mask, cols])[:, 1]
        p_ba = model.predict_proba(queries.ba.loc[mask, cols])[:, 1]
        ps.append((p_ab + 1 - p_ba) / 2)
    return np.clip(np.mean(ps, axis=0), 0.001, 0.999)


def run(data_dir: str, targets: pd.DataFrame, faceit: bool, config: Config = Config()) -> pd.DataFrame:
    cut = pd.Timestamp(targets.week_cutoff.max())
    queries = build_queries(data_dir, cut, targets, faceit)
    out = []
    for week in sorted(targets.week_cutoff.unique()):
        week = pd.Timestamp(week)
        models = fit(queries, week, faceit, config)
        mask = (queries.meta.is_target & (queries.meta.week == week)).to_numpy()
        q = predict_week(models, queries, mask, faceit)
        out.append(pd.DataFrame({"match_id": queries.meta.match_id[mask].to_numpy(), "q": q, "fit_cutoff": week}))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", action="store_true")
    args = ap.parse_args()
    targets = pd.read_parquet(args.targets)
    targets["week_cutoff"] = pd.to_datetime(targets.week_cutoff)
    preds = run(args.data, targets, faceit=not args.no_faceit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(args.out, index=False)
    print(f"wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
