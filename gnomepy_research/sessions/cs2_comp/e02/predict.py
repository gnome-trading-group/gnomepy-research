"""Predict pre-match series probabilities with the e02 model.

    python -m gnomepy_research.sessions.cs2_comp.e02.predict --data <bundle> --targets <parquet> --out <parquet> [--no-faceit]

Each target week is predicted by a model fitted only on series dated before that week's cutoff, with features cut at
the same cutoff. Orientation is symmetric: q averages the model over both team orders.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from gnomepy_research.sessions.cs2_comp.e02.features import Queries, build_queries, feature_columns

BEST_OF = 3
SEED = 0
XGB_PARAMS = {
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


def fit(queries: Queries, week: pd.Timestamp, faceit: bool) -> XGBClassifier:
    train = (queries.meta.week < week) & ~queries.meta.is_target & (queries.meta.bo == BEST_OF)
    idx = train.to_numpy()
    X = pd.concat([queries.ab[idx], queries.ba[idx]], ignore_index=True)
    y = np.concatenate([queries.meta.y.to_numpy()[idx], 1 - queries.meta.y.to_numpy()[idx]]).astype(int)
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X[feature_columns(faceit)], y)
    return model


def predict_week(model: XGBClassifier, queries: Queries, mask: np.ndarray, faceit: bool) -> np.ndarray:
    cols = feature_columns(faceit)
    p_ab = model.predict_proba(queries.ab.loc[mask, cols])[:, 1]
    p_ba = model.predict_proba(queries.ba.loc[mask, cols])[:, 1]
    return np.clip((p_ab + 1 - p_ba) / 2, 0.001, 0.999)


def run(data_dir: str, targets: pd.DataFrame, faceit: bool) -> pd.DataFrame:
    cut = pd.Timestamp(targets.week_cutoff.max())
    queries = build_queries(data_dir, cut, targets, faceit)
    out = []
    for week in sorted(targets.week_cutoff.unique()):
        week = pd.Timestamp(week)
        model = fit(queries, week, faceit)
        mask = (queries.meta.is_target & (queries.meta.week == week)).to_numpy()
        q = predict_week(model, queries, mask, faceit)
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
