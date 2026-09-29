"""
XGBoost round win probability model.

Training:
    python -m gnomepy_research.sessions.cs2_win_probability.model \
        --dataset cs2_round_features \
        --out /tmp/cs2_round_model.xgb

Publish:
    poetry run research artifacts publish /tmp/cs2_round_model.xgb \
        --type xgboost_model --name cs2_round_win_prob \
        --session cs2_win_probability
"""
from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.sessions.cs2_win_probability.features import (
    FEATURE_NAMES,
    extract_features_from_demo_row,
)

logger = logging.getLogger(__name__)


def _load_priors_lookup() -> dict | None:
    """
    Try to load cs2_match_priors from DatasetStore.
    Returns a dict keyed by (match_id, map_name) -> priors_dict, or None.
    """
    try:
        priors_df = DatasetStore().load("cs2_match_priors")
        lookup = {}
        for _, row in priors_df.iterrows():
            mid = row.get("match_id")
            if mid is not None:
                lookup[(int(mid), str(row["map_name"]))] = {
                    "team_a_avg_rating": float(row.get("team_a_rating_diff", 0.0)),
                    "team_b_avg_rating": 0.0,
                    "team_a_map_winrate_long": float(row.get("team_a_map_winrate_long", 0.5)),
                    "team_b_map_winrate_long": float(row.get("team_b_map_winrate_long", 0.5)),
                    "team_a_map_winrate_short": float(row.get("team_a_map_winrate_short", 0.5)),
                    "team_b_map_winrate_short": float(row.get("team_b_map_winrate_short", 0.5)),
                    "h2h_win_rate": float(row.get("h2h_win_rate", 0.5)),
                    "team_a_recent_form": 0.5 + float(row.get("recent_form_diff", 0.0)) / 2,
                    "team_b_recent_form": 0.5 - float(row.get("recent_form_diff", 0.0)) / 2,
                    "team_a_picked_map": bool(row.get("team_a_picked_map", False)),
                    "is_lan": int(row.get("is_lan", 1)),
                }
        logger.info("Loaded %d match priors for training enrichment", len(lookup))
        return lookup
    except Exception as exc:
        logger.info("cs2_match_priors not available (%s) — training without prior features", exc)
        return None


def train(df: pd.DataFrame, out_path: str, n_estimators: int = 500, max_depth: int = 6) -> dict:
    """
    Train XGBoost on round-level snapshots. Returns eval metrics.
    Splits by demo to avoid data leakage. Enriches with point-in-time priors
    from cs2_match_priors dataset when available.
    """
    priors_lookup = _load_priors_lookup()

    demos = df["source_demo"].unique()
    np.random.seed(42)
    np.random.shuffle(demos)

    split = int(len(demos) * 0.8)
    train_demos = set(demos[:split])
    test_demos = set(demos[split:])

    train_df = df[df["source_demo"].isin(train_demos)]
    test_df = df[df["source_demo"].isin(test_demos)]

    def _features(row):
        priors = None
        if priors_lookup and row.get("match_id") is not None:
            priors = priors_lookup.get((int(row["match_id"]), str(row.get("map_name", ""))))
        return extract_features_from_demo_row(row, priors)

    X_train = np.vstack([_features(r) for _, r in train_df.iterrows()])
    y_train = train_df["ct_won"].values.astype(int)

    X_test = np.vstack([_features(r) for _, r in test_df.iterrows()])
    y_test = test_df["ct_won"].values.astype(int)

    logger.info("Training: %d rounds (%d demos), Test: %d rounds (%d demos)",
                len(X_train), len(train_demos), len(X_test), len(test_demos))

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=30,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )
    best_iter = model.best_iteration
    logger.info("Best iteration: %d", best_iter)

    # Lock in the best iteration count and remove early_stopping_rounds so
    # CalibratedClassifierCV can re-train XGBoost in folds without needing eval_set.
    model.set_params(n_estimators=best_iter, early_stopping_rounds=None)

    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    y_prob = calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    logger.info("Test AUC=%.4f  log-loss=%.4f", auc, ll)

    # Contested AUC: equal sides (equipment within $2000, equal players)
    contested_mask = (
        (np.abs(X_test[:, FEATURE_NAMES.index("equip_value_diff")]) < 2000) &
        (X_test[:, FEATURE_NAMES.index("players_alive_diff")] == 0)
    )
    if contested_mask.sum() > 50:
        contested_auc = roc_auc_score(y_test[contested_mask], y_prob[contested_mask])
        logger.info("Contested AUC (equal sides): %.4f on %d samples", contested_auc, contested_mask.sum())
    else:
        contested_auc = float("nan")

    joblib.dump(calibrated, out_path)
    logger.info("Model saved to %s", out_path)

    return {"auc": auc, "log_loss": ll, "contested_auc": contested_auc,
            "n_train": len(X_train), "n_test": len(X_test)}


class CS2RoundModel:
    """
    Thin inference wrapper loaded by the live strategy.
    Load via: resolve_artifact_path("artifact://xgboost_model/cs2_round_win_prob")
    """

    def __init__(self, model_path: str):
        self._model = joblib.load(model_path)

    def predict_ct_win_prob(self, features: np.ndarray) -> float:
        """Return P(CT wins this round) given a feature vector."""
        x = features.reshape(1, -1)
        prob = self._model.predict_proba(x)[0, 1]
        return float(np.clip(prob, 0.01, 0.99))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="DatasetStore name or local parquet path")
    ap.add_argument("--out", required=True, help="Output model path (.xgb)")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()

    if args.dataset.endswith(".parquet") or args.dataset.startswith("/"):
        df = pd.read_parquet(args.dataset)
    else:
        df = DatasetStore().load(args.dataset)

    metrics = train(df, args.out, n_estimators=args.n_estimators, max_depth=args.max_depth)
    print("Metrics:", metrics)
