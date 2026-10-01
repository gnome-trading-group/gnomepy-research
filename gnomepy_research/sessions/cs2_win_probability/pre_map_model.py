"""
Layer 1: P(team_a wins map) from pre-map context alone.

Feeds everything downstream — the map DP inverts this probability into an
implied per-round rate, and the series DP composes it across a veto.

Three departures from the previous trainer, each fixing a measured defect:
features come from cs2_match_priors through the single shared extractor rather
than a training-only copy that silently NaN'd recent form; early stopping uses a
validation block instead of the test set it then reported on; and training is
augmented with A/B-mirrored rows while inference averages both orientations, so
the model cannot exploit HLTV's habit of listing the favourite first.

Usage:
    poetry run python -m gnomepy_research.sessions.cs2_win_probability.pre_map_model \
        --out /tmp/cs2_pre_map_model.xgb
"""
from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.sessions.cs2_win_probability.pre_map_features import (
    _MAP_OHE_NAMES,
    PRE_MAP_FEATURE_NAMES,
)
from gnomepy_research.sessions.cs2_win_probability.splits import (
    calibration_slope,
    calibration_table,
    clustered_bootstrap_ci,
    split_val_for_calibration,
    temporal_split_by_series,
)
from gnomepy_research.sessions.cs2_win_probability.symmetry import (
    augment,
    build_swap_plan,
    swap_features,
    symmetric_predict_proba,
)

logger = logging.getLogger(__name__)

_ELO_FEATURES = ("elo_diff", "elo_map_diff", "elo_map_resid_diff", "elo_side_asym", "elo_games_min")


# A map winner is never certain, and an exact 0/1 makes log-loss and any Kelly
# sizing degenerate, so every calibrated probability is clipped on the way out.
# This lives in one place because the single-row path used to clip and the batch
# path did not, which is a train/serve skew by another name.
_PROB_EPS = 0.01


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _calibrated(temperature: float, raw: np.ndarray) -> np.ndarray:
    """
    Temperature scaling: divide the log-odds by T, then clip.

    This replaced isotonic regression, which was measurably destroying the signal
    it was meant to preserve. Isotonic is non-parametric, and on a ~500-row
    calibration split a 95-feature model's wider output range gave it enough rope
    to overfit: calibration slope 0.75 (predictions a third too extreme in
    log-odds) and a feature set that looked worthless at +0.0013 log-loss. One
    parameter instead of hundreds of knots turns that into -0.0153, and halves the
    run-to-run spread.

    It is also exactly symmetric - scaling log-odds commutes with p -> 1-p - so
    the A/B order-invariance survives calibration without the both-orientations
    fit isotonic needed.
    """
    return np.clip(1.0 / (1.0 + np.exp(-_logit(raw) / temperature)), _PROB_EPS, 1.0 - _PROB_EPS)


def _fit_temperature(p_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Minimise calibration-split log-loss over T. T>1 shrinks, T<1 sharpens."""
    result = minimize_scalar(
        lambda t: log_loss(y_cal, _calibrated(t, p_cal)),
        bounds=(0.2, 5.0), method="bounded",
    )
    return float(result.x)


class Calibrator:
    """
    Temperature scaling or isotonic, chosen per fit rather than assumed.

    Neither wins everywhere, and the difference is large enough to flip a
    conclusion. Isotonic is non-parametric: on ~2,500 calibration rows from a
    34-feature model it fits real curvature and wins, but on ~500 rows from a
    95-feature model - whose output spans a wider range - it overfits badly
    enough to make a genuinely useful feature set look worthless. Temperature
    scaling has one parameter and cannot do that, at the cost of being unable to
    represent curvature when there is enough data to estimate it.

    So both are cross-fitted within the calibration split and the better one is
    refitted on all of it. Isotonic is fitted on both orientations because it is
    not symmetric about 0.5 and would otherwise undo the A/B order-invariance;
    temperature scaling is symmetric by construction.
    """

    def __init__(self, kind: str, temperature: float, isotonic):
        self.kind = kind
        self.temperature = temperature
        self.isotonic = isotonic

    def predict(self, raw: np.ndarray) -> np.ndarray:
        if self.kind == "temperature":
            return _calibrated(self.temperature, raw)
        return np.clip(self.isotonic.predict(raw), _PROB_EPS, 1.0 - _PROB_EPS)


def _fit_isotonic(p_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.concatenate([p_cal, 1.0 - p_cal]), np.concatenate([y_cal, 1 - y_cal]))
    return iso


def fit_calibrator(p_cal: np.ndarray, y_cal: np.ndarray, seed: int = 0) -> Calibrator:
    """Cross-fit both calibrators inside the split, keep the better, refit it on all of it."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_cal))
    folds = ((idx[::2], idx[1::2]), (idx[1::2], idx[::2]))

    scores = {"temperature": 0.0, "isotonic": 0.0}
    for fit, ev in folds:
        if len(np.unique(y_cal[fit])) < 2 or len(np.unique(y_cal[ev])) < 2:
            continue
        t = _fit_temperature(p_cal[fit], y_cal[fit])
        scores["temperature"] += log_loss(y_cal[ev], _calibrated(t, p_cal[ev]))
        iso = _fit_isotonic(p_cal[fit], y_cal[fit])
        scores["isotonic"] += log_loss(
            y_cal[ev], np.clip(iso.predict(p_cal[ev]), _PROB_EPS, 1.0 - _PROB_EPS))

    kind = min(scores, key=scores.get)
    logger.info("calibrator: %s (cross-fit log-loss temp=%.4f iso=%.4f)",
                kind, scores["temperature"], scores["isotonic"])
    return Calibrator(kind, _fit_temperature(p_cal, y_cal), _fit_isotonic(p_cal, y_cal))


def load_training_frame() -> pd.DataFrame:
    """Priors joined to their map outcome, one row per map."""
    ds = DatasetStore()
    priors = ds.load("cs2_match_priors")
    history = ds.load("cs2_match_history")[["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]]
    df = priors.merge(history, on=["match_id", "map_name"], how="inner")
    drawn = df.team_a_score == df.team_b_score
    if drawn.any():
        logger.info("excluding %d drawn map(s) — team_a_won records a draw as a loss", int(drawn.sum()))
        df = df[~drawn]
    return df.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def _fit(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_es: np.ndarray, y_es: np.ndarray,
    n_estimators: int, max_depth: int,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=50,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    return model


def train(
    df: pd.DataFrame | None = None,
    out_path: str | None = None,
    n_estimators: int = 1000,
    max_depth: int = 5,
    feature_names: list[str] | None = None,
    symmetric: bool = True,
    report: bool = True,
) -> dict:
    """Train, calibrate and evaluate. Returns metrics; writes the bundle when out_path is given."""
    df = load_training_frame() if df is None else df
    names = feature_names or PRE_MAP_FEATURE_NAMES
    plan = build_swap_plan(names)

    X = _matrix(df, names)
    y = df["team_a_won"].to_numpy().astype(int)

    tr_idx, val_idx, te_idx = temporal_split_by_series(df)
    es_idx, cal_idx = split_val_for_calibration(df, val_idx)
    pos = {k: df.index.get_indexer(v) for k, v in
           (("tr", tr_idx), ("es", es_idx), ("cal", cal_idx), ("te", te_idx))}

    X_tr, y_tr = X[pos["tr"]], y[pos["tr"]]
    if symmetric:
        X_tr, y_tr = augment(X_tr, y_tr, plan)
    X_es, y_es = X[pos["es"]], y[pos["es"]]

    model = _fit(X_tr, y_tr, X_es, y_es, n_estimators, max_depth)
    logger.info("best iteration: %d", model.best_iteration)

    predict = (lambda M: symmetric_predict_proba(model, M, plan)) if symmetric \
        else (lambda M: model.predict_proba(M)[:, 1])

    calibrator = fit_calibrator(predict(X[pos["cal"]]), y[pos["cal"]])
    p_te = calibrator.predict(predict(X[pos["te"]]))
    y_te = y[pos["te"]]
    groups = df.iloc[pos["te"]]["match_id"].to_numpy()

    auc, auc_lo, auc_hi = clustered_bootstrap_ci(y_te, p_te, groups, roc_auc_score)
    ll, ll_lo, ll_hi = clustered_bootstrap_ci(y_te, p_te, groups, log_loss)
    base_rate = float(y[pos["tr"]].mean())
    metrics = {
        "auc": auc, "auc_ci": (auc_lo, auc_hi),
        "log_loss": ll, "log_loss_ci": (ll_lo, ll_hi),
        "brier": float(brier_score_loss(y_te, p_te)),
        "base_rate_log_loss": float(log_loss(y_te, np.full(len(y_te), base_rate))),
        "calibration_slope": calibration_slope(y_te, p_te),
        "best_iteration": int(model.best_iteration),
        "n_train": int(len(pos["tr"])), "n_test": int(len(y_te)),
        "n_features": len(names),
        "calibrator": calibrator.kind,
        "temperature": calibrator.temperature,
        "test_pred": p_te,
        "test_true": y_te,
        "test_groups": groups,
    }

    if report:
        logger.info(
            "test AUC=%.4f [%.4f, %.4f]  logloss=%.4f [%.4f, %.4f]  (base rate %.4f)  slope=%.3f",
            auc, auc_lo, auc_hi, ll, ll_lo, ll_hi, metrics["base_rate_log_loss"],
            metrics["calibration_slope"],
        )
        logger.info("calibration:\n%s", calibration_table(y_te, p_te).round(3).to_string())

    if symmetric:
        mirror = calibrator.predict(predict(swap_features(X[pos["te"]], plan)))
        metrics["symmetry_residual"] = float(np.abs(p_te + mirror - 1).mean())

    if out_path:
        joblib.dump({"model": model, "calibrator": calibrator,
                     "feature_names": names, "symmetric": symmetric}, out_path)
        logger.info("saved bundle to %s", out_path)
    return metrics


def _matrix(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Feature matrix restricted to `names`, so ablations drop columns without reshaping."""
    scalars = [n for n in names if n not in _MAP_OHE_NAMES]
    cols = df.reindex(columns=scalars).astype(np.float32).to_numpy()
    ohe_names = [n for n in names if n in _MAP_OHE_NAMES]
    if not ohe_names:
        return cols
    maps = df["map_name"].to_numpy()
    ohe = np.stack([(maps == n.removeprefix("map_")).astype(np.float32) for n in ohe_names], axis=1)
    return np.hstack([cols, ohe])


class CS2PreMapModel:
    """Inference wrapper. Always order-invariant when the bundle was trained symmetric."""

    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self._model = bundle["model"]
        self._calibrator = bundle["calibrator"]
        self._names = bundle["feature_names"]
        self._symmetric = bundle["symmetric"]
        self._plan = build_swap_plan(self._names)

    @property
    def feature_names(self) -> list[str]:
        return list(self._names)

    def predict_team_a_win_prob(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(self.predict_batch(x)[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        raw = symmetric_predict_proba(self._model, X, self._plan) if self._symmetric \
            else self._model.predict_proba(X)[:, 1]
        return self._calibrator.predict(raw)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--ablate", action="store_true", help="also report Elo and symmetry ablations")
    args = ap.parse_args()

    frame = load_training_frame()
    full = train(frame, args.out, args.n_estimators, args.max_depth)
    print("full:", {k: v for k, v in full.items() if k != "auc_ci"})

    if args.ablate:
        no_elo = [n for n in PRE_MAP_FEATURE_NAMES if n not in _ELO_FEATURES]
        for label, kwargs in (
            ("no_elo", {"feature_names": no_elo}),
            ("no_symmetry", {"symmetric": False}),
            ("no_elo_no_symmetry", {"feature_names": no_elo, "symmetric": False}),
        ):
            m = train(frame, None, args.n_estimators, args.max_depth, report=False, **kwargs)
            print(f"{label}: AUC {m['auc']:.4f} {m['auc_ci']}  logloss {m['log_loss']:.4f}")
