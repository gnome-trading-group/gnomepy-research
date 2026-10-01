"""
Held-out evaluation of the map DP at halftime.

Halftime is the only mid-map state observable across the whole dataset: for
95.5% of maps we have the final score, the half score and the starting side, and
nothing in between. Within-half states are checkable only on the demo subset.

Baselines the DP must clear, all measured on the same held-out maps:
  0.8631  score race with p=0.5 and no team strength   (hard floor)
  0.8698  XGBoost trained directly on the halftime state
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from gnomepy_research.sessions.cs2_win_probability.map_dp import (
    map_win_prob,
    map_win_prob_from_pre_map,
)
from gnomepy_research.sessions.cs2_win_probability.round_rate_model import MapSideBias
from gnomepy_research.sessions.cs2_win_probability.splits import (
    calibration_slope,
    calibration_table,
    clustered_bootstrap_ci,
    clustered_bootstrap_delta,
)

logger = logging.getLogger(__name__)

HALFTIME_FLOOR_AUC = 0.8616


def halftime_dp_probs(
    df: pd.DataFrame,
    pre_map_probs: np.ndarray,
    bias: MapSideBias,
) -> np.ndarray:
    """Price each map's halftime score from its pre-map probability."""
    out = np.empty(len(df), dtype=float)
    for i, (row, p0) in enumerate(zip(df.itertuples(), pre_map_probs)):
        delta = bias.delta(row.map_name, getattr(row, "elo_side_asym", 0.0))
        out[i] = map_win_prob_from_pre_map(
            float(p0), delta,
            int(row.team_a_h1_score), int(row.team_b_h1_score),
            bool(row.team_a_started_ct),
        )
    return out


def naive_score_race(df: pd.DataFrame) -> np.ndarray:
    """p=0.5 race from the halftime score — the floor the DP must beat."""
    return np.array([
        map_win_prob(0.5, 0.5, int(r.team_a_h1_score), int(r.team_b_h1_score), bool(r.team_a_started_ct))
        for r in df.itertuples()
    ])


def evaluate_halftime(
    df: pd.DataFrame,
    pre_map_probs: np.ndarray,
    bias: MapSideBias,
    calibrator: IsotonicRegression | None = None,
) -> dict:
    """AUC / log-loss / calibration against realised map outcomes, with clustered intervals."""
    y = df["team_a_won"].to_numpy().astype(int)
    groups = df["match_id"].to_numpy()

    raw = halftime_dp_probs(df, pre_map_probs, bias)
    p = calibrator.predict(raw) if calibrator is not None else raw
    floor = naive_score_race(df)

    auc, auc_lo, auc_hi = clustered_bootstrap_ci(y, p, groups, roc_auc_score)
    ll, ll_lo, ll_hi = clustered_bootstrap_ci(y, p, groups, log_loss)
    d_auc, d_lo, d_hi = clustered_bootstrap_delta(y, p, floor, groups, roc_auc_score)

    metrics = {
        "auc": auc, "auc_ci": (auc_lo, auc_hi),
        "log_loss": ll, "log_loss_ci": (ll_lo, ll_hi),
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": calibration_slope(y, p),
        "floor_auc": float(roc_auc_score(y, floor)),
        "floor_log_loss": float(log_loss(y, floor)),
        "auc_vs_floor": d_auc, "auc_vs_floor_ci": (d_lo, d_hi),
        "pre_map_auc": float(roc_auc_score(y, pre_map_probs)),
        "n": int(len(y)),
        "passes_floor": bool(auc >= HALFTIME_FLOOR_AUC),
        "probs": p,
    }
    logger.info(
        "halftime DP: AUC %.4f [%.4f, %.4f]  logloss %.4f  slope %.3f  (floor %.4f, pre-map %.4f)",
        auc, auc_lo, auc_hi, ll, metrics["calibration_slope"],
        metrics["floor_auc"], metrics["pre_map_auc"],
    )
    logger.info("calibration:\n%s", calibration_table(y, p).round(3).to_string())
    return metrics


def fit_dp_calibrator(
    df: pd.DataFrame,
    pre_map_probs: np.ndarray,
    bias: MapSideBias,
) -> IsotonicRegression:
    """
    Correct the DP's overconfidence at the extremes.

    The recursion treats rounds as independent; they are not — economy cycles and
    momentum correlate them — so realised outcomes are less extreme than the DP
    implies. Fitting on both orientations keeps the correction symmetric.
    """
    raw = halftime_dp_probs(df, pre_map_probs, bias)
    y = df["team_a_won"].to_numpy().astype(int)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.concatenate([raw, 1.0 - raw]), np.concatenate([y, 1 - y]))
    return iso
