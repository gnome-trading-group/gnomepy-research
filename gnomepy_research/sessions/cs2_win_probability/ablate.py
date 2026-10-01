"""
Per-block ablation for the harvested features.

Each block is added to the pre-harvest baseline on its own, and the whole set is
added together, always against the same temporal test split. Deltas are paired
clustered bootstraps - resampling whole series, because maps cluster within a
series and row-level resampling would understate the interval by roughly the
square root of the cluster size.

Read the intervals, not the point estimates. At the sample sizes here the SE on
AUC is around +/-0.018, so a block can look helpful and be noise. Two earlier
augmentations died exactly this way.

    python -m gnomepy_research.sessions.cs2_win_probability.ablate \\
        --priors /tmp/priors.parquet --history /tmp/history.parquet
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from gnomepy_research.pipelines.hltv_cs2.context_features import (
    EVENT_FEATURES,
    PAGE_H2H_FEATURES,
    RANK_FEATURES,
    SCHEDULE_FEATURES,
    VETO_FEATURES,
)
from gnomepy_research.pipelines.hltv_cs2.player_form import PLAYER_FORM_FEATURES
from gnomepy_research.sessions.cs2_win_probability.pre_map_features import (
    HARVEST_FEATURE_NAMES,
    PRE_MAP_FEATURE_NAMES,
)
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import train
from gnomepy_research.sessions.cs2_win_probability.splits import clustered_bootstrap_delta

logger = logging.getLogger(__name__)

BLOCKS = {
    "player_form": PLAYER_FORM_FEATURES,
    "schedule": SCHEDULE_FEATURES,
    "rank": RANK_FEATURES,
    "event": EVENT_FEATURES,
    "veto": VETO_FEATURES,
    "page_h2h": PAGE_H2H_FEATURES,
}

BASELINE_NAMES = [n for n in PRE_MAP_FEATURE_NAMES if n not in set(HARVEST_FEATURE_NAMES)]


def _frame(priors: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
    df = priors.merge(history[cols], on=["match_id", "map_name"], how="inner")
    drawn = df.team_a_score == df.team_b_score
    if drawn.any():
        df = df[~drawn]
    return df.sort_values(["match_date", "match_id"]).reset_index(drop=True)


def _fit_repeats(df: pd.DataFrame, names: list[str], n_repeats: int, seed: int) -> dict:
    """
    Refit under several column orders and keep every result.

    XGBoost's colsample_bytree samples columns positionally, so the same feature
    set in a different order is a different model. On three months that reorder
    moves log-loss by more than any block effect, which is enough to invent a
    significant result from nothing. Averaging makes the comparison reproducible;
    keeping the spread makes the fit noise visible instead of implicit.
    """
    rng = np.random.default_rng(seed)
    preds, lls, aucs, slopes = [], [], [], []
    for _ in range(n_repeats):
        shuffled = list(names)
        rng.shuffle(shuffled)
        m = train(df, feature_names=shuffled, report=False)
        preds.append(m["test_pred"])
        lls.append(m["log_loss"])
        aucs.append(m["auc"])
        slopes.append(m["calibration_slope"])
    return {
        "pred": np.mean(preds, axis=0),
        "per_fit": np.array(preds),
        "log_loss": float(np.mean(lls)), "log_loss_sd": float(np.std(lls)),
        "auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
        "cal_slope": float(np.mean(slopes)),
        "true": m["test_true"], "groups": m["test_groups"], "n_features": m["n_features"],
    }


def run(df: pd.DataFrame, n_boot: int = 500, n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Fit the baseline and each variant, returning deltas against the baseline.

    Two noise sources are reported because they are different and both bite.
    The clustered bootstrap resamples whole series, covering test-set sampling
    variability with the fit held fixed. `fit_sd` is the spread across column
    orders, covering fit variability with the data held fixed. A block only
    counts when it clears both.
    """
    variants = {"baseline": BASELINE_NAMES, "all_blocks": PRE_MAP_FEATURE_NAMES}
    for name, block in BLOCKS.items():
        variants[name] = BASELINE_NAMES + [c for c in block if c in df.columns]

    fitted = {}
    for name, names in variants.items():
        usable = [n for n in names if n in df.columns or n.startswith("map_de_")]
        logger.info("fitting %s (%d features) x%d orders...", name, len(usable), n_repeats)
        fitted[name] = _fit_repeats(df, usable, n_repeats, seed)

    base = fitted["baseline"]
    y, groups = base["true"], base["groups"]
    base_ll = np.array([log_loss(y, p) for p in base["per_fit"]])

    rows = []
    for name, m in fitted.items():
        d_ll, ll_lo, ll_hi = clustered_bootstrap_delta(
            y, m["pred"], base["pred"], groups, log_loss, n_boot=n_boot, seed=seed)
        d_auc, _, _ = clustered_bootstrap_delta(
            y, m["pred"], base["pred"], groups, roc_auc_score, n_boot=n_boot, seed=seed)

        paired = np.array([log_loss(y, p) for p in m["per_fit"]]) - base_ll
        clears_fit_noise = bool(abs(paired.mean()) > 2 * paired.std()) if name != "baseline" else False
        rows.append({
            "variant": name,
            "n_features": m["n_features"],
            "auc": m["auc"], "auc_sd": m["auc_sd"],
            "log_loss": m["log_loss"], "fit_sd": m["log_loss_sd"],
            "d_log_loss": d_ll, "ll_lo": ll_lo, "ll_hi": ll_hi,
            "boot_significant": bool(ll_hi < 0 or ll_lo > 0),
            "clears_fit_noise": clears_fit_noise,
            "robust": bool((ll_hi < 0 or ll_lo > 0) and clears_fit_noise),
            "d_auc": d_auc,
            "cal_slope": m["cal_slope"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-repeats", type=int, default=5,
                    help="column orders per variant; 1 reproduces the old single-fit behaviour")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = _frame(pd.read_parquet(args.priors), pd.read_parquet(args.history))
    logger.info("training frame: %d rows, %d series", len(df), df.match_id.nunique())
    table = run(df, n_boot=args.n_boot, n_repeats=args.n_repeats)

    show = table[["variant", "n_features", "auc", "auc_sd", "log_loss", "fit_sd",
                  "d_log_loss", "ll_lo", "ll_hi", "robust", "cal_slope"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out:
        table.to_parquet(args.out, index=False)
