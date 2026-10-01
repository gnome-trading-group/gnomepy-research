"""
Temporal splitting and clustered-bootstrap metrics.

Two problems this fixes. Both trainers previously passed the test set as
`eval_set` for early stopping and then reported metrics on that same data, so
every published number was optimistic. And splits were taken on row index, which
tears a series in half — maps of one match landing on both sides of the boundary
share priors, opponents and a day, so the split leaks.

Effect sizes here are small relative to sampling noise: maps cluster within
series, so a ~2,500-map test fold carries an effective n nearer 1,100 and an AUC
standard error around +/-0.018. Point estimates from a single split cannot
separate a real 0.007 gain from noise, so comparisons use walk-forward folds
with bootstrap intervals clustered on the series.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def temporal_split_by_series(
    df: pd.DataFrame,
    frac_train: float = 0.70,
    frac_val: float = 0.15,
    date_col: str = "match_date",
    group_col: str = "match_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological train/val/test split whose boundaries fall between series.

    Returns positional index arrays. The val block is meant to be halved again by
    `split_val_for_calibration` so early stopping and calibration never share rows.
    """
    order = df.sort_values([date_col, group_col]).index.to_numpy()
    ordered = df.loc[order]
    groups = ordered[group_col].to_numpy()
    boundary = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])

    n = len(ordered)
    cuts = []
    for frac in (frac_train, frac_train + frac_val):
        target = frac * n
        idx = boundary[np.argmin(np.abs(boundary - target))]
        cuts.append(int(idx))
    lo, hi = cuts
    pos = {label: order[sl] for label, sl in
           (("train", slice(0, lo)), ("val", slice(lo, hi)), ("test", slice(hi, n)))}

    for label, idx in pos.items():
        if len(idx) == 0:
            raise ValueError(f"{label} split is empty — check fractions against {n} rows")
    logger.info(
        "temporal split: train=%d val=%d test=%d | test starts %s",
        len(pos["train"]), len(pos["val"]), len(pos["test"]),
        ordered[date_col].iloc[hi].date() if hi < n else "n/a",
    )
    return pos["train"], pos["val"], pos["test"]


def split_val_for_calibration(
    df: pd.DataFrame,
    val_idx: np.ndarray,
    date_col: str = "match_date",
    group_col: str = "match_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Halve the val block into an early-stopping slice and a calibration slice."""
    sub = df.loc[val_idx].sort_values([date_col, group_col])
    groups = sub[group_col].to_numpy()
    boundary = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    cut = int(boundary[np.argmin(np.abs(boundary - 0.5 * len(sub)))])
    order = sub.index.to_numpy()
    return order[:cut], order[cut:]


def walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    date_col: str = "match_date",
    group_col: str = "match_id",
    min_train_frac: float = 0.35,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds with series-aligned boundaries. Returns [(train_idx, test_idx)]."""
    order = df.sort_values([date_col, group_col]).index.to_numpy()
    ordered = df.loc[order]
    groups = ordered[group_col].to_numpy()
    boundary = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    n = len(ordered)

    folds = []
    edges = np.linspace(min_train_frac, 1.0, n_folds + 1)
    for lo_frac, hi_frac in zip(edges[:-1], edges[1:]):
        lo = int(boundary[np.argmin(np.abs(boundary - lo_frac * n))])
        hi = int(boundary[np.argmin(np.abs(boundary - hi_frac * n))]) if hi_frac < 1.0 else n
        if hi <= lo:
            continue
        folds.append((order[:lo], order[lo:hi]))
    logger.info("walk-forward: %d folds, test sizes %s", len(folds), [len(t) for _, t in folds])
    return folds


def clustered_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    (point estimate, lo, hi) resampling whole series rather than rows.

    Resampling rows would treat the two or three maps of a series as independent
    and understate the interval.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    point = float(metric(y_true, y_pred))
    stats = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        stats.append(metric(yt, y_pred[idx]))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def clustered_bootstrap_delta(
    y_true: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    groups: np.ndarray,
    metric,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired interval for metric(a) - metric(b) on the same resampled series."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    point = float(metric(y_true, y_a) - metric(y_true, y_b))
    stats = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        stats.append(metric(yt, y_a[idx]) - metric(yt, y_b[idx]))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile reliability table plus the calibration slope of y on logit(p)."""
    df = pd.DataFrame({"p": y_pred, "y": y_true})
    df["bucket"] = pd.qcut(df.p, n_bins, duplicates="drop")
    out = df.groupby("bucket", observed=True).agg(
        pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")
    ).reset_index(drop=True)
    return out


def calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Slope of a logistic refit on logit(p); 1.0 is calibrated, <1 overconfident."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(y_pred, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(fit_intercept=True).fit(z, y_true).coef_[0][0])
