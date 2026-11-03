"""
The calibration layer: a logistic regression of the series outcome on antisymmetric features (A-minus-B quantities and
rating gaps mapped through a bo3), fitted on walk-forward rows of past weeks. Swapping the teams flips every feature's
sign; an explicit intercept carries HLTV's habit of listing the favourite first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from gnomepy_research.sessions.cs2_comp.g15.ratings import bo3, regulation_map_win

THIN_MAPS90 = 15
THIN_RANK = 100


def clip_logit(p) -> np.ndarray:
    return logit(np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4))


def z_series(gap: np.ndarray) -> np.ndarray:
    """Logit of the bo3 probability implied by a round-logit gap (gap 0 for sides the engines never saw)."""
    g = np.nan_to_num(np.asarray(gap, float), nan=0.0)
    return clip_logit(bo3(regulation_map_win(g)))


def thin_flag(rows: pd.DataFrame) -> np.ndarray:
    maps90 = np.minimum(rows.maps90_a, rows.maps90_b)
    best = np.minimum(rows.rank_a, rows.rank_b)
    return ((rows.is_lan == 0) & ((maps90 <= THIN_MAPS90) | (best > THIN_RANK))).to_numpy(float)


GAP_SUFFIX = "_gap"


def design(rows: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Columns by name: z_<engine> is the bo3 logit of <engine>_gap (and _pgap / _tgap), with _thin and _lan
    interactions and, where the engine counts them, _few (players it barely saw) and _unseen interactions."""
    d = {}
    r = rows
    n = len(r)
    thin = thin_flag(r)
    lan = r.is_lan.to_numpy(float)
    d["one"] = np.ones(n)
    for c in r.columns:
        if c.endswith(GAP_SUFFIX) or c.endswith("_pgap") or c.endswith("_tgap"):
            name = "z_" + c.replace("_gap", "").replace("_pgap", "_p").replace("_tgap", "_t")
            z = z_series(r[c])
            d[name] = z
            d[name + "_thin"] = z * thin
            d[name + "_lan"] = z * lan
            prefix = c.split("_")[0]
            for kind in ("few", "unseen"):
                ca, cb = f"{prefix}_{kind}_a", f"{prefix}_{kind}_b"
                if ca in r:
                    tot = np.nan_to_num(r[ca].to_numpy(float), nan=5.0) + np.nan_to_num(r[cb].to_numpy(float), nan=5.0)
                    d[f"{name}_{kind}"] = z * tot
                    d[f"{prefix}_{kind}"] = np.nan_to_num(r[ca].to_numpy(float), nan=5.0) - np.nan_to_num(r[cb].to_numpy(float), nan=5.0)
    for c in r.columns:
        if c.endswith("_logit"):
            prefix = c.split("_")[0]
            fallback = d.get("z_" + prefix, np.zeros(n))
            v = r[c].to_numpy(float)
            d["z_" + c[:-6]] = np.where(np.isnan(v), fallback, v)
            d["z_" + c[:-6] + "_thin"] = d["z_" + c[:-6]] * thin
    few_a = np.nan_to_num(r.pm_new_a.to_numpy(float), nan=5.0) if "pm_new_a" in r else np.zeros(n)
    few_b = np.nan_to_num(r.pm_new_b.to_numpy(float), nan=5.0) if "pm_new_b" in r else np.zeros(n)
    d["few"] = few_a - few_b
    if "pk_sd" in r:
        d["z_pk_sd"] = d["z_pk"] * np.nan_to_num(r.pk_sd.to_numpy(float), nan=0.3)
    d["rank"] = r.log_rank_gap.to_numpy(float)
    d["points"] = r.log_points_gap.to_numpy(float)
    d["standin"] = (r.standins_a - r.standins_b).to_numpy(float)
    d["new"] = np.nan_to_num((r.new_a - r.new_b).to_numpy(float), nan=0.0) if "new_a" in r else np.zeros(n)
    d["idle"] = np.log1p(r.idle_a.to_numpy(float)) - np.log1p(r.idle_b.to_numpy(float))
    d["chem"] = np.nan_to_num((r.pk_chem_a - r.pk_chem_b).to_numpy(float), nan=0.0) if "pk_chem_a" in r else np.zeros(n)
    d["best"] = np.nan_to_num((r.pk_best_a - r.pk_best_b).to_numpy(float), nan=0.0) if "pk_best_a" in r else np.zeros(n)
    d["sd"] = np.nan_to_num((r.pk_sd_a - r.pk_sd_b).to_numpy(float), nan=0.0) if "pk_sd_a" in r else np.zeros(n)
    d["maps90"] = np.log1p(r.maps90_a.to_numpy(float)) - np.log1p(r.maps90_b.to_numpy(float))
    d["lan"] = lan
    d["thin"] = thin
    if "m0" in r:
        zm = clip_logit(np.nan_to_num(r.m0.to_numpy(float), nan=0.5))
        has = r.m0.notna().to_numpy(float)
        d["zm"] = zm
        d["zm_thin"] = zm * thin
        d["zm_stale"] = zm * np.log1p(np.nan_to_num(r.stale_min.to_numpy(float), nan=0.0))
        d["zm_prints"] = zm * np.log1p(np.nan_to_num(r.n_prints.to_numpy(float), nan=0.0))
        d["drift"] = zm - clip_logit(np.nan_to_num(r.p_24h.to_numpy(float), nan=0.5)) * has
        d["drift48"] = zm - clip_logit(np.nan_to_num(r.p_first.to_numpy(float), nan=0.5)) * has
        prints = np.log1p(np.nan_to_num(r.n_prints.to_numpy(float), nan=0.0))
        stale = np.log1p(np.nan_to_num(r.stale_min.to_numpy(float), nan=0.0))
        for name in [k for k in d if k.startswith("z_")]:
            d[name + "_prints"] = d[name] * prints
            d[name + "_stale"] = d[name] * stale
    return np.column_stack([d[c] for c in cols])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(100):
        p = expit(X @ w)
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1]), X.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
