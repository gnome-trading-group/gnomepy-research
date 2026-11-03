"""
The g09 layer: a ridge logistic regression of the series result on market and rating features, fitted on past series
with a market and their point-in-time features.

Features (all antisymmetric in A and B apart from the intercept, which carries HLTV's team-A ordering):
  z0     logit of the kickoff price (the latest print when none falls in the last 30 minutes; 0 without a market)
  d3     time-weighted mean logit over the last three hours minus z0: the last hour's moves partly revert
  ze     player power rating, priced as a bo3 from the map-level gap
  rk     log HLTV rank of B minus that of A (unranked counts as 300)
A series without any price print gets z0 = d3 = 0 and the no-market column, so the ratings carry it alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g09.core import UNRANKED, bo3, logit

FEATURES = ("zavg", "form", "mapadv")
TRAIN_DAYS = 365.0
L2 = 1.0


def design(rows: pd.DataFrame, features=FEATURES) -> np.ndarray:
    has = rows.m0.notna() | rows.zlast.notna()
    z0 = np.where(rows.m0.notna(), logit(rows.m0.to_numpy(float)), rows.zlast.to_numpy(float))
    z0 = np.where(has, z0, 0.0)
    d3 = np.nan_to_num(rows.tw3.to_numpy(float) - z0) * has
    pe = 1 / (1 + np.exp(-rows.elo_gap.to_numpy(float)))
    ze = logit(bo3(pe))
    rk = np.log(rows.hrank_b.fillna(UNRANKED).to_numpy(float)) - np.log(rows.hrank_a.fillna(UNRANKED).to_numpy(float))
    no = (~has).to_numpy(float)
    best = np.fmin(rows.hrank_a.to_numpy(float), rows.hrank_b.to_numpy(float))
    m90 = np.fmin(rows.maps90_a.to_numpy(float), rows.maps90_b.to_numpy(float)) if "maps90_a" in rows else np.zeros(len(rows))
    thin = ((rows.is_lan.to_numpy(float) == 0) & ((m90 <= 15) | ~(best <= 100))).astype(float)
    lan = rows.is_lan.to_numpy(float)
    liq = np.log1p(rows.nchg3.fillna(0).to_numpy(float)) - 3.5
    cols = {"z0": z0, "d3": d3, "ze": ze, "rk": rk, "nomkt": no, "nomkt_ze": no * ze, "nomkt_rk": no * rk,
            "liq_d3": d3 * liq, "liq_z0": z0 * liq, "liq_ze": ze * liq, "thin": thin, "thin_z0": thin * z0,
            "thin_ze": thin * ze, "lan_ze": lan * ze, "lan_z0": lan * z0}
    variants = [ze]
    for c in rows.columns:
        if c.startswith("elo_gap_"):
            pc = 1 / (1 + np.exp(-rows[c].to_numpy(float)))
            cols["ze_" + c[len("elo_gap_"):]] = logit(bo3(pc))
            variants.append(cols["ze_" + c[len("elo_gap_"):]])
    cols["zavg"] = np.mean(variants, axis=0)
    mm = np.fmin(rows.minmaps_a.to_numpy(float), rows.minmaps_b.to_numpy(float))
    cols["zavg_mm"] = cols["zavg"] * np.exp(-mm / 30.0)
    cols["zavg_known"] = cols["zavg"] * (10.0 - rows.known_a.to_numpy(float) - rows.known_b.to_numpy(float))
    cols["zavg_thin"] = cols["zavg"] * thin
    cols["zavg_lan"] = cols["zavg"] * lan
    return np.column_stack([cols[f] if f in cols else rows[f].to_numpy(float) for f in features] + [np.ones(len(rows))])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = L2) -> np.ndarray:
    w = np.zeros(X.shape[1])
    pen = l2 * np.ones(X.shape[1])
    pen[-1] = 1e-6
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen), X.T @ (y - p) - pen * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def fit_predict(train: pd.DataFrame, targets: pd.DataFrame, features=FEATURES, l2: float = L2) -> np.ndarray:
    cutoff = targets.snap.max()
    train = train[train.kickoff >= cutoff - pd.Timedelta(days=TRAIN_DAYS)]
    w = fit_logistic(design(train, features), train.y.to_numpy(float), l2)
    return 1 / (1 + np.exp(-design(targets, features) @ w))
