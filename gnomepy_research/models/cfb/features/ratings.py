"""Opponent-adjusted offensive/defensive ratings by weighted ridge regression.

    off_ppa[g, i] = mu + O[i] - D[opp] + h * venue + e,   weight = plays

A transparent, non-Bayesian estimate used as (a) a season-to-date feature, (b) the prior-season
rating input to preseason priors, and (c) the input to the non-Bayesian logistic baseline. All
non-FBS teams in a season share one pooled rating, so FCS opponents are never treated as average
FBS teams. Higher O = better offense, higher D = better defense.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr

FCS_TEAM = -1
# Penalty per unit of rating; 400 plays ~ 6 games of evidence before a rating moves halfway to its
# data-only estimate. Tuned in the efficiency-model milestone, not here.
DEFAULT_LAMBDA = 400.0


def ridge_ratings(eff: pd.DataFrame, lam: float = DEFAULT_LAMBDA) -> tuple[pd.DataFrame, dict]:
    """eff: team_game_efficiency rows (already point-in-time filtered). Returns one row per team
    (team_id, off_rating, def_rating, games, plays) and the fitted globals {mu, hfa}."""
    e = eff[eff["usable"]].copy()
    if e.empty:
        return pd.DataFrame(columns=["team_id", "off_rating", "def_rating", "games", "plays"]), {"mu": np.nan, "hfa": np.nan}
    e["off_team"] = np.where(e["team_classification"] == "fbs", e["team_id"], FCS_TEAM)
    e["def_team"] = np.where(e["opponent_classification"] == "fbs", e["opponent_id"], FCS_TEAM)
    teams = np.unique(np.concatenate([e["off_team"].to_numpy(), e["def_team"].to_numpy()]))
    idx = {t: i for i, t in enumerate(teams)}
    n_t, n = len(teams), len(e)

    w = np.sqrt(e["plays"].to_numpy(dtype=float))
    rows = np.repeat(np.arange(n), 4)
    o_col = 2 + np.array([idx[t] for t in e["off_team"]])
    d_col = 2 + n_t + np.array([idx[t] for t in e["def_team"]])
    cols = np.column_stack([np.zeros(n, int), np.ones(n, int), o_col, d_col]).ravel()
    vals = np.column_stack([np.ones(n), e["venue"].to_numpy(float), np.ones(n), -np.ones(n)]).ravel()
    X = csr_matrix((vals * np.repeat(w, 4), (rows, cols)), shape=(n, 2 + 2 * n_t))
    y = e["off_ppa"].to_numpy(float) * w

    # Ridge on team effects only: augment with sqrt(lam) * I rows for the 2*n_t team columns.
    pen_rows = np.arange(2 * n_t)
    P = csr_matrix((np.full(2 * n_t, np.sqrt(lam)), (pen_rows, 2 + pen_rows)), shape=(2 * n_t, 2 + 2 * n_t))
    coef = lsqr(vstack([X, P]).tocsr(), np.concatenate([y, np.zeros(2 * n_t)]), atol=1e-10, btol=1e-10)[0]

    mu, hfa = coef[0], coef[1]
    off, de = coef[2:2 + n_t], coef[2 + n_t:]
    counts = e.groupby("off_team").agg(games=("game_id", "nunique"), plays=("plays", "sum"))
    out = pd.DataFrame({"team_id": teams, "off_rating": off, "def_rating": de})
    out = out.merge(counts, left_on="team_id", right_index=True, how="left").fillna({"games": 0, "plays": 0})
    return out, {"mu": float(mu), "hfa": float(hfa)}

