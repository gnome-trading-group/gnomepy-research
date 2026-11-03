"""
Features of a series' Polymarket path up to kickoff.

The kickoff price is one print of a noisy process: in a thin book the last print can be a single stale or wide
quote. A time average over the final window is a less noisy reading of the same quantity, and how many prints left
the 0.5 placeholder says how much the book has actually been traded, which sets how much the reading is worth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLACEHOLDER = 0.5
KICKOFF_WINDOW_MIN = 30


def path_features(paths: pd.DataFrame, window_min: int = 120) -> pd.DataFrame:
    """One row per match_id: m_last, m0 (last print within 30 min of kickoff), twap over the final window, print
    counts, hours of active trading and the spread of the final window."""
    if paths.empty:
        return pd.DataFrame(columns=["match_id", "m_last", "m0", "twap", "n", "n_active", "active_hours", "sd_window", "n_window"])
    p = paths.sort_values(["match_id", "t"], kind="stable")
    minutes = (p.kickoff - p.t).dt.total_seconds().to_numpy() / 60.0
    pa = p.p_a.to_numpy(float)
    d = pd.DataFrame({"match_id": p.match_id.to_numpy(), "min": minutes, "p": pa, "active": pa != PLACEHOLDER})
    d["p_active"] = np.where(d.active, d["p"], np.nan)
    d["min_active"] = np.where(d.active, d["min"], np.nan)
    last = d.groupby("match_id").agg(m_last=("p", "last"), n=("p", "size"), n_active=("active", "sum"),
                                     first_active_min=("min_active", "max"))
    k = d[d["min"] <= KICKOFF_WINDOW_MIN].groupby("match_id").p.last().rename("m0")
    w = d[d["min"] <= window_min].groupby("match_id").p.agg(twap="mean", sd_window="std", n_window="size")
    out = last.join(k).join(w).reset_index()
    out["active_hours"] = out.first_active_min.fillna(0.0) / 60.0
    out["twap"] = out.twap.fillna(out.m_last)
    out["sd_window"] = out.sd_window.fillna(0.0)
    out["n_window"] = out.n_window.fillna(0).astype(int)
    return out.drop(columns=["first_active_min"])
