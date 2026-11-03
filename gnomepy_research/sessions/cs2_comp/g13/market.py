"""
Per-series features from a Polymarket price path up to kickoff.

The kickoff price is the last print within 30 minutes before kickoff; the rest describe how much the path can be
trusted (how many prints, whether it ever moved) and where it came from (earlier levels of the same path).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KICKOFF_WINDOW = pd.Timedelta(minutes=30)
PATH_COLUMNS = ["m0", "m_last1h", "m_last2h", "m_last3h", "m_last6h", "m_at6h", "m_at24h", "n_prints", "n_last6h", "n_distinct", "mins_stale", "flat"]


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _level_at(t: np.ndarray, p: np.ndarray, at: np.datetime64) -> float:
    """The last print at or before `at`, or the first print if none."""
    i = np.searchsorted(t, at, side="right") - 1
    return float(p[max(i, 0)])


def path_features(paths: pd.DataFrame) -> pd.DataFrame:
    """One row per match_id from prints (match_id, kickoff, t, p_a); a series without prints gets no row."""
    d = paths.assign(t=_naive(paths.t), kickoff=_naive(paths.kickoff)).sort_values(["match_id", "t"])
    rows = []
    for mid, g in d.groupby("match_id", sort=False):
        t = g.t.to_numpy()
        p = g.p_a.to_numpy(float)
        k = np.datetime64(g.kickoff.iloc[0])
        before = t <= k
        t, p = t[before], p[before]
        if len(t) == 0:
            continue
        last = len(t) - 1
        mins_stale = float((k - t[last]) / np.timedelta64(1, "m"))
        in_window = t >= k - np.timedelta64(30, "m")
        m0 = float(p[in_window][-1]) if in_window.any() else float(p[last])
        h1, h2, h3 = (t >= k - np.timedelta64(n, "h") for n in (1, 2, 3))
        h6 = t >= k - np.timedelta64(6, "h")
        rows.append({
            "match_id": mid, "m0": m0,
            "m_last1h": float(p[h1].mean()) if h1.any() else m0,
            "m_last2h": float(p[h2].mean()) if h2.any() else m0,
            "m_last3h": float(p[h3].mean()) if h3.any() else m0,
            "m_last6h": float(p[h6].mean()) if h6.any() else m0,
            "m_at6h": _level_at(t, p, k - np.timedelta64(6, "h")),
            "m_at24h": _level_at(t, p, k - np.timedelta64(24, "h")),
            "n_prints": int(len(t)), "n_last6h": int(h6.sum()), "n_distinct": int(len(np.unique(p))),
            "mins_stale": mins_stale, "flat": int(p.min() == p.max()),
        })
    return pd.DataFrame(rows, columns=["match_id"] + PATH_COLUMNS)
