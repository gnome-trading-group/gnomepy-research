"""Kickoff price and pre-kickoff path summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAST_PRINT_WINDOW = pd.Timedelta(minutes=30)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def path_features(paths: pd.DataFrame) -> pd.DataFrame:
    p = paths.copy()
    p["t"] = pd.to_datetime(p.t).dt.tz_convert(None) if pd.to_datetime(p.t).dt.tz is not None else pd.to_datetime(p.t)
    k = pd.to_datetime(p.kickoff)
    p["kickoff"] = k.dt.tz_convert(None) if k.dt.tz is not None else k
    p["mins"] = (p.kickoff - p.t).dt.total_seconds() / 60.0
    rows = []
    for mid, g in p.sort_values("t").groupby("match_id"):
        pre = g[g.mins >= 0]
        last = pre[pre.mins <= LAST_PRINT_WINDOW.total_seconds() / 60.0]
        r = {"match_id": mid, "n_prints": float(len(pre)), "m0": np.nan}
        if len(last):
            r["m0"] = float(last.p_a.iloc[-1])
        if len(pre):
            ref = pre.p_a.iloc[-1] if np.isnan(r["m0"]) else r["m0"]
            r["m_last"] = float(pre.p_a.iloc[-1])
            for h in (1, 6, 24, 48):
                old = pre[pre.mins >= h * 60]
                r[f"m_ago{h}"] = float(old.p_a.iloc[-1]) if len(old) else np.nan
            r["m_std6"] = float(pre[pre.mins <= 360].p_a.std()) if (pre.mins <= 360).sum() > 2 else np.nan
            r["first_mins"] = float(pre.mins.max())
            r["last_mins"] = float(pre.mins.min())
        rows.append(r)
    return pd.DataFrame(rows)
