"""Features of a series' Polymarket price path up to its kickoff.

The kickoff price m0 is the last print within 30 minutes before kickoff (the price every entry is scored against). The
other features describe how the price got there: where it opened, where it stood hours before kickoff, how much it
moved and how stale the last print is. A thin market that has not moved since it opened carries less information than
its m0 suggests, and drift toward kickoff can carry information the opening line did not."""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK_HOURS = (1, 3, 6, 12, 24)
M0_WINDOW = pd.Timedelta(minutes=30)
EPS = 1e-3


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def path_features(paths: pd.DataFrame) -> pd.DataFrame:
    """One row per match_id: m0 (NaN without a print in the last 30 minutes), the last print and how old it is, the
    price at each lookback, the opening print, and path summaries. Every value comes from prints before kickoff."""
    p = paths[["match_id", "kickoff", "t", "p_a"]].copy()
    p["kickoff"] = _naive(p.kickoff)
    p["t"] = _naive(p.t)
    p = p[p.t <= p.kickoff].sort_values(["match_id", "t"], kind="stable")
    rows = []
    for mid, g in p.groupby("match_id", sort=True):
        t = g.t.to_numpy()
        x = g.p_a.to_numpy(float)
        ko = g.kickoff.iloc[0]
        ko64 = np.datetime64(ko)
        last_age_h = float((ko64 - t[-1]) / np.timedelta64(1, "h"))
        r = {"match_id": mid, "kickoff": ko, "n_prints": len(x), "p_last": x[-1], "last_age_h": last_age_h,
             "m0": x[-1] if last_age_h <= 0.5 else np.nan, "p_open": x[0],
             "open_age_h": float((ko64 - t[0]) / np.timedelta64(1, "h"))}
        for h in LOOKBACK_HOURS:
            i = np.searchsorted(t, ko64 - np.timedelta64(h * 60, "m"), side="right") - 1
            r[f"p_{h}h"] = x[i] if i >= 0 else np.nan
        lx = logit(x)
        recent = t >= ko64 - np.timedelta64(6, "h")
        r["n_changes"] = int((np.diff(x) != 0).sum())
        r["sd_6h"] = float(lx[recent].std()) if recent.sum() > 1 else 0.0
        r["mean_6h"] = float(lx[recent].mean()) if recent.any() else np.nan
        r["lmax"] = float(lx.max())
        r["lmin"] = float(lx.min())
        changes = np.flatnonzero(np.diff(x) != 0)
        r["since_change_h"] = float((ko64 - t[changes[-1] + 1]) / np.timedelta64(1, "h")) if len(changes) else r["open_age_h"]
        rows.append(r)
    return pd.DataFrame(rows)
