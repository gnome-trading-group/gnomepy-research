"""Per-series features of a Polymarket price path up to kickoff.

m0 is the last print within 30 minutes before kickoff (the price the competition scores against); without one, the
last print at all stands in. The earlier prices are the last print at or before each horizon, falling back to m0, so a
missing horizon reads as "no move". The path's activity (how many times the price changed in the last six hours) says
how much trading stands behind m0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g07.data import naive

HORIZONS = {"p1": 60, "p3": 180, "p6": 360, "p24": 1440}
M0_WINDOW = 30.0


def path_features(paths: pd.DataFrame) -> pd.DataFrame:
    """paths: match_id, kickoff, t, p_a. One row per match_id."""
    if paths.empty:
        return pd.DataFrame(columns=["match_id", "m0", "has_m0", "mean6", "nchg6", "nprint", *HORIZONS])
    d = paths.sort_values(["match_id", "t"], kind="stable")
    mid = d.match_id.to_numpy()
    before = ((naive(d.kickoff) - naive(d.t)).dt.total_seconds() / 60.0).to_numpy()
    p = d.p_a.to_numpy(float)
    starts = np.r_[0, np.flatnonzero(mid[1:] != mid[:-1]) + 1]
    ends = np.r_[starts[1:], len(mid)]
    rows = []
    for s, e in zip(starts, ends):
        b, q = before[s:e], p[s:e]
        ok = b >= 0
        b, q = b[ok], q[ok]
        if len(q) == 0:
            rows.append({"match_id": mid[s], "m0": np.nan, "has_m0": 0.0, "nprint": 0})
            continue
        has = b[-1] <= M0_WINDOW
        m0 = q[-1]
        r = {"match_id": mid[s], "m0": m0, "has_m0": float(has), "nprint": len(q)}
        for name, h in HORIZONS.items():
            i = np.flatnonzero(b >= h)
            r[name] = q[i[-1]] if len(i) else m0
        last6 = b <= 360
        r["mean6"] = q[last6].mean() if last6.any() else m0
        chg = np.r_[False, np.diff(q) != 0]
        r["nchg6"] = int((chg & last6).sum())
        rows.append(r)
    return pd.DataFrame(rows)
