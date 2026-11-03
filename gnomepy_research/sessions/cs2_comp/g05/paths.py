"""Per-series summaries of a Polymarket price path up to kickoff.

The kickoff price m0 is the last print within 30 minutes of kickoff. A minute print carries bid-ask bounce and one-off
trades that later revert, so an exponentially weighted mean of the last half hour of logit prices is a less noisy
estimate of where the market stands. The other summaries describe how alive the market is (price changes, minutes
since the last change) and how the price travelled to kickoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KICKOFF_WINDOW_MIN = 30.0
EWM_HALF_LIFE_MIN = 30.0
COLUMNS = ["m0", "has_path", "z0", "e30", "z360", "z1440", "changes", "changes60", "since_change", "prints"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def summarize(paths: pd.DataFrame) -> pd.DataFrame:
    """paths: match_id, kickoff, t, p_a (prints before kickoff). One row per match_id, indexed by it."""
    if paths.empty:
        return pd.DataFrame(columns=COLUMNS, index=pd.Index([], name="match_id"))
    d = paths[["match_id", "kickoff", "t", "p_a"]].copy()
    d["mb"] = (_naive(d.kickoff) - _naive(d.t)).dt.total_seconds().to_numpy() / 60.0
    d = d[d.mb >= 0].sort_values(["match_id", "t"], kind="stable")
    rows = {}
    for mid, s in d.groupby("match_id", sort=False):
        mb = s.mb.to_numpy()
        p = s.p_a.to_numpy(float)
        z = _logit(p)
        recent = mb <= KICKOFF_WINDOW_MIN
        m0 = p[recent][-1] if recent.any() else p[-1]
        w = 0.5 ** (mb / EWM_HALF_LIFE_MIN)
        changed = np.nonzero(np.diff(p))[0]
        rows[mid] = {
            "m0": m0, "has_path": 1.0, "z0": float(_logit(m0)), "e30": float((w * z).sum() / w.sum()),
            "z360": float(z[np.searchsorted(-mb, -360.0) - 1]) if mb[0] > 360 else float(z[0]),
            "z1440": float(z[np.searchsorted(-mb, -1440.0) - 1]) if mb[0] > 1440 else float(z[0]),
            "changes": float(len(changed)), "changes60": float((mb[changed + 1] <= 60).sum()),
            "since_change": float(mb[changed[-1] + 1]) if len(changed) else float(mb[0]), "prints": float(len(p))}
    out = pd.DataFrame.from_dict(rows, orient="index")[COLUMNS]
    out.index.name = "match_id"
    return out


def complete(summary: pd.DataFrame, match_ids) -> pd.DataFrame:
    """Rows for every match_id; a series without prints gets an even price and has_path 0."""
    out = summary.reindex(pd.Index(list(match_ids), name="match_id"))
    missing = out.m0.isna()
    out.loc[missing, ["m0", "has_path", "z0", "e30", "z360", "z1440", "changes", "changes60", "since_change", "prints"]] = \
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2880.0, 0.0]
    return out
