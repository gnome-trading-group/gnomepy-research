"""Per-series summaries of a Polymarket price path up to kickoff."""
from __future__ import annotations

import numpy as np
import pandas as pd

KICKOFF_WINDOW_MIN = 30.0
LAGS_MIN = (5, 15, 30, 60, 120, 180, 360, 720, 1440)
TWAP_MIN = (15, 30, 60, 120, 240, 480)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.005, 0.995)
    return np.log(p / (1 - p))


def path_features(paths: pd.DataFrame) -> pd.DataFrame:
    """paths: match_id, kickoff, t, p_a. Only prints strictly before each kickoff are used."""
    d = paths[["match_id", "kickoff", "t", "p_a"]].copy()
    d = d[d.t < d.kickoff].sort_values(["match_id", "t"], kind="stable")
    rows = []
    for mid, g in d.groupby("match_id", sort=True):
        k = g.kickoff.iloc[0]
        mins = ((k - g.t).dt.total_seconds() / 60.0).to_numpy()
        p = g.p_a.to_numpy(float)
        rec = {"match_id": mid, "kickoff": k, "n_prints": len(p), "p_last": p[-1], "last_age_min": mins[-1],
               "m0": p[-1] if mins[-1] <= KICKOFF_WINDOW_MIN else np.nan, "p_open": p[0], "open_age_h": mins[0] / 60.0}
        for lag in LAGS_MIN:
            i = np.searchsorted(-mins, -float(lag), side="right") - 1
            rec[f"p_{lag}m"] = p[i] if i >= 0 else np.nan
        changes = np.flatnonzero(np.diff(p) != 0)
        rec["n_changes"] = len(changes)
        rec["n_changes_6h"] = int((mins[changes + 1] <= 360).sum()) if len(changes) else 0
        rec["stale_min"] = mins[changes[-1] + 1] if len(changes) else mins[0]
        z = _logit(p)
        for w in TWAP_MIN:
            m = mins <= w
            rec[f"zbar_{w}m"] = z[m].mean() if m.any() else np.nan
            rec[f"zmed_{w}m"] = np.median(z[m]) if m.any() else np.nan
        rec["z_min"], rec["z_max"] = z.min(), z.max()
        dz = np.diff(z)
        rec["z_absmove"] = np.abs(dz).sum()
        rows.append(rec)
    return pd.DataFrame(rows)
