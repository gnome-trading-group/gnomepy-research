"""
Point-in-time series tables: HLTV bo3 series with their outcome and context, and per-series features of the Polymarket
price path up to kickoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

KICKOFF_WINDOW = pd.Timedelta(minutes=30)
HORIZONS_H = (1, 3, 6, 12, 24)
SMOOTH_MIN = (10, 20, 30, 45, 60, 90, 120, 180, 240, 360)
EPS = 1e-3


def naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def week_start(t: pd.Series) -> pd.Series:
    t = naive(t)
    return (t - pd.to_timedelta(t.dt.weekday, unit="D")).dt.normalize()


def hltv_series(data_dir: str | Path, before: pd.Timestamp) -> pd.DataFrame:
    """Every decided HLTV bo3 series known before `before`, with its outcome y (team A won)."""
    h = load(data_dir, "hltv/cs2_match_history", before)
    h = h[~showmatch_mask(h) & h.team_a_id.notna() & h.team_b_id.notna() & (h.bo_type == 3)]
    g = h.groupby("match_id")
    s = g.agg(match_time=("match_time", "first"), match_date=("match_date", "first"), team_a_id=("team_a_id", "first"),
              team_b_id=("team_b_id", "first"), event_name=("event_name", "first"), event_tier=("event_tier", "first"),
              is_lan=("is_lan", "first"), team_a_rank=("team_a_rank", "first"), team_b_rank=("team_b_rank", "first"),
              wins=("team_a_won", "sum"), maps=("team_a_won", "size"),
              team_a_lineup_ids=("team_a_lineup_ids", "first"), team_b_lineup_ids=("team_b_lineup_ids", "first"))
    s = s[s.wins * 2 != s.maps].reset_index()
    s["y"] = (s.wins * 2 > s.maps).astype(int)
    s["time"] = naive(s.match_time.fillna(s.match_date))
    return s


def market_series_targets(data_dir: str | Path, before: pd.Timestamp) -> pd.DataFrame:
    """HLTV bo3 series with a market path, shaped like the targets file (week_cutoff, announced fives, stand-ins)."""
    paths = pd.read_parquet(Path(data_dir) / "market/series_paths.parquet", columns=["match_id"]).match_id.unique()
    s = hltv_series(data_dir, before)
    s = s[s.match_id.isin(paths)].copy()
    s["week_cutoff"] = week_start(s.time)
    lu = load(data_dir, "hltv/cs2_match_lineups", before, ["match_id", "match_date", "team_id", "player_id", "is_standin"])
    st = lu[lu.is_standin.fillna(False).astype(bool)].groupby(["match_id", "team_id"]).player_id.apply(list).to_dict()
    for side in ("a", "b"):
        s[f"team_{side}_standin_ids"] = [st.get((m, t), []) for m, t in zip(s.match_id, s[f"team_{side}_id"])]
    return s.reset_index(drop=True)


def _last_before(t: np.ndarray, p: np.ndarray, at: np.datetime64) -> float:
    i = np.searchsorted(t, at, side="right") - 1
    return float(p[i]) if i >= 0 else np.nan


def _time_weighted(t: np.ndarray, lp: np.ndarray, k: np.datetime64, minutes: int) -> float:
    """Mean logit price over the last `minutes` before kickoff, each print weighted by how long it stood; the price
    before the window's first print stands from the window start."""
    start = k - np.timedelta64(minutes, "m")
    i = max(np.searchsorted(t, start, side="right") - 1, 0)
    tt = np.maximum(t[i:], start)
    dur = np.diff(np.append(tt, k)) / np.timedelta64(1, "s")
    if dur.sum() <= 0:
        return float(lp[-1])
    return float((lp[i:] * dur).sum() / dur.sum())


def path_features(paths: pd.DataFrame) -> pd.DataFrame:
    """One row per series from its prints before kickoff: the kickoff price m0 (last print within 30 minutes of
    kickoff, NaN without one), the last price whatever its age, earlier prices and how active the market was."""
    d = paths.assign(t=naive(paths.t), kickoff=naive(paths.kickoff))
    d = d[d.t < d.kickoff].sort_values(["match_id", "t"], kind="stable")
    rows = []
    for mid, g in d.groupby("match_id", sort=False):
        t = g.t.to_numpy()
        p = g.p_a.to_numpy(float)
        k = g.kickoff.iloc[0].to_datetime64()
        last_t = t[-1]
        age_min = (k - last_t) / np.timedelta64(1, "m")
        changes = np.flatnonzero(np.diff(p) != 0)
        last_change = t[changes[-1] + 1] if len(changes) else t[0]
        lp = logit(p)
        r = {"match_id": mid, "kickoff": k, "p_last": p[-1], "last_age_min": age_min,
             "m0": p[-1] if age_min <= 30 else np.nan, "n_prints": len(p), "n_changes": len(changes),
             "first_age_h": (k - t[0]) / np.timedelta64(1, "h"),
             "since_change_h": (k - last_change) / np.timedelta64(1, "h"),
             "path_range": float(lp.max() - lp.min()), "path_mean": float(lp.mean())}
        for h in HORIZONS_H:
            r[f"p_{h}h"] = _last_before(t, p, k - np.timedelta64(h, "h"))
        for w in SMOOTH_MIN:
            r[f"tw_{w}"] = _time_weighted(t, lp, k, w)
        n1 = np.searchsorted(t, k - np.timedelta64(6, "h"))
        r["changes_6h"] = int((np.diff(p[n1:]) != 0).sum()) if len(p) - n1 > 1 else 0
        rows.append(r)
    return pd.DataFrame(rows)
