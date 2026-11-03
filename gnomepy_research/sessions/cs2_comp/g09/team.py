"""
Team-level adjustments a bookmaker makes to a power rating, from the maps before each request's snapshot time and the
rating's own pre-map residuals (result minus what the rating expected):

  form     recency-weighted residual over recent maps: is the team running above or below its rating
  mapadv   expected residual edge over the maps a greedy bo3 veto leaves, from each team's shrunk per-map residuals
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORM_HALF_LIFE_DAYS = 21.0
FORM_WINDOW_DAYS = 90.0
MAP_WINDOW_DAYS = 180.0
MAP_SHRINK = 6.0
POOL_WINDOW_DAYS = 60.0


def _long(maps: pd.DataFrame, resid: np.ndarray) -> pd.DataFrame:
    a = pd.DataFrame({"time": maps.time.to_numpy(), "team": maps.team_a_id.to_numpy(), "map": maps.map_name.to_numpy(), "res": resid})
    b = pd.DataFrame({"time": maps.time.to_numpy(), "team": maps.team_b_id.to_numpy(), "map": maps.map_name.to_numpy(), "res": -resid})
    return pd.concat([a, b], ignore_index=True).sort_values("time", kind="stable").reset_index(drop=True)


def _veto(diff: dict[str, float], pool: list[str]) -> float:
    """Greedy bo3 veto on A's per-map edge: A bans its worst, B bans A's best, A picks its best, B picks A's worst,
    two more bans, and the decider is weighted by the chance a bo3 reaches it."""
    left = sorted(pool, key=lambda m: diff.get(m, 0.0))
    if len(left) < 3:
        return 0.0
    left.pop(0)
    left.pop(-1)
    pick_a = left.pop(-1)
    pick_b = left.pop(0)
    while len(left) > 1:
        left.pop(0)
        if len(left) > 1:
            left.pop(-1)
    return (diff.get(pick_a, 0.0) + diff.get(pick_b, 0.0) + 0.5 * diff.get(left[0], 0.0)) / 2.5


def team_features(maps: pd.DataFrame, resid: np.ndarray, rows: pd.DataFrame) -> pd.DataFrame:
    """rows: key, snap, team_a_id, team_b_id. Returns key, form_a, form_b, form (A minus B) and mapadv."""
    lg = _long(maps, resid)
    times = lg.time.to_numpy()
    by_team = {t: g.index.to_numpy() for t, g in lg.groupby("team")}
    res_all = lg.res.to_numpy()
    map_names, map_code = np.unique(lg["map"].fillna("").to_numpy(), return_inverse=True)
    mtimes, mnames = maps.time.to_numpy(), maps.map_name.fillna("").to_numpy()
    pools: dict = {}
    out = []
    for r in rows.itertuples(index=False):
        snap = np.datetime64(r.snap)
        if snap not in pools:
            recent = mnames[(mtimes < snap) & (mtimes >= snap - np.timedelta64(int(POOL_WINDOW_DAYS), "D"))]
            names, counts = np.unique(recent, return_counts=True)
            pools[snap] = [str(m) for m in names[np.argsort(-counts, kind="stable")][:7]]
        pool = pools[snap]
        feats = {"key": r.key}
        side_maps = {}
        for side, team in (("a", r.team_a_id), ("b", r.team_b_id)):
            idx = by_team.get(team, np.array([], int))
            idx = idx[times[idx] < snap]
            if len(idx) == 0:
                feats[f"form_{side}"] = 0.0
                side_maps[side] = {}
                continue
            age = (snap - times[idx]) / np.timedelta64(1, "D")
            res = res_all[idx]
            ok = ~np.isnan(res)
            w = np.where(ok & (age <= FORM_WINDOW_DAYS), 0.5 ** (age / FORM_HALF_LIFE_DAYS), 0.0)
            feats[f"form_{side}"] = float((w * np.nan_to_num(res)).sum() / (w.sum() + 3.0))
            mw = ok & (age <= MAP_WINDOW_DAYS)
            codes = map_code[idx][mw]
            tot = np.bincount(codes, weights=res[mw], minlength=len(map_names))
            cnt = np.bincount(codes, minlength=len(map_names))
            side_maps[side] = {str(map_names[c]): tot[c] / (cnt[c] + MAP_SHRINK) for c in np.flatnonzero(cnt)}
        diff = {m: side_maps["a"].get(m, 0.0) - side_maps["b"].get(m, 0.0) for m in pool}
        feats["mapadv"] = _veto(diff, pool)
        out.append(feats)
    f = pd.DataFrame(out)
    f["form"] = f.form_a - f.form_b
    return f
