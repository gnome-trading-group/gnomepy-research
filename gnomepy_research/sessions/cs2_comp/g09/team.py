"""
Team-level context a bookmaker adjusts a power rating for, from the maps before each request's snapshot time and the
rating's own pre-map residuals (result minus what the rating expected):

  form     recency-weighted residual over recent maps: is the team running above or below its rating
  mapadv   expected residual edge over the maps a greedy bo3 veto leaves, from each team's shrunk per-map residuals
  rest     log days since the team last played; load: maps in the last three days
  new      players of the announced five who did not play the team's last map (roster news)
  setting  shrunk residual over the past year on maps played in the same setting (LAN or online) as the series
Each is returned per side and as the A-minus-B difference.
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
    a = pd.DataFrame({"time": maps.time.to_numpy(), "team": maps.team_a_id.to_numpy(), "map": maps.map_name.to_numpy(),
                      "res": resid, "five": maps.team_a_player_ids.to_numpy(), "lan": maps.is_lan.to_numpy(float)})
    b = pd.DataFrame({"time": maps.time.to_numpy(), "team": maps.team_b_id.to_numpy(), "map": maps.map_name.to_numpy(),
                      "res": -resid, "five": maps.team_b_player_ids.to_numpy(), "lan": maps.is_lan.to_numpy(float)})
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
    lg = _long(maps, resid)
    times = lg.time.to_numpy()
    by_team = {t: g.index.to_numpy() for t, g in lg.groupby("team")}
    all_maps = maps[["time", "map_name"]]
    out = []
    for r in rows.itertuples(index=False):
        snap = np.datetime64(r.snap)
        recent = all_maps[(all_maps.time < snap) & (all_maps.time >= snap - np.timedelta64(int(POOL_WINDOW_DAYS), "D"))]
        pool = list(recent.map_name.value_counts().index[:7]) if len(recent) else []
        feats = {"key": r.key}
        side_maps = {}
        for side, team, five in (("a", r.team_a_id, r.lineup_a), ("b", r.team_b_id, r.lineup_b)):
            idx = by_team.get(team)
            if idx is None:
                feats.update({f"form_{side}": 0.0, f"nform_{side}": 0.0, f"rest_{side}": np.log1p(60.0),
                              f"load_{side}": 0.0, f"maps90_{side}": 0.0, f"setting_{side}": 0.0,
                              f"new_{side}": 5.0 if five else 0.0})
                side_maps[side] = {}
                continue
            idx = idx[times[idx] < snap]
            if len(idx) == 0:
                feats.update({f"form_{side}": 0.0, f"nform_{side}": 0.0, f"rest_{side}": np.log1p(60.0),
                              f"load_{side}": 0.0, f"maps90_{side}": 0.0, f"setting_{side}": 0.0,
                              f"new_{side}": 5.0 if five else 0.0})
                side_maps[side] = {}
                continue
            age = (snap - times[idx]) / np.timedelta64(1, "D")
            res = lg.res.to_numpy()[idx]
            ok = ~np.isnan(res)
            w = np.where(ok & (age <= FORM_WINDOW_DAYS), 0.5 ** (age / FORM_HALF_LIFE_DAYS), 0.0)
            feats[f"form_{side}"] = float((w * np.nan_to_num(res)).sum() / (w.sum() + 3.0))
            feats[f"nform_{side}"] = float(w.sum())
            feats[f"rest_{side}"] = float(np.log1p(min(age.min(), 60.0)))
            feats[f"load_{side}"] = float((age <= 3).sum())
            feats[f"maps90_{side}"] = float((age <= 90).sum())
            last_five = lg.five.iloc[idx[-1]]
            last = set() if last_five is None else {int(x) for x in last_five}
            feats[f"new_{side}"] = float(len([p for p in five if p not in last])) if five else 0.0
            sw = ok & (age <= 365) & (lg.lan.to_numpy()[idx] == float(r.is_lan))
            feats[f"setting_{side}"] = float(res[sw].sum() / (sw.sum() + 10.0))
            mw = ok & (age <= MAP_WINDOW_DAYS)
            mm = pd.DataFrame({"map": lg["map"].to_numpy()[idx][mw], "res": res[mw]}).groupby("map").res.agg(["sum", "size"])
            side_maps[side] = (mm["sum"] / (mm["size"] + MAP_SHRINK)).to_dict()
        diff = {m: side_maps["a"].get(m, 0.0) - side_maps["b"].get(m, 0.0) for m in pool}
        feats["mapadv"] = _veto(diff, pool)
        out.append(feats)
    f = pd.DataFrame(out)
    for c in ("form", "rest", "load", "new", "setting"):
        f[c] = f[f"{c}_a"] - f[f"{c}_b"]
    return f
