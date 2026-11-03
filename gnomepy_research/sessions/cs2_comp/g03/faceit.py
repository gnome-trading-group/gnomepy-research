"""Rolling FACEIT per-player form and skill level, keyed by HLTV player id."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp import bundle

WINDOW_DAYS = 180
HALF_LIFE_DAYS = 90.0
PRIOR_MAPS = 10.0
PRIOR_SCORE = 0.5


def faceit_frames(data_dir, cutoff):
    m = bundle.load(data_dir, "faceit/faceit_matches", cutoff, ["match_id", "finished_at"])
    ex = bundle.load(data_dir, "faceit/faceit_exclusions", cutoff)
    whole = set(ex[ex.map_name.isna()].match_id)
    per_map = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
    s = bundle.load(data_dir, "faceit/faceit_player_map_stats", cutoff,
                    ["match_id", "faction", "hltv_id", "map_name", "kills", "deaths", "map_position_in_series"])
    s = s[~s.match_id.isin(whole)]
    s = s[[(a, b) not in per_map for a, b in zip(s.match_id, s.map_name)]]
    full = s.groupby(["match_id", "map_position_in_series", "faction"]).kills.transform("size") == 5
    s = s[full & s.hltv_id.notna() & (s.kills + s.deaths > 0)]
    s = s.merge(m, on="match_id")
    s["d"] = pd.to_datetime(s.finished_at, unit="s")
    s["score"] = s.kills / (s.kills + s.deaths)
    r = bundle.load(data_dir, "faceit/faceit_rosters", cutoff, ["match_id", "hltv_id", "skill_level", "played"])
    r = r[r.hltv_id.notna() & r.skill_level.notna()].merge(m, on="match_id")
    r["d"] = pd.to_datetime(r.finished_at, unit="s")
    return s[["hltv_id", "d", "score"]], r[["hltv_id", "d", "skill_level", "match_id"]]


def rolling(stats: pd.DataFrame, lvl: pd.DataFrame, weeks):
    lvl = lvl.sort_values(["d", "match_id"], kind="stable")
    out = {}
    for w in weeks:
        win = stats[(stats.d < w) & (stats.d >= w - pd.Timedelta(days=WINDOW_DAYS))]
        age = (w - win.d).dt.total_seconds().to_numpy() / 86400.0
        wt = 0.5 ** (age / HALF_LIFE_DAYS)
        g = pd.DataFrame({"p": win.hltv_id.to_numpy(), "w": wt, "ws": wt * win.score.to_numpy()}).groupby("p").sum()
        mean = (g.ws + PRIOR_MAPS * PRIOR_SCORE) / (g.w + PRIOR_MAPS)
        l = lvl[(lvl.d < w) & (lvl.d >= w - pd.Timedelta(days=WINDOW_DAYS))].groupby("hltv_id").skill_level.last()
        out[w] = (mean.to_dict(), g.w.to_dict(), l.to_dict())
    return out
