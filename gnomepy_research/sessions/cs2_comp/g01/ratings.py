"""HLTV map and series tables prepared for the skill fits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask


def prepare_maps(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist[~showmatch_mask(hist)].copy()
    h["t"] = pd.to_datetime(h.match_time.fillna(h.match_date), utc=True).dt.tz_convert(None)
    h = h.dropna(subset=["team_a_won", "team_a_score", "team_b_score"])
    h = h[h.team_a_player_ids.map(len).eq(5) & h.team_b_player_ids.map(len).eq(5)]
    return h.sort_values(["t", "match_id", "map_position_in_series"], kind="stable").reset_index(drop=True)


def series_table(maps: pd.DataFrame) -> pd.DataFrame:
    g = maps.groupby("match_id", sort=False)
    s = g.agg(t=("t", "min"), team_a_id=("team_a_id", "first"), team_b_id=("team_b_id", "first"),
              wins_a=("team_a_won", "sum"), n=("team_a_won", "size"), bo=("bo_type", "first"),
              lan=("is_lan", "first"), tier=("event_tier", "first"),
              pa=("team_a_player_ids", "first"), pb=("team_b_player_ids", "first")).reset_index()
    s["y"] = (s.wins_a * 2 > s.n).astype(float)
    s.loc[s.wins_a * 2 == s.n, "y"] = np.nan
    return s


def monday(ts: pd.Series) -> pd.Series:
    d = ts.dt.normalize()
    return d - pd.to_timedelta(d.dt.weekday, unit="D")
