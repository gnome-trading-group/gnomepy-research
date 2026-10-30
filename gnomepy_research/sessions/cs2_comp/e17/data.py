"""The map stream the filter consumes: HLTV maps and cleaned FACEIT maps, one chronological frame.

FACEIT cleaning follows the brief: no-stats series, the exclusion list, series linked to an HLTV series and maps that
are not 5v5 are dropped. Players are linked through hltv_faceit_map; an unlinked FACEIT player keeps an id derived
from its FACEIT id (negative, so it never collides with an HLTV id).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

HLTV, FACEIT = 0, 1
STREAM_COLUMNS = ["time", "source", "match_id", "map_pos", "team_a", "team_b", "players_a", "players_b", "rounds_a", "rounds_b",
                  "hltv_time_of_link"]


def synthetic_id(faceit_id: str) -> int:
    return -(int(faceit_id.replace("-", ""), 16) % (1 << 61)) - 1


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def hltv_maps(data_dir, before: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", before,
             ["match_id", "match_date", "match_time", "team_a_id", "team_b_id", "team_a_player_ids", "team_b_player_ids",
              "team_a_score", "team_b_score", "map_position_in_series", "event_name", "format", "bo_type", "team_a_won", "is_lan"])
    h = h[~showmatch_mask(h) & (h.team_a_score + h.team_b_score > 0)].copy()
    h["time"] = _naive(h.match_time.fillna(h.match_date))
    h["source"] = HLTV
    h["map_pos"] = h.map_position_in_series.astype(int)
    h["team_a"] = h.team_a_id.astype(int)
    h["team_b"] = h.team_b_id.astype(int)
    h["players_a"] = [tuple(int(p) for p in x) for x in h.team_a_player_ids]
    h["players_b"] = [tuple(int(p) for p in x) for x in h.team_b_player_ids]
    h["rounds_a"] = h.team_a_score.astype(int)
    h["rounds_b"] = h.team_b_score.astype(int)
    h["hltv_time_of_link"] = pd.NaT
    return h


def hltv_series(maps: pd.DataFrame) -> pd.DataFrame:
    """One row per HLTV series with its outcome, for calibration and dev scoring."""
    g = maps.sort_values(["time", "match_id", "map_pos"]).groupby("match_id", sort=False)
    s = g.agg(time=("time", "first"), team_a=("team_a", "first"), team_b=("team_b", "first"), bo_type=("bo_type", "first"),
              is_lan=("is_lan", "first"), won=("team_a_won", "mean"), maps=("map_pos", "size")).reset_index()
    s["y"] = (s.won > 0.5).astype(int)
    return s


def faceit_maps(data_dir, before: pd.Timestamp) -> pd.DataFrame:
    fm = load(data_dir, "faceit/faceit_matches", before,
              ["match_id", "finished_at", "has_stats", "faction1_id", "faction2_id", "faction1_hltv_team", "faction2_hltv_team"])
    fm = fm[fm.has_stats]
    fh = load(data_dir, "faceit/faceit_match_history", before,
              ["match_id", "match_date", "map_name", "map_position_in_series", "team_a_faceit_id", "team_b_faceit_id", "team_a_faceit_ids",
               "team_b_faceit_ids", "team_a_players", "team_b_players", "team_a_score", "team_b_score"])
    fh = fh.merge(fm, on="match_id")
    ex = load(data_dir, "faceit/faceit_exclusions", before, ["match_id", "map_name"])
    whole = set(ex[ex.map_name.isna()].match_id)
    partial = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
    keep = ~fh.match_id.isin(whole) & pd.Series([(m, n) not in partial for m, n in zip(fh.match_id, fh.map_name)], index=fh.index)
    keep &= (fh.team_a_players == 5) & (fh.team_b_players == 5)
    keep &= fh.team_a_faceit_ids.map(lambda x: x is not None and len(x) == 5) & fh.team_b_faceit_ids.map(lambda x: x is not None and len(x) == 5)
    keep &= fh.team_a_score + fh.team_b_score > 0
    fh = fh[keep].copy()
    link = load(data_dir, "faceit/faceit_hltv_series_link", before, ["faceit_match_id", "hltv_match_id"])
    hl = load(data_dir, "hltv/cs2_match_history", before, ["match_id", "match_date", "match_time"]).drop_duplicates("match_id")
    hl["t"] = _naive(hl.match_time.fillna(hl.match_date))
    link_time = link.merge(hl, left_on="hltv_match_id", right_on="match_id").set_index("faceit_match_id").t
    fh["hltv_time_of_link"] = fh.match_id.map(link_time)
    ids = load(data_dir, "faceit/hltv_faceit_map", None, ["hltv_id", "faceit_id"])
    to_hltv = dict(zip(ids.faceit_id, ids.hltv_id.astype(int)))

    def players(fids) -> tuple[int, ...]:
        return tuple(to_hltv.get(f, synthetic_id(f)) for f in fids)

    fh["time"] = pd.to_datetime(fh.finished_at, unit="s")
    fh["source"] = FACEIT
    fh["map_pos"] = fh.map_position_in_series.astype(int)
    fh["team_a"] = [int(t) if pd.notna(t) else None for t in fh.faction1_hltv_team]
    fh["team_b"] = [int(t) if pd.notna(t) else None for t in fh.faction2_hltv_team]
    fh["players_a"] = [players(x) for x in fh.team_a_faceit_ids]
    fh["players_b"] = [players(x) for x in fh.team_b_faceit_ids]
    fh["rounds_a"] = fh.team_a_score.astype(int)
    fh["rounds_b"] = fh.team_b_score.astype(int)
    return fh


def map_stream(data_dir, before: pd.Timestamp, use_faceit: bool) -> pd.DataFrame:
    parts = [hltv_maps(data_dir, before)[STREAM_COLUMNS]]
    if use_faceit:
        parts.append(faceit_maps(data_dir, before)[STREAM_COLUMNS])
    s = pd.concat(parts, ignore_index=True)
    s["hltv_time_of_link"] = pd.to_datetime(s.hltv_time_of_link)
    s = s.sort_values(["time", "source", "match_id", "map_pos"], kind="mergesort").reset_index(drop=True)
    # A column of ints and None would become float with NaN; the filters test teams against None.
    for col in ("team_a", "team_b"):
        s[col] = pd.Series([int(t) if pd.notna(t) else None for t in s[col]], index=s.index, dtype=object)
    return s


def rankings(data_dir, before: pd.Timestamp) -> pd.DataFrame:
    return load(data_dir, "hltv/cs2_team_rankings", before, ["date", "team_id", "points"])


def monday(t: pd.Series | pd.Timestamp):
    t = pd.to_datetime(t)
    if isinstance(t, pd.Timestamp):
        return (t - pd.Timedelta(days=t.weekday())).normalize()
    return (t - pd.to_timedelta(t.dt.weekday, unit="D")).dt.normalize()


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))
