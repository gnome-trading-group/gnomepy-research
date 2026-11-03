"""
Point-in-time inputs for g15: HLTV maps in time order, HLTV bo3 series with their announced fives and outcome, weekly
ranking points, and the Polymarket kickoff price of every past series that had a market.

Everything is read once through `bundle.load` at the run's latest cutoff; a weekly snapshot then keeps the rows whose
own time is before its cutoff, by the same rule `bundle.load` applies, so a snapshot never depends on which other
weeks are in the run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths

FIVE = 5
KICKOFF_WINDOW = pd.Timedelta(minutes=30)
PATH_HOURS = 48.0


def naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def week_cutoff(t: pd.Series) -> pd.Series:
    t = naive(t)
    return (t - pd.to_timedelta(t.dt.weekday, unit="D")).dt.normalize()


def _five(v) -> tuple[int, ...] | None:
    if v is None or isinstance(v, float):
        return None
    ids = tuple(int(x) for x in v)
    if len(ids) != FIVE or len(set(ids)) != FIVE:
        return None
    return ids


def load_maps(data_dir: str | Path, until: pd.Timestamp) -> pd.DataFrame:
    """Decided HLTV maps before `until`, time-ordered: who played, regulation-ish rounds, and context."""
    h = load(data_dir, "hltv/cs2_match_history", until)
    h = h[~showmatch_mask(h)].copy()
    h["time"] = naive(h.match_time.fillna(h.match_date))
    h = h[(h.team_a_score != h.team_b_score) & (np.maximum(h.team_a_score, h.team_b_score) >= 13)]
    h["five_a"] = [_five(v) for v in h.team_a_player_ids]
    h["five_b"] = [_five(v) for v in h.team_b_player_ids]
    h = h[h.five_a.notna() & h.five_b.notna()]
    ot = (h.team_a_score + h.team_b_score) > 24
    a_won = h.team_a_score > h.team_b_score
    # Overtime says only who edged it: score it as the narrowest regulation result.
    h["rounds_a"] = np.where(ot, np.where(a_won, 13, 12), h.team_a_score).astype(int)
    h["rounds_b"] = np.where(ot, np.where(a_won, 12, 13), h.team_b_score).astype(int)
    cols = ["match_id", "time", "match_date", "team_a_id", "team_b_id", "five_a", "five_b", "rounds_a", "rounds_b",
            "map_name", "is_lan", "event_tier", "bo_type", "team_a_rank", "team_b_rank", "map_position_in_series", "mapstatsid"]
    h = h[cols].rename(columns={"team_a_id": "team_a", "team_b_id": "team_b", "team_a_rank": "rank_a", "team_b_rank": "rank_b"})
    h["match_date"] = naive(h.match_date)
    return h.sort_values(["time", "match_id", "map_position_in_series"], kind="stable").reset_index(drop=True)


def attach_ratings(data_dir: str | Path, maps: pd.DataFrame, until: pd.Timestamp) -> pd.DataFrame:
    """Each player's HLTV map rating in the order of the five, as rating_a0..4 / rating_b0..4 (NaN where unrated)."""
    st = load(data_dir, "hltv/cs2_player_map_stats", until, ["match_date", "mapstatsid", "player_id", "side", "rating"])
    st = st[(st.side == "all") & st.rating.notna()]
    key = dict(zip(zip(st.mapstatsid, st.player_id.astype(int)), st.rating.astype(float)))
    out = maps.copy()
    for side in ("a", "b"):
        fives = out[f"five_{side}"].tolist()
        ids = out.mapstatsid.tolist()
        for j in range(FIVE):
            out[f"rating_{side}{j}"] = [key.get((m, f[j]), np.nan) for m, f in zip(ids, fives)]
    return out


def load_series(data_dir: str | Path, until: pd.Timestamp) -> pd.DataFrame:
    """HLTV bo3 series with an outcome before `until`, with the fives announced for them and their stand-ins."""
    h = load(data_dir, "hltv/cs2_match_history", until)
    h = h[~showmatch_mask(h)].copy()
    h["time"] = naive(h.match_time.fillna(h.match_date))
    g = h.groupby("match_id")
    s = g.agg(time=("time", "min"), match_date=("match_date", "first"), team_a=("team_a_id", "first"),
              team_b=("team_b_id", "first"), bo_type=("bo_type", "first"), is_lan=("is_lan", "first"),
              event_tier=("event_tier", "first"), wins=("team_a_won", "sum"), maps=("team_a_won", "size"),
              lineup_a=("team_a_lineup_ids", "first"), lineup_b=("team_b_lineup_ids", "first"),
              rank_a=("team_a_rank", "first"), rank_b=("team_b_rank", "first")).reset_index()
    s = s[(s.bo_type == 3) & (s.wins * 2 != s.maps)].copy()
    s["y"] = (s.wins * 2 > s.maps).astype(int)
    s["lineup_a"] = [_five(v) for v in s.lineup_a]
    s["lineup_b"] = [_five(v) for v in s.lineup_b]
    s["match_date"] = naive(s.match_date)
    lu = load(data_dir, "hltv/cs2_match_lineups", until, ["match_id", "match_date", "team_id", "player_id", "is_standin"])
    st = lu[lu.is_standin == 1].groupby(["match_id", "team_id"]).player_id.apply(lambda x: tuple(int(p) for p in x))
    s["standins_a"] = [st.get((m, t), ()) for m, t in zip(s.match_id, s.team_a)]
    s["standins_b"] = [st.get((m, t), ()) for m, t in zip(s.match_id, s.team_b)]
    s["week_cutoff"] = week_cutoff(s.time)
    return s.sort_values(["time", "match_id"]).reset_index(drop=True)


def load_rankings(data_dir: str | Path, until: pd.Timestamp) -> pd.DataFrame:
    r = load(data_dir, "hltv/cs2_team_rankings", until, ["date", "team_id", "rank", "points"])
    r["date"] = naive(r.date)
    return r.sort_values(["date", "rank"]).reset_index(drop=True)


def kickoff_price(paths: pd.DataFrame) -> pd.DataFrame:
    """Per series: the last print within 30 minutes of kickoff, plus a few shape features of the 48 hours before."""
    p = paths.copy()
    p["t"] = naive(p.t)
    p["kickoff"] = naive(p.kickoff)
    p = p.sort_values(["match_id", "t"], kind="stable")
    p["before"] = (p.kickoff - p.t)
    g = p.groupby("match_id")
    out = g.agg(kickoff=("kickoff", "first"), n_prints=("t", "size"), p_first=("p_a", "first"), p_last=("p_a", "last"),
                t_last=("t", "last"), t_first=("t", "first"), p_min=("p_a", "min"), p_max=("p_a", "max"))
    near = p[p.before <= KICKOFF_WINDOW].groupby("match_id").p_a.last()
    out["m0"] = near
    out["stale_min"] = (out.kickoff - out.t_last).dt.total_seconds() / 60.0
    out["span_hours"] = (out.kickoff - out.t_first).dt.total_seconds() / 3600.0
    day = p[p.before <= pd.Timedelta(hours=24)].groupby("match_id").p_a.first()
    out["p_24h"] = day
    return out.reset_index()


def load_market(data_dir: str | Path, until: pd.Timestamp) -> pd.DataFrame:
    paths = load(data_dir, "market/series_paths", until)
    if paths.empty:
        return pd.DataFrame(columns=["match_id", "m0"])
    return kickoff_price(paths)


def load_target_market(data_dir: str | Path, match_ids) -> pd.DataFrame:
    paths = target_paths(data_dir, match_ids)
    if paths.empty:
        return pd.DataFrame(columns=["match_id", "m0"])
    return kickoff_price(paths)
