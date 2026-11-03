"""
Point-in-time inputs for g12, read once through `bundle.load` up to the latest cutoff of a run.

Every row carries `known`, the time it became public, so the replay in `raters.py` can apply it only to cutoffs after
that time. Reading once at the latest cutoff and filtering by `known` gives exactly the rows a bundle cut at an earlier
cutoff would hold, because `known` is at least as late as the bundle's own cut column for every table.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths

FIVE = 5
KICKOFF_WINDOW = pd.Timedelta(minutes=30)


def naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def monday(ts: pd.Series) -> pd.Series:
    d = naive(ts).dt.floor("D")
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def five_tuple(v) -> tuple[int, ...] | None:
    if v is None:
        return None
    ids = tuple(sorted({int(x) for x in v}))
    return ids if len(ids) == FIVE else None


def path_features(paths: pd.DataFrame) -> pd.DataFrame:
    """Per series: the kickoff price m0 (last print within 30 minutes before kickoff) and the shape of its path."""
    p = paths.copy()
    p["t"] = naive(p.t)
    p["kickoff"] = naive(p.kickoff)
    p = p[p.t < p.kickoff].sort_values(["match_id", "t"], kind="stable")
    if p.empty:
        return pd.DataFrame(columns=["match_id", "kickoff", "m0", "m_24h", "m_6h", "m_first", "age_h", "changes", "last_change_h"])
    rows = []
    for mid, g in p.groupby("match_id", sort=True):
        ko = g.kickoff.iloc[0]
        t = g.t.to_numpy()
        pa = g.p_a.to_numpy(float)
        last_t = t[-1]
        m0 = pa[-1] if (ko - last_t) <= KICKOFF_WINDOW else np.nan

        def at(hours: float) -> float:
            i = np.searchsorted(t, np.datetime64(ko - pd.Timedelta(hours=hours)), side="right") - 1
            return pa[i] if i >= 0 else np.nan

        changed = np.flatnonzero(np.diff(pa) != 0)
        last_change = t[changed[-1] + 1] if len(changed) else t[0]
        rows.append({"match_id": int(mid), "kickoff": ko, "m0": m0, "m_24h": at(24), "m_6h": at(6), "m_first": pa[0],
                     "age_h": (ko - t[0]).total_seconds() / 3600.0, "changes": len(changed),
                     "last_change_h": (ko - last_change).total_seconds() / 3600.0})
    return pd.DataFrame(rows)


@dataclass
class Inputs:
    maps: pd.DataFrame          # one decisive HLTV map per row, time-sorted
    series: pd.DataFrame        # one HLTV series per row with outcome, announced fives, week_cutoff
    rankings: pd.DataFrame      # date, team_id, points, rank
    market: pd.DataFrame        # per past series: kickoff, m0 and path shape (prints known before `until`)
    faceit_maps: pd.DataFrame | None
    faceit_levels: pd.DataFrame | None


def _ratings(data_dir: Path, until: pd.Timestamp) -> dict:
    st = load(data_dir, "hltv/cs2_player_map_stats", until, ["match_id", "match_date", "map_name", "player_id", "side", "rating"])
    st = st[(st.side == "all") & st.rating.notna() & st.player_id.notna()]
    return dict(zip(zip(st.match_id.astype(np.int64), st.map_name, st.player_id.astype(np.int64)), st.rating.astype(float)))


def load_hltv(data_dir: Path, until: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = load(data_dir, "hltv/cs2_match_history", until)
    h = h[~showmatch_mask(h) & h.team_a_id.notna() & h.team_b_id.notna()].copy()
    h["time"] = naive(h.match_time.fillna(h.match_date))
    h["known"] = np.maximum(h.time, naive(h.match_date))
    h = h.sort_values(["time", "match_id", "map_position_in_series"], kind="stable")

    g = h.groupby("match_id", sort=False)
    s = g.agg(time=("time", "min"), known=("known", "max"), team_a=("team_a_id", "first"), team_b=("team_b_id", "first"),
              bo_type=("bo_type", "first"), is_lan=("is_lan", "first"), tier=("event_tier", "first"),
              wins=("team_a_won", "sum"), nmaps=("team_a_won", "size"),
              lineup_a=("team_a_lineup_ids", "first"), lineup_b=("team_b_lineup_ids", "first"),
              played_a=("team_a_player_ids", "first"), played_b=("team_b_player_ids", "first")).reset_index()
    s["y"] = np.where(s.wins * 2 > s.nmaps, 1.0, np.where(s.wins * 2 < s.nmaps, 0.0, np.nan))
    s["week_cutoff"] = monday(s.time)
    s["five_a"] = [five_tuple(l) or five_tuple(p) for l, p in zip(s.lineup_a, s.played_a)]
    s["five_b"] = [five_tuple(l) or five_tuple(p) for l, p in zip(s.lineup_b, s.played_b)]
    lu = load(data_dir, "hltv/cs2_match_lineups", until, ["match_id", "match_date", "team_id", "is_standin"])
    sub = lu.groupby(["match_id", "team_id"]).is_standin.sum()
    subd = dict(zip(zip(sub.index.get_level_values(0).astype(np.int64), sub.index.get_level_values(1).astype(np.int64)), sub.to_numpy()))
    s["sub_a"] = [int(subd.get((m, int(t)), 0)) for m, t in zip(s.match_id, s.team_a)]
    s["sub_b"] = [int(subd.get((m, int(t)), 0)) for m, t in zip(s.match_id, s.team_b)]
    s = s.drop(columns=["lineup_a", "lineup_b", "played_a", "played_b", "wins"])
    s = s.astype({"team_a": np.int64, "team_b": np.int64, "match_id": np.int64})

    m = h[(h.team_a_score != h.team_b_score)].copy()
    rating = _ratings(data_dir, until)
    fa = [five_tuple(v) for v in m.team_a_player_ids]
    fb = [five_tuple(v) for v in m.team_b_player_ids]
    ok = np.array([a is not None and b is not None for a, b in zip(fa, fb)])
    m = m[ok]
    fa = [x for x, k in zip(fa, ok) if k]
    fb = [x for x, k in zip(fb, ok) if k]
    sa, sb = m.team_a_score.to_numpy(int), m.team_b_score.to_numpy(int)
    ot = sa + sb > 24
    mid = m.match_id.astype(np.int64).to_numpy()
    maps = pd.DataFrame({
        "match_id": mid, "time": m.time.to_numpy(), "known": m.known.to_numpy(),
        "team_a": m.team_a_id.astype(np.int64).to_numpy(), "team_b": m.team_b_id.astype(np.int64).to_numpy(),
        "won_a": (sa > sb).astype(float), "rounds_a": np.where(ot, 12, sa), "rounds": np.where(ot, 24, sa + sb),
        "is_lan": m.is_lan.fillna(0).to_numpy(int), "bo_type": m.bo_type.to_numpy(int),
        "five_a": fa, "five_b": fb,
        "rating_a": [[rating.get((x, n, p), np.nan) for p in f] for x, n, f in zip(mid, m.map_name, fa)],
        "rating_b": [[rating.get((x, n, p), np.nan) for p in f] for x, n, f in zip(mid, m.map_name, fb)],
    })
    maps = maps.sort_values(["known", "time", "match_id"], kind="stable")
    return maps.reset_index(drop=True), s


def load_faceit(data_dir: Path, until: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FACEIT organised 5v5 maps on HLTV player ids where linked (others keep a negative id), minus series HLTV
    already has; and each player's FACEIT skill level by the time it was known."""
    link = load(data_dir, "faceit/hltv_faceit_map")
    f2h = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))
    fm = load(data_dir, "faceit/faceit_matches", until, ["match_id", "finished_at", "has_stats"])
    fm["known"] = pd.to_datetime(fm.finished_at, unit="s")
    ex = load(data_dir, "faceit/faceit_exclusions", until)
    whole = set(ex.match_id[ex.map_name.isna()])
    bad = set(zip(ex.match_id[ex.map_name.notna()], ex.map_name[ex.map_name.notna()]))
    dup = set(load(data_dir, "faceit/faceit_hltv_series_link", until).faceit_match_id)
    cols = ["match_id", "map_name", "match_time", "match_date", "team_a_score", "team_b_score", "team_a_ot_score",
            "team_b_ot_score", "team_a_faceit_ids", "team_b_faceit_ids", "team_a_players", "team_b_players"]
    fh = load(data_dir, "faceit/faceit_match_history", until, cols)
    fh = fh.merge(fm[fm.has_stats][["match_id", "known"]], on="match_id")
    fh = fh[~fh.match_id.isin(whole | dup)]
    fh = fh[[(a, b) not in bad for a, b in zip(fh.match_id, fh.map_name)]]
    fh = fh[(fh.team_a_players == FIVE) & (fh.team_b_players == FIVE) & (fh.team_a_score != fh.team_b_score)]

    def uid(f: str) -> int:
        h = f2h.get(f)
        return int(h) if h is not None else -int(hashlib.blake2b(f.encode(), digest_size=7).hexdigest(), 16)

    fh = fh.assign(time=naive(fh.match_time.fillna(fh.match_date))).sort_values(["known", "match_id", "map_name"], kind="stable")
    fa = [tuple(sorted({uid(x) for x in v})) if v is not None else () for v in fh.team_a_faceit_ids]
    fb = [tuple(sorted({uid(x) for x in v})) if v is not None else () for v in fh.team_b_faceit_ids]
    ok = np.array([len(a) == FIVE and len(b) == FIVE for a, b in zip(fa, fb)])
    fh = fh[ok]
    sa, sb = fh.team_a_score.to_numpy(int), fh.team_b_score.to_numpy(int)
    ot = ((fh.team_a_ot_score.fillna(0) + fh.team_b_ot_score.fillna(0)) > 0).to_numpy() | (sa + sb > 24)
    st = load(data_dir, "faceit/faceit_player_map_stats", until, ["match_id", "map_name", "faceit_id", "adr", "kills", "deaths"])
    st = st.dropna()
    rounds_of = dict(zip(zip(fh.match_id, fh.map_name), sa + sb))
    n = np.array([rounds_of.get(k, np.nan) for k in zip(st.match_id, st.map_name)], float)
    # HLTV's rating weights on the per-round stats FACEIT reports; only deviations within a five are used, so the
    # terms FACEIT lacks (KAST, impact) only cost precision.
    proxy = 0.0032 * st.adr.to_numpy(float) + (0.36 * st.kills.to_numpy(float) - 0.53 * st.deaths.to_numpy(float)) / n
    ok_st = np.isfinite(proxy)
    perf = dict(zip(zip(st.match_id[ok_st], st.map_name[ok_st], [uid(f) for f in st.faceit_id[ok_st]]), proxy[ok_st]))
    fa_ok = [x for x, k in zip(fa, ok) if k]
    fb_ok = [x for x, k in zip(fb, ok) if k]
    maps = pd.DataFrame({"match_id": fh.match_id.to_numpy(), "time": fh.time.to_numpy(), "known": fh.known.to_numpy(),
                         "won_a": (sa > sb).astype(float), "rounds_a": np.where(ot, 12, sa), "rounds": np.where(ot, 24, sa + sb),
                         "five_a": fa_ok, "five_b": fb_ok,
                         "rating_a": [[perf.get((m, n, p), np.nan) for p in f] for m, n, f in zip(fh.match_id, fh.map_name, fa_ok)],
                         "rating_b": [[perf.get((m, n, p), np.nan) for p in f] for m, n, f in zip(fh.match_id, fh.map_name, fb_ok)]})
    ro = load(data_dir, "faceit/faceit_rosters", until, ["match_id", "faceit_id", "played", "skill_level"])
    ro = ro[ro.played & ro.skill_level.notna() & ro.faceit_id.isin(f2h.keys())]
    ro = ro.merge(fm[["match_id", "known"]], on="match_id")
    levels = pd.DataFrame({"player": ro.faceit_id.map(f2h).astype(np.int64).to_numpy(), "known": ro.known.to_numpy(),
                           "level": ro.skill_level.to_numpy(float)}).sort_values(["known", "player"], kind="stable")
    return maps.reset_index(drop=True), levels.reset_index(drop=True)


def load_inputs(data_dir: str | Path, until: pd.Timestamp, faceit: bool = True) -> Inputs:
    data_dir = Path(data_dir)
    until = pd.Timestamp(until)
    maps, series = load_hltv(data_dir, until)
    rk = load(data_dir, "hltv/cs2_team_rankings", until, ["date", "team_id", "rank", "points"])
    rk = rk.assign(date=naive(rk["date"])).sort_values(["date", "team_id"], kind="stable").reset_index(drop=True)
    sp = load(data_dir, "market/series_paths", until)
    sp = sp[naive(sp.kickoff) < until]
    market = path_features(sp).merge(series[["match_id", "team_a", "team_b"]], on="match_id")
    market = market.sort_values(["kickoff", "match_id"], kind="stable").reset_index(drop=True)
    fmaps, levels = load_faceit(data_dir, until) if faceit else (None, None)
    return Inputs(maps, series, rk, market, fmaps, levels)


def load_target_market(data_dir: str | Path, match_ids) -> pd.DataFrame:
    return path_features(target_paths(data_dir, match_ids))
