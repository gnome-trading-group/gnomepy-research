"""
One time-ordered pass over the maps that prices each week's targets from the state at that week's cutoff.

The pass is causal: the state at a cutoff has seen exactly the maps dated before it, so a bundle cut at that cutoff
reproduces the same prices.

With FACEIT on, two filters run side by side: the joint one sees HLTV and FACEIT maps, the other only HLTV maps. A
series where a team has no FACEIT history has nothing for the joint filter to add, and the two populations' skills sit
on slightly different scales there, so it takes the HLTV-only price.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e08.data import FACEIT, HLTV, Maps, RankPoints
from gnomepy_research.sessions.cs2_comp.e08.filter import Params, SkillFilter
from gnomepy_research.sessions.cs2_comp.e08.series import (RECENT_MAPS, MapTable, Moves, SeriesParams, calibrate,
                                                             infer_lineup, price_series)


def naive_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def walk(maps: Maps, targets: pd.DataFrame, moves: Moves, ranks: RankPoints, params: Params, sp: SeriesParams,
         track_from: int | None = None) -> tuple[pd.DataFrame, SkillFilter]:
    f = SkillFilter(params)
    g = SkillFilter(params) if params.use_faceit else None
    if track_from is not None:
        f.track_from = track_from
    table = MapTable(sp.sigma_m)
    recent: dict[int, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=RECENT_MAPS))
    faceit_seen: set[int] = set()
    times = maps.time.astype("datetime64[ns]").astype(np.int64)
    ts = times.tolist()
    src, comp = maps.src.tolist(), maps.comp.tolist()
    pa, pb = maps.players_a.tolist(), maps.players_b.tolist()
    fa, fb = maps.feat_a, maps.feat_b
    ra, n = maps.rounds_a.tolist(), maps.rounds.tolist()
    da, db = maps.dev_a, maps.dev_b
    ta, tb = maps.team_a.tolist(), maps.team_b.tolist()
    days = maps.time.astype("datetime64[D]")

    rows = []
    i = 0
    targets = targets.assign(match_time=naive_utc(targets.match_time), week_cutoff=pd.to_datetime(targets.week_cutoff))
    for cutoff, week in targets.groupby("week_cutoff", sort=True):
        stop = int(np.searchsorted(times, cutoff.to_datetime64().astype("datetime64[ns]").astype(np.int64), side="left"))
        while i < stop:
            if src[i] == FACEIT:
                if params.use_faceit:
                    f.update(ts[i], FACEIT, comp[i], pa[i], pb[i], fa[i], fb[i], ra[i], n[i], da[i], db[i])
                    faceit_seen.update(pa[i])
                    faceit_seen.update(pb[i])
            else:
                f.update(ts[i], HLTV, comp[i], pa[i], pb[i], fa[i], fb[i], ra[i], n[i], da[i], db[i])
                if g is not None:
                    g.update(ts[i], HLTV, comp[i], pa[i], pb[i], fa[i], fb[i], ra[i], n[i], da[i], db[i])
                day = pd.Timestamp(days[i])
                recent[ta[i]].append((day, pa[i]))
                recent[tb[i]].append((day, pb[i]))
            i += 1
        for r in week.itertuples():
            team_a, team_b = int(r.team_a_id), int(r.team_b_id)
            lineups = [infer_lineup(t, recent, moves, cutoff) for t in (team_a, team_b)]
            fe = [float(ranks.lookup(np.array([t]), np.array([cutoff.to_datetime64()]))[0]) for t in (team_a, team_b)]
            t = int(r.match_time.to_datetime64().astype("datetime64[ns]").astype(np.int64))
            args = (table, sp, t, lineups[0][0], lineups[0][1], fe[0], lineups[1][0], lineups[1][1], fe[1])
            q_joint, mean, var = price_series(f, *args)
            q_hltv = price_series(g, *args)[0] if g is not None else q_joint
            n_face = [sum(p in faceit_seen for p in lineups[k][0]) for k in (0, 1)]
            rows.append({
                "match_id": r.match_id, "q_joint": q_joint, "q_hltv": q_hltv, "gap": mean, "gap_var": var,
                "fit_cutoff": cutoff, "n_face_a": n_face[0], "n_face_b": n_face[1],
                "guess": sum(1 - p for p in lineups[0][1]) + sum(1 - p for p in lineups[1][1]),
            })
    out = pd.DataFrame(rows)
    covered = np.minimum(out.n_face_a, out.n_face_b) >= params.hybrid_min
    out["q"] = [calibrate(j if c else h, sp.slope) for j, h, c in zip(out.q_joint, out.q_hltv, covered)]
    return out, f
