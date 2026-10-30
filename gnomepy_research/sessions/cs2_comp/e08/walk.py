"""
One time-ordered pass over the maps that prices each week's targets from the state at that week's cutoff.

The pass is causal: the state at a cutoff has seen exactly the maps dated before it, so a bundle cut at that cutoff
reproduces the same prices.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e08.data import FACEIT, HLTV, FaceitFives, Maps, RankPoints
from gnomepy_research.sessions.cs2_comp.e08.filter import Params, SkillFilter, combine
from gnomepy_research.sessions.cs2_comp.e08.series import (RECENT_MAPS, MapTable, Moves, SeriesParams, calibrate,
                                                             infer_lineup, price_series)


def naive_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def prequential(f: SkillFilter, ff: SkillFilter, t: int, pa, pb, fa, fb, ra: int, n: int) -> None:
    """Round log loss of an HLTV map under the FACEIT-linked state, recorded beside the filter's own."""
    mu_a, _ = combine(f, ff, pa, fa, t)
    mu_b, _ = combine(f, ff, pb, fb, t)
    d = float(f.weights(mu_a) @ mu_a - f.weights(mu_b) @ mu_b)
    pi = min(max(1 / (1 + np.exp(-d)), 1e-9), 1 - 1e-9)
    f.linked_nll += -(ra * np.log(pi) + (n - ra) * np.log(1 - pi)) / n


def walk(maps: Maps, targets: pd.DataFrame, moves: Moves, ranks: RankPoints, params: Params, sp: SeriesParams,
         fives: FaceitFives | None = None, final_slope: float | None = None, track_from: int | None = None) -> tuple[pd.DataFrame, SkillFilter, SkillFilter | None]:
    f = SkillFilter(params)
    ff = (f if params.joint else SkillFilter(params, FACEIT)) if params.use_faceit else None
    linked = ff if not params.joint else None
    if track_from is not None:
        f.track_from = track_from
        if ff is not None:
            ff.track_from = track_from
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
            if src[i] == FACEIT and not params.use_faceit:
                i += 1
                continue
            if src[i] == HLTV:
                if linked is not None and ts[i] >= f.track_from:
                    prequential(f, linked, ts[i], pa[i], pb[i], fa[i], fb[i], ra[i], n[i])
                f.update(ts[i], HLTV, comp[i], pa[i], pb[i], fa[i], fb[i], ra[i], n[i], da[i], db[i])
                day = pd.Timestamp(days[i])
                recent[ta[i]].append((day, pa[i]))
                recent[tb[i]].append((day, pb[i]))
            else:
                ff.update(ts[i], FACEIT, comp[i], pa[i], pb[i], fa[i], fb[i], ra[i], n[i], da[i], db[i])
                faceit_seen.update(pa[i])
                faceit_seen.update(pb[i])
            i += 1
        feats = {}
        for r in week.itertuples():
            team_a, team_b = int(r.team_a_id), int(r.team_b_id)
            lineups = [infer_lineup(t, recent, moves, cutoff, fives if params.use_faceit and params.fives else None) for t in (team_a, team_b)]
            fe = [float(ranks.lookup(np.array([t]), np.array([cutoff.to_datetime64()]))[0]) for t in (team_a, team_b)]
            t = int(r.match_time.to_datetime64().astype("datetime64[ns]").astype(np.int64))
            q, mean, var = price_series(f, linked, table, sp, t, lineups[0][0], lineups[0][1], fe[0], lineups[1][0], lineups[1][1], fe[1])
            rows.append({
                "match_id": r.match_id, "q_raw": q, "gap": mean, "gap_var": var, "fit_cutoff": cutoff,
                "n_face_a": sum(p in faceit_seen for p in lineups[0][0]), "n_face_b": sum(p in faceit_seen for p in lineups[1][0]),
                "guess": sum(1 - p for p in lineups[0][1]) + sum(1 - p for p in lineups[1][1]),
            })
    out = pd.DataFrame(rows)
    out["q"] = [calibrate(q, final_slope if final_slope is not None else sp.slope) for q in out.q_raw]
    return out, f, ff
