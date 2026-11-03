"""
One feature row per series to price, built by a single pass over the maps in time order with a snapshot at every
distinct week cutoff among the jobs.

A job is a series with its cutoff, team ids and announced fives (with stand-ins). At a cutoff every engine has seen
exactly the maps whose time is before it, so a week's rows are the same whether the run holds one week or a hundred.
"""
from __future__ import annotations

import collections
import math
import time as clock
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g15.data import FIVE, load_maps, load_rankings
from gnomepy_research.sessions.cs2_comp.g15.ratings import LOG_UNRANKED, PlayerKalman, PlayerMargin, Rankings, TeamMargin, UNRANKED

RECENT_DAYS = 90
NEW_PLAYER_MAPS = 10


class TeamContext:
    """The five a team last fielded, when, and how many maps it played in the last 90 days."""

    def __init__(self):
        self.last_five: dict[int, tuple[int, ...]] = {}
        self.last_time: dict[int, pd.Timestamp] = {}
        self.times: dict[int, collections.deque] = collections.defaultdict(collections.deque)
        self.maps: collections.Counter = collections.Counter()

    def observe(self, t: pd.Timestamp, team: int, five: tuple[int, ...]) -> None:
        self.last_five[team] = five
        self.last_time[team] = t
        self.times[team].append(t)
        self.maps[team] += 1

    def recent(self, team: int, cutoff: pd.Timestamp) -> int:
        q = self.times.get(team)
        if not q:
            return 0
        start = cutoff - pd.Timedelta(days=RECENT_DAYS)
        while q and q[0] < start:
            q.popleft()
        return len(q)

    def idle_days(self, team: int, cutoff: pd.Timestamp) -> float:
        t = self.last_time.get(team)
        return 365.0 if t is None else min(365.0, (cutoff - t).total_seconds() / 86400.0)

    def new_players(self, team: int, five: tuple[int, ...]) -> int:
        last = self.last_five.get(team)
        return FIVE if last is None else FIVE - len(set(five) & set(last))


def _lineup(v) -> tuple[int, ...] | None:
    if v is None or isinstance(v, float):
        return None
    ids = tuple(int(x) for x in v)
    return ids if len(ids) == FIVE and len(set(ids)) == FIVE else None


def _count(v) -> int:
    return 0 if v is None or isinstance(v, float) else len(v)


def build_rows(data_dir: str | Path, jobs: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """jobs: match_id, week_cutoff, team_a, team_b, lineup_a, lineup_b, standins_a, standins_b, is_lan, event_tier.
    A job without an announced five for a side uses the five that side fielded last."""
    jobs = jobs.copy()
    jobs["week_cutoff"] = pd.to_datetime(jobs.week_cutoff)
    until = jobs.week_cutoff.max()
    maps = load_maps(data_dir, until)
    rankings = load_rankings(data_dir, until)
    rk = Rankings(rankings)
    team_eng = TeamMargin(rk)
    player_eng = PlayerMargin(rk)
    kalman = PlayerKalman(rankings)
    ctx = TeamContext()
    mt = maps.time.to_numpy()
    recs = maps.to_dict("records")
    i = 0
    out = []
    for cutoff in sorted(jobs.week_cutoff.unique()):
        cutoff = pd.Timestamp(cutoff)
        t0 = clock.time()
        stop = int(np.searchsorted(mt, np.datetime64(cutoff), side="left"))
        while i < stop:
            r = recs[i]
            day = r["match_date"]
            ta, tb, fa, fb, ra, rb = int(r["team_a"]), int(r["team_b"]), r["five_a"], r["five_b"], int(r["rounds_a"]), int(r["rounds_b"])
            team_eng.observe(day, ta, tb, ra, rb)
            player_eng.observe(day, ta, tb, fa, fb, ra, rb)
            kalman.observe(day, ta, tb, fa, fb, ra, rb)
            ctx.observe(r["time"], ta, fa)
            ctx.observe(r["time"], tb, fb)
            i += 1
        week = jobs[jobs.week_cutoff == cutoff]
        for j in week.itertuples(index=False):
            ta, tb = int(j.team_a), int(j.team_b)
            fa, fb = _lineup(j.lineup_a), _lineup(j.lineup_b)
            announced_a, announced_b = fa is not None, fb is not None
            fa = fa or ctx.last_five.get(ta)
            fb = fb or ctx.last_five.get(tb)
            row = {"match_id": j.match_id, "week_cutoff": cutoff, "team_a": ta, "team_b": tb,
                   "announced_a": announced_a, "announced_b": announced_b,
                   "has_five_a": fa is not None, "has_five_b": fb is not None,
                   "is_lan": int(j.is_lan), "event_tier": int(j.event_tier),
                   "standins_a": _count(j.standins_a), "standins_b": _count(j.standins_b),
                   "rank_a": rk.rank_of(ta, cutoff), "rank_b": rk.rank_of(tb, cutoff),
                   "points_a": rk.points_of(ta, cutoff), "points_b": rk.points_of(tb, cutoff),
                   "maps90_a": ctx.recent(ta, cutoff), "maps90_b": ctx.recent(tb, cutoff),
                   "maps_a": ctx.maps[ta], "maps_b": ctx.maps[tb],
                   "idle_a": ctx.idle_days(ta, cutoff), "idle_b": ctx.idle_days(tb, cutoff),
                   "tm_gap": team_eng.gap(ta, tb, cutoff), "tm_n_a": team_eng.n[ta], "tm_n_b": team_eng.n[tb]}
            if fa is not None and fb is not None:
                mus_a, unk_a = player_eng.five(fa, ta, cutoff)
                mus_b, unk_b = player_eng.five(fb, tb, cutoff)
                k = kalman.price(fa, fb, ta, tb, float(row["standins_a"]), float(row["standins_b"]), cutoff)
                row.update({"pm_gap": float(np.mean(mus_a) - np.mean(mus_b)), "pm_unknown_a": unk_a, "pm_unknown_b": unk_b,
                            "pm_min_maps_a": min(player_eng.n[p] for p in fa), "pm_min_maps_b": min(player_eng.n[p] for p in fb),
                            "pm_new_a": sum(player_eng.n[p] < NEW_PLAYER_MAPS for p in fa),
                            "pm_new_b": sum(player_eng.n[p] < NEW_PLAYER_MAPS for p in fb),
                            "pm_sd_a": float(np.std(mus_a)), "pm_sd_b": float(np.std(mus_b)),
                            "pk_gap": k["gap"], "pk_sd": k["sd"], "pk_chem_a": k["chem_a"], "pk_chem_b": k["chem_b"],
                            "pk_sd_a": k["sd_a"], "pk_sd_b": k["sd_b"], "pk_best_a": k["best_a"], "pk_best_b": k["best_b"],
                            "pk_worst_a": k["worst_a"], "pk_worst_b": k["worst_b"],
                            "new_a": ctx.new_players(ta, fa), "new_b": ctx.new_players(tb, fb)})
            else:
                row.update({"pm_gap": np.nan, "pk_gap": np.nan, "pk_sd": np.nan})
            for extra in ("y", "m0", "match_time"):
                if hasattr(j, extra):
                    row[extra] = getattr(j, extra)
            out.append(row)
        if verbose:
            print(f"{cutoff:%Y-%m-%d} maps seen {i} series {len(week)} {clock.time() - t0:.1f}s", flush=True)
    df = pd.DataFrame(out)
    df["log_rank_gap"] = (LOG_UNRANKED - np.log(df.rank_a.clip(1, UNRANKED))) - (LOG_UNRANKED - np.log(df.rank_b.clip(1, UNRANKED)))
    df["log_points_gap"] = np.log1p(df.points_a) - np.log1p(df.points_b)
    return df


def history_jobs(series: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Past bo3 series as jobs, with their outcome, for a walk-forward between two cutoffs (inclusive)."""
    s = series[(series.week_cutoff >= first) & (series.week_cutoff <= last)]
    return s[["match_id", "week_cutoff", "team_a", "team_b", "lineup_a", "lineup_b", "standins_a", "standins_b",
              "is_lan", "event_tier", "y", "time"]].rename(columns={"time": "match_time"}).reset_index(drop=True)


def target_jobs(targets: pd.DataFrame) -> pd.DataFrame:
    t = targets.copy()
    wc = pd.to_datetime(t.week_cutoff)
    t["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    mt = pd.to_datetime(t.match_time)
    t["match_time"] = mt.dt.tz_convert(None) if mt.dt.tz is not None else mt
    return pd.DataFrame({"match_id": t.match_id, "week_cutoff": t.week_cutoff, "team_a": t.team_a_id, "team_b": t.team_b_id,
                         "lineup_a": t.team_a_lineup_ids, "lineup_b": t.team_b_lineup_ids,
                         "standins_a": t.team_a_standin_ids if "team_a_standin_ids" in t else None,
                         "standins_b": t.team_b_standin_ids if "team_b_standin_ids" in t else None,
                         "is_lan": t.is_lan, "event_tier": t.event_tier, "match_time": t.match_time})
