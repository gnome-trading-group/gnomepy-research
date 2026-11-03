"""
Per-series context a player-skill model cannot carry, each computed at the series' week cutoff from what was known:

- `resid`: how wrong the market has been about each team lately. Each team's past market-joined series contribute
  (won − market price); the shrunk running mean of A minus that of B.
- `h2h`: the head-to-head map record of the two teams, shrunk towards even.
- `pool`: a veto-aware map-pool advantage. Each team's strength on a map is its recency-weighted win rate there
  relative to its win rate overall, so the overall level stays with the skill prior. In a bo3 each side picks its
  best remaining map and the decider sits in the middle, so the advantage is the mean of the best, the worst and
  the median of A's per-map edges over the maps both teams have played.

One pass over HLTV maps and market-joined series in time order; every row only sees what was known before its
cutoff.
"""
from __future__ import annotations

import collections
import math

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g16.data import Inputs

CONTEXT_COLUMNS = ["resid", "h2h", "pool"]
RESID_SHRINK = 5.0
H2H_SHRINK = 4.0
MAP_SHRINK = 3.0
MAP_HALF_LIFE_DAYS = 180.0
MIN_COMMON_MAPS = 3


class _MapRecord:
    __slots__ = ("days", "won")

    def __init__(self):
        self.days: list = []
        self.won: list = []


def _weighted(rec: _MapRecord, cutoff: np.datetime64) -> tuple[float, float]:
    if not rec.days:
        return 0.0, 0.0
    age = (cutoff.astype("datetime64[D]") - np.array(rec.days, dtype="datetime64[D]")).astype(float)
    w = 0.5 ** (age / MAP_HALF_LIFE_DAYS)
    return float(w @ np.array(rec.won, float)), float(w.sum())


def context_rows(inputs: Inputs, to_price: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """to_price: match_id, week_cutoff, team_a, team_b. market: match_id, time, team_a, team_b, y, price (series the
    market priced, with their outcome). Returns match_id, week_cutoff and CONTEXT_COLUMNS."""
    maps = inputs.maps[inputs.maps.source == "hltv"]
    known = maps.known.to_numpy()
    mcols = {c: maps[c].to_numpy() for c in ("day", "team_a", "team_b", "won_a", "map_name")}
    market = market.sort_values(["time", "match_id"]).reset_index(drop=True)
    mtime = market.time.to_numpy()
    rcols = {c: market[c].to_numpy() for c in ("team_a", "team_b", "y", "price")}
    h2h: dict = collections.defaultdict(lambda: [0.0, 0.0])
    team_map: dict = collections.defaultdict(_MapRecord)
    team_all: dict = collections.defaultdict(_MapRecord)
    resid: dict = collections.defaultdict(lambda: [0.0, 0.0])
    done_maps = done_market = 0
    out = []
    for cutoff in sorted(pd.unique(to_price.week_cutoff)):
        cutoff = pd.Timestamp(cutoff)
        cut64 = np.datetime64(cutoff)
        stop = int(np.searchsorted(known, cut64, side="left"))
        for i in range(done_maps, stop):
            a, b, won = int(mcols["team_a"][i]), int(mcols["team_b"][i]), float(mcols["won_a"][i])
            key = (a, b) if a < b else (b, a)
            rec = h2h[key]
            rec[0] += won if a < b else 1.0 - won
            rec[1] += 1.0
            day = mcols["day"][i].astype("datetime64[D]") if isinstance(mcols["day"][i], np.datetime64) else np.datetime64(pd.Timestamp(mcols["day"][i]).date())
            name = mcols["map_name"][i]
            for team, w in ((a, won), (b, 1.0 - won)):
                for r in (team_map[(team, name)], team_all[team]):
                    r.days.append(day)
                    r.won.append(w)
        done_maps = max(done_maps, stop)
        stop = int(np.searchsorted(mtime, cut64, side="left"))
        for i in range(done_market, stop):
            e = float(rcols["y"][i]) - float(rcols["price"][i])
            for team, sign in ((int(rcols["team_a"][i]), 1.0), (int(rcols["team_b"][i]), -1.0)):
                r = resid[team]
                r[0] += sign * e
                r[1] += 1.0
        done_market = max(done_market, stop)
        week = to_price[to_price.week_cutoff == cutoff]
        for r in week.itertuples(index=False):
            a, b = int(r.team_a), int(r.team_b)
            ra, rb = resid.get(a, [0.0, 0.0]), resid.get(b, [0.0, 0.0])
            key = (a, b) if a < b else (b, a)
            hw, hn = h2h.get(key, [0.0, 0.0])
            wins_a = hw if a < b else hn - hw
            out.append({"match_id": int(r.match_id), "week_cutoff": cutoff,
                        "resid": ra[0] / (ra[1] + RESID_SHRINK) - rb[0] / (rb[1] + RESID_SHRINK),
                        "h2h": (2 * wins_a - hn) / (hn + H2H_SHRINK),
                        "pool": _pool_advantage(team_map, team_all, a, b, cut64)})
    return pd.DataFrame(out)


def _strength(team_map: dict, team_all: dict, team: int, name: str, cut64: np.datetime64) -> float | None:
    rec = team_map.get((team, name))
    if rec is None:
        return None
    s, n = _weighted(rec, cut64)
    if n < 1.0:
        return None
    sa, na = _weighted(team_all[team], cut64)
    return (s + 0.5 * MAP_SHRINK) / (n + MAP_SHRINK) - (sa + 0.5 * MAP_SHRINK) / (na + MAP_SHRINK)


def _pool_advantage(team_map: dict, team_all: dict, a: int, b: int, cut64: np.datetime64) -> float:
    names = {name for team, name in team_map if team == a} & {name for team, name in team_map if team == b}
    edges = []
    for name in sorted(names):
        sa, sb = _strength(team_map, team_all, a, name, cut64), _strength(team_map, team_all, b, name, cut64)
        if sa is not None and sb is not None:
            edges.append(sa - sb)
    if len(edges) < MIN_COMMON_MAPS:
        return 0.0
    edges.sort()
    return (edges[0] + edges[-1] + edges[len(edges) // 2]) / 3.0
