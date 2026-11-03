"""
Map-pool structure of a bo3, which neither e12 nor a series price sees.

Per team and map over the trailing year: wins in excess of the team's overall win rate, shrunk toward zero. A bo3
plays each side's own pick and a decider, so the series-level edge takes the map difference d = excess_A - excess_B
over the active pool at its maximum (A's pick), its minimum (B's pick) and its median (the decider).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

MAP_COLUMNS = ["mp_edge", "mp_pick", "mp_opp", "mp_decider", "mp_depth", "mp_known", "mp_h2h", "mp_h2h_n"]
WINDOW = pd.Timedelta(days=365)
SHRINK = 5.0
POOL_SIZE = 7


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _long(h: pd.DataFrame) -> pd.DataFrame:
    a = pd.DataFrame({"team": h.team_a_id.to_numpy(), "map": h.map_name.to_numpy(), "won": h.team_a_won.to_numpy(float), "known": h.known.to_numpy()})
    b = pd.DataFrame({"team": h.team_b_id.to_numpy(), "map": h.map_name.to_numpy(), "won": 1 - h.team_a_won.to_numpy(float), "known": h.known.to_numpy()})
    return pd.concat([a, b], ignore_index=True)


def map_rows(data_dir, frame: pd.DataFrame, until: pd.Timestamp) -> pd.DataFrame:
    """frame: match_id, team_a, team_b, week_cutoff. One row per frame row, in order."""
    h = load(data_dir, "hltv/cs2_match_history", until)
    h = h[~showmatch_mask(h) & h.team_a_id.notna() & h.team_b_id.notna() & h.map_name.notna()
          & (h.team_a_score != h.team_b_score)].copy()
    h["known"] = np.maximum(_naive(h.match_time.fillna(h.match_date)), _naive(h.match_date))
    long = _long(h)
    series = h.groupby("match_id").agg(team_a=("team_a_id", "first"), team_b=("team_b_id", "first"), known=("known", "max"),
                                       wins=("team_a_won", "sum"), maps=("team_a_won", "size"))
    series = series[series.wins * 2 != series.maps]
    series["a_won"] = (series.wins * 2 > series.maps).astype(float)
    out = np.full((len(frame), len(MAP_COLUMNS)), np.nan)
    for c, idx in frame.groupby("week_cutoff").indices.items():
        c = pd.Timestamp(c)
        w = long[(long.known < c) & (long.known >= c - WINDOW)]
        pool = w["map"].value_counts().index[:POOL_SIZE].tolist()
        w = w[w["map"].isin(pool)]
        team = w.groupby("team").won.agg(["sum", "size"])
        rate = (team["sum"] / team["size"]).to_dict()
        per = w.groupby(["team", "map"]).won.agg(["sum", "size"])
        excess = {}
        for (t, m), r in per.iterrows():
            excess[(t, m)] = (r["sum"] - r["size"] * rate[t]) / (r["size"] + SHRINK)
        played = per["size"].groupby(level=0).size().to_dict()
        sw = series[(series.known < c) & (series.known >= c - WINDOW)]
        h2h = {}
        for t_a, t_b, won in zip(sw.team_a.astype(int), sw.team_b.astype(int), sw.a_won):
            k = (min(t_a, t_b), max(t_a, t_b))
            sign = 1.0 if t_a < t_b else -1.0
            n, s_ = h2h.get(k, (0, 0.0))
            h2h[k] = (n + 1, s_ + sign * (2 * won - 1))
        for i in idx:
            r = frame.iloc[i]
            a, b = int(r.team_a), int(r.team_b)
            n_h2h, s_h2h = h2h.get((min(a, b), max(a, b)), (0, 0.0))
            s_h2h = s_h2h if a < b else -s_h2h
            d = np.array([excess.get((a, m), 0.0) - excess.get((b, m), 0.0) for m in pool])
            d.sort()
            out[i] = [d[-1] + d[0] + d[len(d) // 2], d[-1], d[0], d[len(d) // 2],
                      float(played.get(a, 0) - played.get(b, 0)), float(min(team["size"].get(a, 0), team["size"].get(b, 0))),
                      s_h2h / (n_h2h + 2.0), float(n_h2h)]
    rows = pd.DataFrame(out, columns=MAP_COLUMNS)
    rows.insert(0, "match_id", frame.match_id.to_numpy())
    return rows
