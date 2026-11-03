"""
A second skill estimate per series from the repo's player-skill Kalman filter (`models.cs2.hltv.player_skill`),
read at the series' weekly cutoff.

The filter walks HLTV maps in date order. At each Monday cutoff, every series of the coming week is priced with
its announced five from the state at that moment, before any map of that week is observed, so a Friday series
never sees Monday's results. The filter also says how unsure it is about the gap, which the layer can use to
weigh the skill price against the market.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.models.cs2.hltv.player_skill import PlayerSkillFilter, _bo3, map_win_prob
from gnomepy_research.sessions.cs2_comp.bundle import load

EKF_COLUMNS = ["z_ekf", "ekf_sd", "ekf_chem", "ekf_new"]


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _five(v) -> tuple[int, ...]:
    if v is None or isinstance(v, float):
        return ()
    return tuple(int(p) for p in v)


def ekf_rows(data_dir, frame: pd.DataFrame, until: pd.Timestamp) -> pd.DataFrame:
    """frame: match_id, team_a, team_b, week_cutoff, lineup_a, lineup_b, sub_a, sub_b. One row per frame row."""
    h = load(data_dir, "hltv/cs2_match_history", until)
    h = h[~showmatch_mask(h) & h.team_a_id.notna() & h.team_b_id.notna()].copy()
    h["known"] = np.maximum(_naive(h.match_time.fillna(h.match_date)), _naive(h.match_date))
    h["day"] = h.known.dt.normalize()
    h = h[(h.team_a_score.fillna(0) + h.team_b_score.fillna(0)) > 0].sort_values(["known", "match_id", "map_position_in_series"])
    rankings = load(data_dir, "hltv/cs2_team_rankings", until, ["date", "team_id", "points"])
    rankings = rankings.assign(date=_naive(rankings["date"]))
    f = PlayerSkillFilter(rankings=rankings)
    out = {}
    cutoffs = sorted(pd.Timestamp(c) for c in frame.week_cutoff.unique())
    by_cutoff = {c: g for c, g in frame.groupby("week_cutoff")}
    maps = h.itertuples(index=False)
    pending = next(maps, None)
    for c in cutoffs:
        while pending is not None and pending.known < c:
            r = pending
            f.observe_map(r.day, int(r.team_a_id), int(r.team_b_id), _five(r.team_a_player_ids), _five(r.team_b_player_ids),
                          int(r.team_a_score), int(r.team_b_score))
            pending = next(maps, None)
        for r in by_cutoff[c].itertuples(index=False):
            out[(r.match_id, c)] = _price(f, r, c)
    rows = pd.DataFrame([out[(m, pd.Timestamp(c))] for m, c in zip(frame.match_id, frame.week_cutoff)], columns=EKF_COLUMNS)
    rows.insert(0, "match_id", frame.match_id.to_numpy())
    return rows


def _price(f: PlayerSkillFilter, r, day: pd.Timestamp) -> list[float]:
    five_a, five_b = _five(r.lineup_a), _five(r.lineup_b)
    if len(five_a) != 5 or len(five_b) != 5 or len(set(five_a + five_b)) != 10:
        return [float("nan")] * len(EKF_COLUMNS)
    a = f.lineup(five_a, int(r.team_a), float(r.sub_a or 0), day)
    b = f.lineup(five_b, int(r.team_b), float(r.sub_b or 0), day)
    d, s2, _, _ = f.gap(a, b)
    p = _bo3(map_win_prob(d, s2))
    p = min(max(p, 1e-4), 1 - 1e-4)
    new = sum(f.games[q] < f.cfg.new_player_maps for q in five_a) + sum(f.games[q] < f.cfg.new_player_maps for q in five_b)
    return [math.log(p / (1 - p)), math.sqrt(max(s2, 0.0)), a.kappa - b.kappa, float(new)]
