"""Chronological map events from the competition bundle: HLTV maps and cleaned FACEIT organised maps.

Every event carries the time at which its result is known, so a filter pass that stops at a cutoff has seen exactly the
events finished before it. FACEIT players that have an HLTV id share it, the rest get a stable negative key derived from
the FACEIT id, so keys never depend on which rows are present.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask

logger = logging.getLogger(__name__)

HLTV, FACEIT = 0, 1
FIVE = 5

DIVISIONS = (
    ("premier", 3), ("challenger", 3), ("advanced", 3), ("cct", 3),
    ("main", 2), ("ecl", 2),
    ("intermediate", 1),
    ("open", 0), ("entry", 0),
    ("qualifier", 4),
)
OTHER_DIVISION = 5
N_DIVISIONS = 6


def division(competition_name: str) -> int:
    name = (competition_name or "").lower()
    for key, group in DIVISIONS:
        if key in name:
            return group
    return OTHER_DIVISION


def faceit_key(faceit_id: str) -> int:
    return -1 - (int(faceit_id.replace("-", ""), 16) % (1 << 60))


@dataclass(frozen=True)
class MapEvent:
    t: float
    day: float
    source: int
    match_id: str
    position: int
    team_a: int
    team_b: int
    players_a: tuple
    players_b: tuple
    rounds_a: int
    rounds_b: int
    division: int
    dev_a: tuple | None = None
    dev_b: tuple | None = None


def _seconds(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts)
    ts = ts.dt.tz_convert(None) if ts.dt.tz is not None else ts
    return (ts.astype("datetime64[ns]").astype("int64") // 10**9).to_numpy(np.float64)


def _fives(col: pd.Series) -> list:
    return [None if v is None or isinstance(v, float) else tuple(int(p) for p in v) for v in col]


PROXY_FIT_END = "2025-10-01"
PROXY_FIT_START = "2025-01-01"


def deviations(players: tuple, perf: dict | None) -> tuple | None:
    """Each player's performance minus his team's mean, or None when any of the five is missing."""
    if not perf:
        return None
    vals = [perf.get(p) for p in players]
    if any(v is None for v in vals):
        return None
    m = sum(vals) / FIVE
    return tuple(v - m for v in vals)


def hltv_ratings(data_dir: Path) -> dict:
    s = pd.read_parquet(Path(data_dir) / "hltv/cs2_player_map_stats.parquet",
                        columns=["mapstatsid", "player_id", "side", "rating"])
    s = s[(s.side == "all") & s.rating.notna() & s.mapstatsid.notna()]
    out: dict = {}
    for m, p, r in zip(s.mapstatsid.astype(np.int64), s.player_id.astype(np.int64), s.rating.astype(float)):
        out.setdefault(m, {})[p] = r
    return out


@dataclass(frozen=True)
class Proxy:
    """Linear map from per-round kills, deaths and ADR to the HLTV rating scale, fitted on HLTV maps before PROXY_FIT_END."""
    with_adr: tuple
    without_adr: tuple

    def __call__(self, kills: float, deaths: float, adr: float, rounds: int) -> float:
        kpr, dpr = kills / rounds, deaths / rounds
        if adr == adr:
            c = self.with_adr
            return c[0] + c[1] * kpr + c[2] * dpr + c[3] * adr / 100.0
        c = self.without_adr
        return c[0] + c[1] * kpr + c[2] * dpr


def fit_proxy(data_dir: Path, history: pd.DataFrame) -> Proxy:
    s = pd.read_parquet(Path(data_dir) / "hltv/cs2_player_map_stats.parquet",
                        columns=["mapstatsid", "match_date", "side", "kills", "deaths", "adr", "rating"])
    s = s[(s.side == "all") & (pd.to_datetime(s.match_date) >= PROXY_FIT_START) & (pd.to_datetime(s.match_date) < PROXY_FIT_END)]
    rounds = (history.team_a_score + history.team_b_score).groupby(history.mapstatsid).first()
    s = s.assign(rounds=s.mapstatsid.map(rounds)).dropna(subset=["rounds", "rating", "adr", "kills", "deaths"])
    s = s[s.rounds >= 16]
    y = s.rating.to_numpy(float)
    kpr, dpr, adr = (s.kills / s.rounds).to_numpy(float), (s.deaths / s.rounds).to_numpy(float), s.adr.to_numpy(float) / 100.0
    ones = np.ones(len(s))
    full = np.linalg.lstsq(np.column_stack([ones, kpr, dpr, adr]), y, rcond=None)[0]
    slim = np.linalg.lstsq(np.column_stack([ones, kpr, dpr]), y, rcond=None)[0]
    return Proxy(tuple(float(x) for x in full), tuple(float(x) for x in slim))


def hltv_events(history: pd.DataFrame, ratings: dict | None = None) -> list[MapEvent]:
    h = history[~showmatch_mask(history) & (history.team_a_score != history.team_b_score)].copy()
    h = h[h.team_a_score.notna() & h.team_b_score.notna()]
    when = h.match_time.fillna(h.match_date)
    t = _seconds(when)
    fa, fb = _fives(h.team_a_player_ids), _fives(h.team_b_player_ids)
    out = []
    for i, r in enumerate(h.itertuples(index=False)):
        pa, pb = fa[i], fb[i]
        if pa is None or pb is None or len(pa) != FIVE or len(pb) != FIVE:
            continue
        perf = ratings.get(int(r.mapstatsid)) if ratings is not None and r.mapstatsid == r.mapstatsid else None
        out.append(MapEvent(t[i], t[i] / 86400.0, HLTV, str(r.match_id), int(r.map_position_in_series or 0),
                            int(r.team_a_id), int(r.team_b_id), pa, pb, int(r.team_a_score), int(r.team_b_score),
                            OTHER_DIVISION, deviations(pa, perf), deviations(pb, perf)))
    return out


def faceit_events(data_dir: Path, proxy: Proxy | None = None) -> list[MapEvent]:
    data_dir = Path(data_dir)
    matches = pd.read_parquet(data_dir / "faceit/faceit_matches.parquet",
                              columns=["match_id", "finished_at", "competition_name"])
    matches = matches[matches.finished_at.notna()]
    finished = dict(zip(matches.match_id, matches.finished_at.astype(float)))
    div = dict(zip(matches.match_id, (division(n) for n in matches.competition_name)))
    exclusions = pd.read_parquet(data_dir / "faceit/faceit_exclusions.parquet")
    whole = set(exclusions[exclusions.map_name.isna()].match_id)
    per_map = set(zip(exclusions[exclusions.map_name.notna()].match_id, exclusions[exclusions.map_name.notna()].map_name))
    links = pd.read_parquet(data_dir / "faceit/faceit_hltv_series_link.parquet", columns=["faceit_match_id"])
    linked = set(links.faceit_match_id)
    ids = pd.read_parquet(data_dir / "faceit/hltv_faceit_map.parquet")
    to_hltv = dict(zip(ids.faceit_id, ids.hltv_id.astype(int)))

    cols = ["match_id", "map_name", "map_position_in_series", "team_a_score", "team_b_score", "team_a_faceit_ids",
            "team_b_faceit_ids", "team_a_id", "team_b_id"]
    hist = pd.read_parquet(data_dir / "faceit/faceit_match_history.parquet", columns=cols)
    hist = hist[~hist.match_id.isin(whole) & ~hist.match_id.isin(linked) & hist.match_id.isin(finished.keys())
                & (hist.team_a_score != hist.team_b_score)]
    perf_of: dict = {}
    if proxy is not None:
        st = pd.read_parquet(data_dir / "faceit/faceit_player_map_stats.parquet",
                             columns=["match_id", "map_name", "faceit_id", "kills", "deaths", "adr"])
        st = st[st.match_id.isin(set(hist.match_id))]
        for m, mp, f, k, d, a in zip(st.match_id, st.map_name, st.faceit_id, st.kills.astype(float), st.deaths.astype(float),
                                     st.adr.astype(float)):
            perf_of.setdefault((m, mp), {})[f] = (k, d, a)
    out = []
    for r in hist.itertuples(index=False):
        if (r.match_id, r.map_name) in per_map or r.team_a_faceit_ids is None or r.team_b_faceit_ids is None:
            continue
        if len(r.team_a_faceit_ids) != FIVE or len(r.team_b_faceit_ids) != FIVE:
            continue
        pa = tuple(to_hltv.get(f) or faceit_key(f) for f in r.team_a_faceit_ids)
        pb = tuple(to_hltv.get(f) or faceit_key(f) for f in r.team_b_faceit_ids)
        t = finished[r.match_id]
        rounds = int(r.team_a_score) + int(r.team_b_score)
        dev_a = dev_b = None
        raw = perf_of.get((r.match_id, r.map_name))
        if raw is not None and rounds > 0:
            scaled = {f: proxy(k, d, a, rounds) for f, (k, d, a) in raw.items()}
            dev_a = deviations(r.team_a_faceit_ids, scaled)
            dev_b = deviations(r.team_b_faceit_ids, scaled)
        ta = int(r.team_a_id) if r.team_a_id == r.team_a_id else -1
        tb = int(r.team_b_id) if r.team_b_id == r.team_b_id else -1
        out.append(MapEvent(t, t / 86400.0, FACEIT, r.match_id, int(r.map_position_in_series or 0), ta, tb, pa, pb,
                            int(r.team_a_score), int(r.team_b_score), div[r.match_id], dev_a, dev_b))
    return out


def build_events(data_dir: Path, use_faceit: bool, use_stats: bool = True) -> list[MapEvent]:
    history = pd.read_parquet(Path(data_dir) / "hltv/cs2_match_history.parquet")
    events = hltv_events(history, hltv_ratings(data_dir) if use_stats else None)
    if use_faceit:
        events += faceit_events(data_dir, fit_proxy(data_dir, history) if use_stats else None)
    events.sort(key=lambda e: (e.t, e.match_id, e.position))
    logger.info("events: %d (%d HLTV)", len(events), sum(e.source == HLTV for e in events))
    return events
