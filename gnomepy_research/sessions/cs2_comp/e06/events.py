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


def _seconds(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts)
    ts = ts.dt.tz_convert(None) if ts.dt.tz is not None else ts
    return (ts.astype("datetime64[ns]").astype("int64") // 10**9).to_numpy(np.float64)


def _fives(col: pd.Series) -> list:
    return [None if v is None or isinstance(v, float) else tuple(int(p) for p in v) for v in col]


def hltv_events(history: pd.DataFrame) -> list[MapEvent]:
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
        out.append(MapEvent(t[i], t[i] / 86400.0, HLTV, str(r.match_id), int(r.map_position_in_series or 0),
                            int(r.team_a_id), int(r.team_b_id), pa, pb, int(r.team_a_score), int(r.team_b_score),
                            OTHER_DIVISION))
    return out


def faceit_events(data_dir: Path) -> list[MapEvent]:
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
    out = []
    for r in hist.itertuples(index=False):
        if (r.match_id, r.map_name) in per_map or r.team_a_faceit_ids is None or r.team_b_faceit_ids is None:
            continue
        if len(r.team_a_faceit_ids) != FIVE or len(r.team_b_faceit_ids) != FIVE:
            continue
        pa = tuple(to_hltv.get(f) or faceit_key(f) for f in r.team_a_faceit_ids)
        pb = tuple(to_hltv.get(f) or faceit_key(f) for f in r.team_b_faceit_ids)
        t = finished[r.match_id]
        out.append(MapEvent(t, t / 86400.0, FACEIT, r.match_id, int(r.map_position_in_series or 0), -1, -1, pa, pb,
                            int(r.team_a_score), int(r.team_b_score), div[r.match_id]))
    return out


def build_events(data_dir: Path, use_faceit: bool) -> list[MapEvent]:
    history = pd.read_parquet(Path(data_dir) / "hltv/cs2_match_history.parquet")
    events = hltv_events(history)
    if use_faceit:
        events += faceit_events(data_dir)
    events.sort(key=lambda e: (e.t, e.match_id, e.position))
    logger.info("events: %d (%d HLTV)", len(events), sum(e.source == HLTV for e in events))
    return events
