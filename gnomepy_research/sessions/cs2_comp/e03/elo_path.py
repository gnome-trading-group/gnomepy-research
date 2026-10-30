"""HLTV team Elo read in date order, with each week's targets rated at that week's Monday.

Series on a day are rated before any of that day's results are observed, and a target is rated from the ladder as it
stood at its week cutoff. Targets never write to the ladder, so a team first met in a target week gets the same seed
whether or not that target is present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.elo import EloConfig, EloEngine

def _ids(values) -> list[int]:
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return []
    return [int(p) for p in values]


def _peek_rating(engine: EloEngine, team_id: int, as_of: pd.Timestamp, lineup: list[int]) -> float:
    rating = engine._global.get(team_id)
    if rating is not None:
        return engine._decayed(rating, as_of)
    seed = engine._inherited_seed(team_id, lineup, as_of) if lineup else None
    return seed if seed is not None else engine._seed(team_id, as_of)


def _snapshot_series(engine: EloEngine, rows: pd.DataFrame, day: pd.Timestamp, last_lineup: dict) -> dict:
    first = rows.iloc[0]
    id_a, id_b = int(first["team_a_id"]), int(first["team_b_id"])
    lineup_a, lineup_b = last_lineup.get(id_a, []), last_lineup.get(id_b, [])
    elo_a, _ = engine.rating_global(id_a, day, lineup_a)
    elo_b, _ = engine.rating_global(id_b, day, lineup_b)
    return {
        "match_id": int(first["match_id"]),
        "match_date": day,
        "match_time": first["match_time"],
        "bo_type": int(first["bo_type"]),
        "team_a_id": id_a,
        "team_b_id": id_b,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "lineup_a": lineup_a,
        "lineup_b": lineup_b,
        "maps_a": int(rows["team_a_won"].sum()),
        "maps_b": int(len(rows) - rows["team_a_won"].sum()),
    }


def _snapshot_targets(engine: EloEngine, cut: pd.Timestamp, targets: pd.DataFrame, last_lineup: dict) -> pd.DataFrame:
    out = []
    for t in targets.itertuples(index=False):
        id_a, id_b = int(t.team_a_id), int(t.team_b_id)
        lineup_a, lineup_b = last_lineup.get(id_a, []), last_lineup.get(id_b, [])
        out.append({
            "match_id": int(t.match_id),
            "match_date": cut,
            "match_time": cut,
            "bo_type": 3,
            "team_a_id": id_a,
            "team_b_id": id_b,
            "elo_a": _peek_rating(engine, id_a, cut, lineup_a),
            "elo_b": _peek_rating(engine, id_b, cut, lineup_b),
            "lineup_a": lineup_a,
            "lineup_b": lineup_b,
            "maps_a": np.nan,
            "maps_b": np.nan,
        })
    return pd.DataFrame(out)


def _remember_lineups(first: pd.Series, last_lineup: dict) -> None:
    last_lineup[int(first["team_a_id"])] = _ids(first["team_a_player_ids"])
    last_lineup[int(first["team_b_id"])] = _ids(first["team_b_player_ids"])


def run(maps: pd.DataFrame, rankings: pd.DataFrame | None, target_cuts: dict[pd.Timestamp, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = EloEngine(EloConfig(), rankings)
    last_lineup: dict[int, list[int]] = {}
    series_rows: list[dict] = []
    target_frames: list[pd.DataFrame] = []
    cuts = sorted(target_cuts)
    next_cut = 0

    for day, day_maps in maps.groupby("match_date", sort=True):
        while next_cut < len(cuts) and cuts[next_cut] <= day:
            target_frames.append(_snapshot_targets(engine, cuts[next_cut], target_cuts[cuts[next_cut]], last_lineup))
            next_cut += 1
        groups = [g.sort_values("map_position_in_series") for _, g in day_maps.groupby("match_id", sort=True)]
        for g in groups:
            series_rows.append(_snapshot_series(engine, g, day, last_lineup))
        for g in groups:
            engine.observe_series(g)
            _remember_lineups(g.iloc[0], last_lineup)

    while next_cut < len(cuts):
        target_frames.append(_snapshot_targets(engine, cuts[next_cut], target_cuts[cuts[next_cut]], last_lineup))
        next_cut += 1

    return pd.DataFrame(series_rows), pd.concat(target_frames, ignore_index=True)
