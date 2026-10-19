import dataclasses

import pandas as pd
import pytest

from gnomepy_research.models.cs2.hltv.elo import EloConfig, EloEngine, compute_elo_features

RANKINGS = pd.DataFrame({
    "date": pd.to_datetime(["2026-01-01"] * 3),
    "team_id": [1, 2, 3],
    "points": [1000.0, 100.0, 10.0],
})
LINEUPS = {1: [11, 12, 13, 14, 15], 2: [21, 22, 23, 24, 25], 3: [31, 32, 33, 34, 35]}


def _series(match_id, day, team_a, team_b, a_won=True, lineup_a=None, lineup_b=None):
    return pd.DataFrame([{
        "match_id": match_id,
        "match_date": pd.Timestamp(day),
        "map_name": "de_mirage",
        "map_position_in_series": 1,
        "team_a_id": team_a,
        "team_b_id": team_b,
        "team_a_won": int(a_won),
        "team_a_score": 13 if a_won else 5,
        "team_b_score": 5 if a_won else 13,
        "team_a_lineup_ids": lineup_a if lineup_a is not None else LINEUPS.get(team_a, []),
        "team_b_lineup_ids": lineup_b if lineup_b is not None else LINEUPS.get(team_b, []),
    }])


def _seed(engine, team_id, day="2026-01-05"):
    return engine._seed(team_id, pd.Timestamp(day))


def test_unranked_team_seeds_below_every_ranked_team():
    engine = EloEngine(EloConfig(), RANKINGS)
    assert _seed(engine, 99) < _seed(engine, 3) < _seed(engine, 2) < _seed(engine, 1)


def test_unranked_seed_falls_back_to_base_before_any_ranking():
    engine = EloEngine(EloConfig(), RANKINGS)
    assert _seed(engine, 99, "2025-12-01") == EloConfig().base


def test_idle_team_keeps_its_rating():
    engine = EloEngine(EloConfig(), RANKINGS)
    engine.observe_series(_series(1, "2026-01-05", 1, 2))
    before, _ = engine.rating_global(1, pd.Timestamp("2026-01-06"))
    after, _ = engine.rating_global(1, pd.Timestamp("2027-01-06"))
    assert after == before


def _engine_after_team_1_plays():
    engine = EloEngine(EloConfig(), RANKINGS)
    engine.snapshot_series(_series(1, "2026-01-05", 1, 2))
    engine.observe_series(_series(1, "2026-01-05", 1, 2))
    return engine


def test_new_team_inherits_from_its_players_previous_team():
    engine = _engine_after_team_1_plays()
    day = pd.Timestamp("2026-01-06")
    team_1, _ = engine.rating_global(1, day)
    lineup = [11, 12, 13, 14, 99]
    rating, games = engine.rating_global(50, day, lineup)
    w = 4 / 5
    assert rating == pytest.approx(w * team_1 + (1 - w) * engine._seed(50, day))
    assert games == 0


def test_new_team_with_too_few_known_players_takes_the_ranking_seed():
    engine = _engine_after_team_1_plays()
    day = pd.Timestamp("2026-01-06")
    rating, _ = engine.rating_global(50, day, [11, 12, 97, 98, 99])
    assert rating == engine._seed(50, day)


def test_inheritance_can_be_disabled():
    engine = EloEngine(dataclasses.replace(EloConfig(), inherit_min_players=0), RANKINGS)
    engine.snapshot_series(_series(1, "2026-01-05", 1, 2))
    engine.observe_series(_series(1, "2026-01-05", 1, 2))
    day = pd.Timestamp("2026-01-06")
    rating, _ = engine.rating_global(50, day, LINEUPS[1])
    assert rating == engine._seed(50, day)


def test_same_day_moves_are_not_inherited():
    # Team 1's players turn up for team 50 on the day team 1 plays: the move is only
    # known once that day is folded in, so team 50 still starts from its ranking seed.
    history = pd.concat([
        _series(1, "2026-01-05", 2, 3),
        _series(2, "2026-01-06", 1, 2),
        _series(3, "2026-01-06", 50, 3, lineup_a=LINEUPS[1]),
    ], ignore_index=True)
    features = compute_elo_features(history, RANKINGS).set_index("match_id")
    engine = EloEngine(EloConfig(), RANKINGS)
    assert features.loc[3, "elo_a"] == pytest.approx(engine._seed(50, pd.Timestamp("2026-01-06")))


def test_next_day_moves_are_inherited():
    history = pd.concat([
        _series(1, "2026-01-05", 1, 2),
        _series(2, "2026-01-06", 50, 3, lineup_a=LINEUPS[1]),
    ], ignore_index=True)
    features = compute_elo_features(history, RANKINGS).set_index("match_id")
    engine = EloEngine(EloConfig(), RANKINGS)
    assert features.loc[2, "elo_a"] != pytest.approx(engine._seed(50, pd.Timestamp("2026-01-06")))
    assert features.loc[2, "elo_a"] > features.loc[1, "elo_a"]
