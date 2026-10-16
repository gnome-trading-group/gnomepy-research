import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cs2.hltv.build_priors import build_priors_for

MAPS = ["de_mirage", "de_inferno", "de_nuke"]
TEAMS = {1: "Alpha", 2: "Bravo", 3: "Charlie", 4: "Delta"}
START = pd.Timestamp("2026-09-01")


def _lineup(team_id: int) -> np.ndarray:
    return np.array([team_id * 100 + i for i in range(5)])


def _series(match_id: int, day: pd.Timestamp, a: int, b: int, rng: np.random.Generator) -> list[dict]:
    rows, a_maps, b_maps = [], 0, 0
    for pos, map_name in enumerate(MAPS, start=1):
        if a_maps == 2 or b_maps == 2:
            break
        a_won = int(rng.random() < 0.5)
        a_score, b_score = (13, int(rng.integers(3, 12))) if a_won else (int(rng.integers(3, 12)), 13)
        rows.append({
            "match_id": match_id, "match_date": day, "event_name": "Test Cup", "event_tier": 1,
            "format": "Best of 3 (Online)", "bo_type": 3, "is_lan": 0,
            "team_a_name": TEAMS[a], "team_a_id": a, "team_b_name": TEAMS[b], "team_b_id": b,
            "team_a_rank": float(a), "team_b_rank": float(b), "map_name": map_name, "map_position_in_series": pos,
            "is_decider": float(pos == 3), "team_a_series_score": a_maps, "team_b_series_score": b_maps,
            "team_a_score": a_score, "team_b_score": b_score, "team_a_won": a_won,
            "team_a_picked_map": float(pos == 1), "team_a_h1_score": float(min(a_score, 7)),
            "team_b_h1_score": float(min(b_score, 5)), "team_a_started_ct": float(pos % 2),
            "team_a_player_ids": _lineup(a), "team_b_player_ids": _lineup(b),
            "team_a_lineup_ids": _lineup(a), "team_b_lineup_ids": _lineup(b),
            "team_a_avg_rating": 1.0, "team_b_avg_rating": 1.0,
            "team_a_vrs_rank": float(a), "team_b_vrs_rank": float(b),
            "stage_text": "Group A", "is_elimination": 0.0, "is_qualifier": 0.0, "is_playoff": 0.0, "is_group": 1.0,
        })
        a_maps, b_maps = a_maps + a_won, b_maps + (1 - a_won)
    return rows


def _history() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    pairs = [(1, 2), (3, 4), (1, 3), (2, 4), (1, 4), (2, 3)]
    rows = []
    for day in range(30):
        a, b = pairs[day % len(pairs)]
        rows += _series(1000 + day, START + pd.Timedelta(days=day), a, b, rng)
    return pd.DataFrame(rows)


def _rankings() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": START + pd.Timedelta(days=week * 7), "team_name": name, "team_id": tid, "rank": tid,
         "points": 100 - tid, "change": 0}
        for week in range(5) for tid, name in TEAMS.items()
    ])


def _target(match_id: int, day: pd.Timestamp, a: int, b: int) -> dict:
    """An upcoming series the way the upcoming pipeline builds one: no map, no outcome."""
    return {
        "match_id": match_id, "match_date": day, "event_name": "Test Cup", "event_tier": 1, "bo_type": 3, "is_lan": 0,
        "team_a_name": TEAMS[a], "team_a_id": a, "team_b_name": TEAMS[b], "team_b_id": b,
        "team_a_rank": float(a), "team_b_rank": float(b), "team_a_vrs_rank": float(a), "team_b_vrs_rank": float(b),
        "team_a_lineup_ids": _lineup(a), "team_b_lineup_ids": _lineup(b),
        "stage_text": "Playoffs", "is_elimination": 1.0, "is_qualifier": 0.0, "is_playoff": 1.0, "is_group": 0.0,
        "map_name": "", "map_position_in_series": 1, "team_a_series_score": 0, "team_b_series_score": 0,
        "is_decider": 0.0,
    }


def _features(out: pd.DataFrame, match_id: int) -> pd.Series:
    return out[out.match_id == match_id].drop(columns=["match_date"]).iloc[0]


def test_a_team_with_upcoming_matches_on_two_days_prices_each_as_if_alone():
    history, rankings = _history(), _rankings()
    last = history.match_date.max()
    today = _target(2001, last + pd.Timedelta(days=1), 1, 2)
    tomorrow = _target(2002, last + pd.Timedelta(days=2), 1, 3)

    both = build_priors_for(history, rankings, pd.DataFrame([today, tomorrow]))
    alone = build_priors_for(history, rankings, pd.DataFrame([tomorrow]))

    assert sorted(both.match_id) == [2001, 2002]
    pd.testing.assert_series_equal(_features(both, 2002), _features(alone, 2002), check_names=False)
    assert not np.isnan(_features(both, 2002)[["elo_a", "team_a_recent_form", "own_h2h_rate"]].astype(float)).any()


def test_same_day_targets_share_a_pass_without_reading_each_other():
    history, rankings = _history(), _rankings()
    day = history.match_date.max() + pd.Timedelta(days=1)
    first, second = _target(2001, day, 1, 2), _target(2002, day, 1, 3)

    both = build_priors_for(history, rankings, pd.DataFrame([first, second]))
    alone = build_priors_for(history, rankings, pd.DataFrame([second]))

    pd.testing.assert_series_equal(_features(both, 2002), _features(alone, 2002), check_names=False)


@pytest.mark.parametrize("targets", [[], None])
def test_no_targets_gives_no_rows(targets):
    frame = pd.DataFrame(targets or [], columns=["match_id", "match_date", "map_name"])
    assert build_priors_for(_history(), _rankings(), frame).empty
