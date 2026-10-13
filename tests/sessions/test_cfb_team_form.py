import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.features.team_form import (
    FORM_FEATURES,
    is_garbage,
    is_success,
    team_form,
    team_game_form,
    with_luck,
)


def test_success_definition():
    down = np.array([1, 1, 2, 2, 3, 4])
    dist = np.array([10, 10, 10, 10, 5, 1])
    gain = np.array([5, 4, 7, 6, 5, 0])
    assert is_success(down, dist, gain).tolist() == [True, False, True, False, True, False]


def test_garbage_time_thresholds():
    assert is_garbage(np.array([2, 2, 3, 4, 1]), np.array([45, 44, 35, 30, 50]), np.array([6, 7, 6, 7, 0])).tolist() == [
        True, False, True, True, False]


def rows(team_vals: dict[int, list[float]], col: str) -> pd.DataFrame:
    out = []
    for t, vals in team_vals.items():
        for v in vals:
            out.append({"team_id": t, "season": 2024, **{f: 0.0 for f in FORM_FEATURES}, col: v})
    return pd.DataFrame(out)


def test_team_with_more_games_moves_further_from_league():
    f = team_form(rows({1: [0.6], 2: [0.6] * 6, 3: [0.4] * 6}, "early_success"), 2024)
    assert f.loc[2, "early_success"] > f.loc[1, "early_success"] > 0


def test_no_rows_is_empty():
    assert team_form(rows({}, "pace").reindex(columns=["team_id", "season", *FORM_FEATURES]), 2024).empty


def test_constant_column_gives_zero_not_nan():
    f = team_form(rows({1: [70.0], 2: [70.0]}, "pace"), 2024)
    assert (f["pace"] == 0).all()


def test_with_luck_signs_late_success_above_prediction_positive():
    df = pd.DataFrame({
        "early_success": [0.4, 0.5, 0.6, 0.5],
        "late_success": [0.4, 0.5, 0.6, 0.9],
        "pts_per_qd": [4.0, 4.0, 4.0, 6.0],
    })
    out = with_luck(df)
    assert out["late_down_luck"].iloc[3] > 0
    assert out["finishing_luck"].iloc[3] > 0
    assert out["finishing_luck"].mean() == pytest.approx(0.0)


def _play(offense, defense, period, off_score, def_score, down, distance, gain, ppa=0.1):
    return dict(game_id=1, season=2024, offense=offense, defense=defense, period=period,
                offense_score=off_score, defense_score=def_score, down=down, distance=distance,
                yards_gained=gain, ppa=ppa)


def test_team_game_form_hand_computed():
    games = pd.DataFrame([dict(game_id=1, season=2024, kickoff=pd.Timestamp("2024-09-01 12:00"),
                               home_id=10, away_id=20, home_team="A", away_team="B")])
    plays = pd.DataFrame([
        _play("A", "B", 1, 0, 0, 1, 10, 6),
        _play("A", "B", 1, 0, 0, 2, 4, 1),
        _play("A", "B", 1, 0, 0, 3, 3, 3),
        _play("A", "B", 4, 40, 10, 1, 10, 10),
        _play("B", "A", 1, 0, 0, 1, 10, 2),
        _play("B", "A", 1, 0, 0, 3, 5, 6),
    ])
    drives = pd.DataFrame([
        dict(game_id=1, offense_id=10, defense_id=20, start_yards_to_goal=75, end_yards_to_goal=0, off_points=7),
        dict(game_id=1, offense_id=10, defense_id=20, start_yards_to_goal=55, end_yards_to_goal=90, off_points=0),
        dict(game_id=1, offense_id=20, defense_id=10, start_yards_to_goal=80, end_yards_to_goal=30, off_points=3),
        dict(game_id=1, offense_id=20, defense_id=10, start_yards_to_goal=60, end_yards_to_goal=70, off_points=0),
    ])
    out = team_game_form(plays, drives, games).set_index("team_id")
    assert out.loc[10, "pace"] == 3
    assert out.loc[20, "pace"] == 2
    assert out.loc[10, "early_success"] == pytest.approx(0.5)
    assert out.loc[10, "late_success"] == pytest.approx(1.0)
    assert out.loc[10, "pts_per_qd"] == pytest.approx(7.0)
    assert out.loc[20, "pts_per_qd"] == pytest.approx(3.0)
    assert out.loc[10, "field_position"] == pytest.approx(5.0)
    assert out.loc[20, "field_position"] == pytest.approx(-5.0)
    assert out.loc[10, "available_at"] == pd.Timestamp("2024-09-02 12:00")
    assert (out["season"] == 2024).all()
