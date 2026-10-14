from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.features.qb_status import (
    REPLACEMENT_DROPBACKS, apply_overrides, historical_starters, load_qb_status, qb_deltas, qb_quality,
    replacement_ppa, team_qb_baseline,
)
from gnomepy_research.models.cfb.features.source import SourceData


def db(rows):
    return pd.DataFrame(rows, columns=["qb_id", "team_id", "season", "kickoff", "play_seq", "ppa"])


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_replacement_uses_only_early_career_dropbacks():
    k = ts("2024-09-01")
    vet = [("v", 1, 2024, k, i, 0.5) for i in range(400)]
    rookie = [("r", 2, 2024, k, i, -0.1) for i in range(50)]
    rep = replacement_ppa(db(vet + rookie))
    expected = (REPLACEMENT_DROPBACKS * 0.5 + 50 * -0.1) / (REPLACEMENT_DROPBACKS + 50)
    assert rep == pytest.approx(expected)


def test_unknown_backup_is_replacement_not_nan():
    k = ts("2024-09-01")
    d = db([("v", 1, 2024, k, i, 0.4) for i in range(300)])
    q = qb_quality(d)
    assert np.isfinite(q.get("backup", replacement_ppa(d)))
    assert q["v"] > replacement_ppa(d) - 1e-12


def test_baseline_is_dropback_weighted():
    k = ts("2024-09-01")
    d = db([("a", 1, 2024, k, i, 0.0) for i in range(90)] + [("b", 1, 2024, k, 100 + i, 0.0) for i in range(10)])
    q = pd.Series({"a": 0.3, "b": -0.1})
    assert team_qb_baseline(d, q)[1] == pytest.approx(0.9 * 0.3 + 0.1 * -0.1)


def make_source() -> SourceData:
    rows = [
        (1, 1, "2024-08-31 16:00", 10, 20),
        (2, 2, "2024-09-07 16:00", 10, 30),
        (3, 3, "2024-09-14 16:00", 10, 40),
    ]
    games = pd.DataFrame([{
        "game_id": gid, "season": 2024, "week": w, "season_type": "regular", "kickoff": ts(k), "completed": True,
        "home_id": h, "away_id": a, "home_team": f"T{h}", "away_team": f"T{a}", "home_points": 30,
        "away_points": 20, "periods": 4, "home_pregame_elo": 1500, "away_pregame_elo": 1500,
        "home_classification": "fbs", "away_classification": "fbs", "neutral_site": False,
    } for gid, w, k, h, a in rows])
    passers = pd.DataFrame([
        {"game_id": gid, "season": 2024, "offense": off, "passer": p, "passer_key": key, "ppa": ppa, "play_seq": i}
        for gid, opp in [(1, 20), (2, 30)]
        for off, p, key, ppa in [("T10", "A Bee", "a.bee", 0.5), (f"T{opp}", "O Pp", "o.pp", 0.0)]
        for i in range(40)
    ])
    roster = pd.DataFrame({
        "season": [2024], "team_id": [10], "player_id": ["1"], "position": ["QB"], "first_name": ["Al"],
        "last_name": ["Bee"], "available_at": [ts("2024-08-01")],
    })
    empty_eff = pd.DataFrame({
        "game_id": [1], "season": [2024], "team_id": [10], "opponent_id": [20], "plays": [60], "off_ppa": [0.2],
        "available_at": [ts("2024-09-01 16:00")], "usable": [True],
    })

    def avail(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(available_at=ts("2024-08-30"))

    return SourceData.from_tables({
        "cfb_games": games,
        "cfb_team_game_efficiency": empty_eff,
        "cfb_passers": passers,
        "cfb_talent": avail(pd.DataFrame({"season": [2024], "team_id": [10], "talent": [700.0]})),
        "cfb_recruiting_teams": avail(pd.DataFrame({"class_year": [2025], "team_id": [10], "points": [200.0]})),
        "cfb_recruits": avail(pd.DataFrame({"class_year": [2024], "athlete_id": ["1"], "rating": [0.9]})),
        "cfb_returning": avail(pd.DataFrame({"season": [2024], "team_id": [10], "pct_ppa": [0.5]})),
        "cfb_portal": pd.DataFrame({
            "season": [2024], "origin_id": [20], "destination_id": [10], "destination": ["T10"],
            "origin_available_at": [ts("2024-01-05")], "destination_available_at": [ts("2024-08-30")],
        }),
        "cfb_coaches": pd.DataFrame({
            "coach_id": [1], "team_id": pd.array([10], dtype="Int64"), "season": [2023], "hire_date": [ts("2019-12-01")],
            "games": [12], "wins": [0], "losses": [0], "srs": [5.0], "sp_overall": [0.0], "sp_offense": [0.0],
            "sp_defense": [0.0], "postseason_rank": [None], "results_available_at": [ts("2024-02-01")],
        }),
        "cfb_rosters": roster,
    })


def test_backup_with_no_history_lowers_delta_only_for_the_game_he_starts():
    src = make_source()
    starters = pd.DataFrame({"game_id": [1, 2, 3], "team_id": [10, 10, 10], "qb_id": ["1", "1", "backup"]})
    out = qb_deltas(src, src.games, starters).set_index("game_id")
    assert out.loc[2, "home_qb_delta"] == pytest.approx(0.0)
    assert out.loc[3, "home_qb_delta"] < 0
    assert out.loc[3, "away_qb_delta"] == 0.0


def test_historical_starters_are_first_passers():
    src = make_source()
    s = historical_starters(src).set_index(["game_id", "team_id"])["qb_id"]
    assert s[(1, 10)] == "1"


HEADER = "season,season_type,week,team,passer_key,recorded_at\n"


def write_csv(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "qb.csv"
    p.write_text(HEADER + body)
    return p


def test_unknown_team_raises_with_name(tmp_path):
    p = write_csv(tmp_path, "2024,regular,3,Nowhere St,j.doe,2024-09-13T12:00Z\n")
    with pytest.raises(ValueError, match="Nowhere St"):
        load_qb_status(p, make_source())


def test_rostered_passer_gets_player_id_and_unknown_gets_synthetic(tmp_path):
    p = write_csv(tmp_path, "2024,regular,3,T10, A.Bee ,2024-09-13T12:00Z\n2024,regular,3,T40,j.doe,2024-09-13T12:00Z\n")
    out = load_qb_status(p, make_source()).set_index("team_id")["qb_id"]
    assert out[10] == "1" and out[40] == "t40:j.doe"


def test_override_for_upcoming_game_has_game_id_and_utc_time(tmp_path):
    p = write_csv(tmp_path, "2024,regular,3,T10,a.bee,2024-09-13T12:00Z\n")
    out = load_qb_status(p, make_source())
    assert list(out.columns) == ["game_id", "team_id", "qb_id", "recorded_at"]
    assert out.loc[0, "game_id"] == 3 and out.loc[0, "recorded_at"] == ts("2024-09-13 12:00")


def test_empty_csv_loads_empty_frame(tmp_path):
    out = load_qb_status(write_csv(tmp_path, ""), make_source())
    assert out.empty and list(out.columns) == ["game_id", "team_id", "qb_id", "recorded_at"]


def test_duplicate_game_team_keeps_latest_recorded(tmp_path):
    p = write_csv(tmp_path, "2024,regular,3,T10,x.old,2024-09-10T12:00Z\n2024,regular,3,T10,y.new,2024-09-12T12:00Z\n")
    out = load_qb_status(p, make_source())
    assert out["qb_id"].tolist() == ["t10:y.new"]


def test_missing_game_raises_with_row_number(tmp_path):
    p = write_csv(tmp_path, "2024,regular,3,T10,a.bee,2024-09-13T12:00Z\n2024,regular,9,T10,a.bee,2024-09-13T12:00Z\n")
    with pytest.raises(ValueError, match=r"row 2.*T10"):
        load_qb_status(p, make_source())


def test_ambiguous_game_raises_with_row_number(tmp_path):
    src = make_source()
    dup = src.games[src.games["game_id"] == 3].assign(game_id=99, away_id=50, away_team="T50")
    src = replace(src, games=pd.concat([src.games, dup], ignore_index=True))
    p = write_csv(tmp_path, "2024,regular,3,T10,a.bee,2024-09-13T12:00Z\n")
    with pytest.raises(ValueError, match=r"row 1.*T10"):
        load_qb_status(p, src)


def test_override_replaces_starter():
    starters = pd.DataFrame({"game_id": [1, 1], "team_id": [10, 20], "qb_id": ["a", "b"]})
    over = pd.DataFrame({"game_id": [1, 2], "team_id": [10, 10], "qb_id": ["backup", "backup"]})
    out = apply_overrides(starters, over).set_index(["game_id", "team_id"])["qb_id"]
    assert out[(1, 10)] == "backup" and out[(1, 20)] == "b" and out[(2, 10)] == "backup"
