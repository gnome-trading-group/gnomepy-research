from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.features.physical import (
    blue_chip_ratio,
    physical_features,
    physical_table,
    top_weight,
)
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import physical_diffs, physical_match_rate


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def recruits(team_id: int, stars: list[float], class_year: int = 2024, available: str = "2024-02-01") -> pd.DataFrame:
    return pd.DataFrame({"recruit_type": "HighSchool", "class_year": class_year, "team_id": float(team_id),
                         "stars": stars, "available_at": ts(available)})


def roster(team_id: int, positions: list[str], weights: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"season": 2024, "team_id": float(team_id), "position": positions, "weight": weights,
                         "available_at": ts("2024-08-01")})


def source(rec: pd.DataFrame, ro: pd.DataFrame) -> SourceData:
    games = pd.DataFrame({"season": [2024], "kickoff": [ts("2024-08-31 16:00")], "home_points": [30.0],
                          "away_points": [20.0], "periods": [4], "completed": [True],
                          "home_pregame_elo": [1500.0], "away_pregame_elo": [1500.0]})
    talent = pd.DataFrame({"season": 2024, "team_id": [1.0, 2.0], "talent": [900.0, 600.0], "available_at": ts("2024-07-01")})
    empty = pd.DataFrame({"available_at": pd.Series(dtype="datetime64[ns, UTC]")})
    portal = pd.DataFrame({"origin_available_at": pd.Series(dtype="datetime64[ns, UTC]"),
                           "destination_available_at": pd.Series(dtype="datetime64[ns, UTC]"),
                           "destination": pd.Series(dtype=object), "destination_id": pd.Series(dtype=float)})
    coaches = pd.DataFrame({"assignment_available_at": pd.Series(dtype="datetime64[ns, UTC]"),
                            "results_available_at": pd.Series(dtype="datetime64[ns, UTC]"),
                            **{c: pd.Series(dtype=float) for c in ("games", "wins", "losses", "srs", "sp_overall",
                                                                   "sp_offense", "sp_defense", "postseason_rank")}})
    return SourceData(games=games, efficiency=empty, passers=empty, talent=talent, recruiting_teams=empty,
                      recruits=rec, returning=empty, portal=portal, coaches=coaches, rosters=ro)


def test_blue_chip_ratio_counts_four_and_five_stars():
    r = recruits(1, [5, 4, 4, 3, 3, 3, 3, 3, 3, 3] * 2)
    assert blue_chip_ratio(r, 2024).loc[1] == 0.3


def test_blue_chip_ratio_nan_below_twenty_signees():
    r = recruits(1, [5.0] * 19)
    assert np.isnan(blue_chip_ratio(r, 2024).reindex([1]).iloc[0])


def test_blue_chip_ratio_uses_last_four_classes():
    r = pd.concat([recruits(1, [5.0] * 20, 2021), recruits(1, [3.0] * 20, 2020)])
    assert blue_chip_ratio(r, 2024).loc[1] == 1.0


def test_top_weight_takes_heaviest_k_and_ignores_other_positions():
    ro = roster(1, ["OL", "OT", "G", "C", "OL", "OL", "DL", "QB"], [300, 310, 320, 290, 280, 0, 400, 500])
    assert top_weight(ro, {"OL", "OT", "G", "C"}, 5).loc[1] == np.mean([300, 310, 320, 290, 280])
    assert top_weight(ro, {"DL"}, 4).loc[1] == 400


def test_missing_weights_are_nan():
    ro = roster(1, ["OL", "OL"], [0, np.nan])
    assert top_weight(ro, {"OL"}, 5).empty


def test_point_in_time_excludes_late_recruits_and_rosters():
    late = recruits(1, [5.0] * 20, 2024, "2024-09-15")
    src = source(pd.concat([recruits(1, [3.0] * 20), late]), roster(1, ["OL"], [300.0]))
    src.rosters.loc[:, "available_at"] = ts("2024-09-01")
    row = physical_features(src, 2024).set_index("team_id").loc[1]
    assert row["blue_chip_ratio"] == 0.0
    assert np.isnan(row["ol_weight"])
    assert row["talent"] == 900.0


def table() -> pd.DataFrame:
    return pd.DataFrame({"season": 2024, "team_id": [1, 2, 3], "talent": [900.0, 600.0, 750.0],
                         "blue_chip_ratio": [0.6, 0.1, 0.3], "ol_weight": [320.0, 290.0, 305.0],
                         "dl_weight": [300.0, 270.0, 285.0]})


def games(pairs: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame({"season": 2024, "home_id": [h for h, _ in pairs], "away_id": [a for _, a in pairs]})


def test_outcome_terms_are_antisymmetric_under_swap():
    fwd = physical_diffs(games([(1, 2), (2, 3)]), table())
    rev = physical_diffs(games([(2, 1), (3, 2)]), table())
    np.testing.assert_allclose(fwd, -rev)
    assert (fwd[0] > 0).all()


def test_missing_team_scores_zero():
    d = physical_diffs(games([(1, 99), (98, 99)]), table())
    z_talent = (900 - table()["talent"].mean()) / table()["talent"].std()
    assert d[0, 0] == z_talent
    np.testing.assert_array_equal(d[1], 0.0)


def test_physical_table_one_row_per_team_season():
    src = source(recruits(1, [4.0] * 20), roster(1, ["OL"], [300.0]))
    t = physical_table(src, [2024])
    assert list(t.columns) == ["season", "team_id", "talent", "blue_chip_ratio", "ol_weight", "dl_weight"]
    assert t["team_id"].is_unique


def test_float_and_string_team_ids_still_match():
    t = table().assign(team_id=lambda d: d["team_id"].astype(float))
    g = games([(1, 2), (2, 3)]).astype({"home_id": str, "away_id": str})
    assert physical_match_rate(g, t) == 1.0
    np.testing.assert_allclose(physical_diffs(g, t, min_match=0.95), physical_diffs(games([(1, 2), (2, 3)]), table()))


def test_low_match_rate_raises():
    with pytest.raises(ValueError, match="match rate"):
        physical_diffs(games([(1, 99), (98, 99)]), table(), min_match=0.95)
