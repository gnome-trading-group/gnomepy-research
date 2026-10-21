import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cs2.hltv.build_priors import _TeamHistoryAccumulator, compute_team_history_priors
from gnomepy_research.models.cs2.hltv.pit import assert_pit_consistent

DAY0 = pd.Timestamp("2026-01-01")
RANKINGS = pd.DataFrame({"date": [DAY0, DAY0], "team_id": [1, 2], "points": [900, 300], "rank": [3, 20]})


def _map(match_id, day, a, b, a_won, map_name="de_mirage", pos=1):
    return {"match_id": match_id, "match_date": DAY0 + pd.Timedelta(days=day), "map_name": map_name,
            "map_position_in_series": pos, "team_a_id": a, "team_b_id": b, "team_a_name": f"T{a}",
            "team_b_name": f"T{b}", "team_a_won": a_won, "team_a_score": 13 if a_won else 7,
            "team_b_score": 7 if a_won else 13}


def _row(out, match_id, map_name="de_mirage"):
    return out.set_index(["match_id", "map_name"]).loc[(match_id, map_name)]


def test_recent_form_is_the_last_ten_maps_within_sixty_days():
    rows = [_map(i, 1, 1, 3, 1) for i in range(5)] + [_map(10 + i, 70, 1, 3, i % 2) for i in range(12)]
    rows.append(_map(99, 75, 1, 2, 1))
    row = _row(compute_team_history_priors(pd.DataFrame(rows), RANKINGS), 99)
    assert row.team_a_recent_form == pytest.approx(0.5)
    assert row.team_a_overall_winrate == pytest.approx((5 + 6) / 17)


def test_same_day_results_are_not_visible():
    rows = [_map(1, 5, 1, 2, 1), _map(2, 5, 1, 2, 1), _map(3, 6, 1, 2, 0)]
    out = compute_team_history_priors(pd.DataFrame(rows), RANKINGS)
    assert np.isnan(_row(out, 1).team_a_recent_form) and np.isnan(_row(out, 2).team_a_recent_form)
    assert _row(out, 3).team_a_recent_form == 1.0


def test_head_to_head_is_read_from_each_sides_view():
    rows = [_map(1, 1, 1, 2, 1), _map(2, 2, 2, 1, 1), _map(3, 3, 1, 2, 1), _map(4, 4, 2, 1, 0)]
    out = compute_team_history_priors(pd.DataFrame(rows), RANKINGS)
    assert _row(out, 4).h2h_win_rate == pytest.approx(1 / 3)


def test_map_winrate_counts_only_that_map():
    rows = [_map(1, 1, 1, 3, 1, "de_nuke"), _map(2, 2, 1, 3, 0, "de_mirage"), _map(3, 3, 1, 2, 1, "de_nuke")]
    out = compute_team_history_priors(pd.DataFrame(rows), RANKINGS)
    assert _row(out, 3, "de_nuke").team_a_map_winrate_long == 1.0
    assert _row(out, 3, "de_nuke").team_a_overall_winrate == 0.5


def test_stale_ranking_is_unknown_but_keeps_its_age():
    out = compute_team_history_priors(pd.DataFrame([_map(1, 10, 1, 2, 1), _map(2, 150, 1, 2, 1)]), RANKINGS)
    fresh, stale = _row(out, 1), _row(out, 2)
    assert fresh.team_a_rating_diff == 600 and fresh.rank_diff == -17
    assert np.isnan(stale.team_a_rating_diff) and stale.ranking_age_days == 150


def test_point_in_time():
    rng = np.random.default_rng(0)
    rows = [_map(100 + d * 3 + k, d, int(a), int(b), int(rng.integers(0, 2)), m, pos)
            for d in range(12) for k, (a, b) in enumerate([(1, 2), (2, 3), (3, 1)])
            for pos, m in enumerate(("de_nuke", "de_mirage"), start=1)]
    assert_pit_consistent(pd.DataFrame(rows), lambda: _TeamHistoryAccumulator(RANKINGS), sample=6)
