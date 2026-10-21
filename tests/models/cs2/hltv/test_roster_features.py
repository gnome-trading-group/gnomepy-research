import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cs2.hltv.pit import assert_pit_consistent
from gnomepy_research.models.cs2.hltv.roster_features import (
    LINEUPS_AUX,
    ROSTER_FEATURES,
    _RosterAccumulator,
    compute_roster_features,
)

FIVES = {1: [11, 12, 13, 14, 15], 2: [21, 22, 23, 24, 25]}
DAY0 = pd.Timestamp("2026-01-01")


def _series(n_days=8):
    rows = [{"match_id": 100 + d, "match_date": DAY0 + pd.Timedelta(days=d), "map_name": "de_mirage",
             "map_position_in_series": 1, "team_a_id": 1, "team_b_id": 2, "team_a_won": 1,
             "team_a_score": 13, "team_b_score": 7} for d in range(n_days)]
    return pd.DataFrame(rows)


def _lineups(history, five_a=None, standins=(), only=None):
    """Every series' announced lineups; five_a/standins apply to match `only` (or every match if None)."""
    rows = []
    for mid in history.match_id.unique():
        target = only is None or mid == only
        for team, five in ((1, five_a if (five_a and target) else FIVES[1]), (2, FIVES[2])):
            for slot, p in enumerate(five):
                rows.append({"match_id": mid, "team_id": team, "player_id": p, "slot": slot,
                             "is_standin": target and p in standins})
    return pd.DataFrame(rows)


def _move(day, person, kind="join", role="player", frm=0, to=1, from_status=None, to_status="active"):
    return {"event_date": pd.Timestamp(day), "kind": kind, "person_id": person, "person_role": role,
            "from_team_id": frm, "to_team_id": to, "from_status": from_status, "to_status": to_status}


def _last(out, history):
    return out.set_index("match_id").loc[history.match_id.max()]


def test_settled_five_is_all_core():
    h = _series()
    row = _last(compute_roster_features(h, _lineups(h)), h)
    assert row.team_a_roster_core_present == 5 and row.team_a_roster_core_missing == 0
    assert row.team_a_roster_new_faces == 0 and row.team_a_roster_usage_min == 1.0


def test_a_new_face_replaces_a_core_player():
    h = _series()
    last = h.match_id.max()
    row = _last(compute_roster_features(h, _lineups(h, five_a=[11, 12, 13, 14, 99], only=last)), h)
    assert row.team_a_roster_new_faces == 1 and row.team_a_roster_core_missing == 1
    assert row.team_a_roster_usage_min == 0.0


def test_too_little_history_has_no_core():
    h = _series(n_days=3)
    row = _last(compute_roster_features(h, _lineups(h)), h)
    assert np.isnan(row.team_a_roster_core_present)


def test_standin_who_is_the_coach_is_flagged():
    h = _series()
    last = h.match_id.max()
    lineups = _lineups(h, five_a=[11, 12, 13, 14, 77], standins=(77,), only=last)
    moves = pd.DataFrame([_move("2025-09-01", 77, role="coach", to_status="coach")])
    row = _last(compute_roster_features(h, lineups, moves), h)
    assert row.team_a_roster_standins == 1 and row.team_a_roster_standin_coach == 1
    assert row.team_a_roster_coach_change_90 == 0


def test_recent_joiner_counts_for_thirty_days_from_the_move():
    h = _series(n_days=40)
    moves = pd.DataFrame([_move(DAY0 + pd.Timedelta(days=2), 15)])
    out = compute_roster_features(h, _lineups(h), moves).set_index("match_id")
    by_day = out.team_a_roster_recent_joiners.to_numpy()
    assert by_day[2] == 0, "a move dated the match day is not yet known"
    assert by_day[3] == 1 and by_day[32] == 1 and by_day[33] == 0


def test_core_player_leaving_counts():
    h = _series()
    moves = pd.DataFrame([_move(DAY0 + pd.Timedelta(days=6), 15, kind="leave", frm=1, to=0,
                                from_status="active", to_status=None)])
    row = _last(compute_roster_features(h, _lineups(h), moves), h)
    assert row.team_a_roster_core_lost_30 == 1 and row.team_a_roster_moves_30 == 1
    assert row.team_a_roster_days_since_move == 1


def test_missing_lineup_side_is_nan():
    h = _series()
    lineups = _lineups(h)
    lineups = lineups[~((lineups.match_id == h.match_id.max()) & (lineups.team_id == 1))]
    row = _last(compute_roster_features(h, lineups), h)
    assert np.isnan(row.team_a_roster_moves_30) and not np.isnan(row.team_b_roster_moves_30)
    assert np.isnan(row.roster_moves_30_diff)


def test_swapping_sides_mirrors_every_feature():
    h = _series()
    swapped = h.assign(team_a_id=2, team_b_id=1)
    lineups = _lineups(h, five_a=[11, 12, 13, 14, 99], only=h.match_id.max())
    ab, ba = _last(compute_roster_features(h, lineups), h), _last(compute_roster_features(swapped, lineups), h)
    for name in ROSTER_FEATURES:
        if name.endswith("_diff"):
            assert ab[name] == pytest.approx(-ba[name], nan_ok=True)
        else:
            mirror = name.replace("team_a_", "team_x_").replace("team_b_", "team_a_").replace("team_x_", "team_b_")
            assert ab[name] == pytest.approx(ba[mirror], nan_ok=True)


def test_point_in_time():
    h = _series(n_days=10)
    moves = pd.DataFrame([_move(DAY0 + pd.Timedelta(days=4), 15)])
    assert_pit_consistent(h, lambda: _RosterAccumulator(moves), aux={LINEUPS_AUX: _lineups(h)}, sample=6)
