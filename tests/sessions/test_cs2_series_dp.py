import numpy as np
import pandas as pd
import pytest

from gnomepy_research.sessions.cs2_win_probability.series_dp import (
    _MAP_SPECIFIC,
    build_node_frame,
    series_win_prob,
)

SERIES_LEVEL = {"match_id": 7, "elo_diff": 42.0, "rank_best_diff": -3.0, "team_a_banned_first": 1.0}


def _row(position, map_name, winrate, picked):
    return {**SERIES_LEVEL, "map_position_in_series": position, "map_name": map_name,
            "is_decider": float(position == 3),
            "team_a_map_winrate_long": winrate, "team_b_map_winrate_long": 1 - winrate,
            "team_a_map_winrate_short": winrate, "team_b_map_winrate_short": 1 - winrate,
            "elo_map_diff": 10.0 * position, "elo_map_resid_diff": 1.0 * position,
            "team_a_picked_map": picked, "team_a_picked_map_exact": picked, "map_pick_index": float(position),
            "team_a_series_score": 0.0, "team_b_series_score": 0.0}


def _bo3(played):
    rows = [_row(1, "de_mirage", 0.60, 1.0), _row(2, "de_nuke", 0.40, 0.0), _row(3, "de_ancient", 0.75, np.nan)]
    return pd.DataFrame(rows[:played])


def _predict(frame):
    """A stand-in model that is sensitive to every input the leak could travel through."""
    wr = frame["team_a_map_winrate_long"].fillna(0.5)
    picked = frame["team_a_picked_map_exact"].fillna(0.5)
    logit = 0.01 * frame["elo_diff"] + 2.0 * (wr - 0.5) + 0.3 * (picked - 0.5) \
        + 0.2 * (frame["team_a_series_score"] - frame["team_b_series_score"])
    return (1 / (1 + np.exp(-logit))).to_numpy()


def test_inputs_do_not_depend_on_whether_the_decider_was_played():
    """
    History only records played maps. A 2-0 series has no map-3 row and a 2-1
    series does; if that changed the model's inputs, the inputs would encode the
    series going the distance - part of the outcome.
    """
    two_nil, _ = build_node_frame(_bo3(played=2), 2, [])
    two_one, _ = build_node_frame(_bo3(played=3), 2, [])
    pd.testing.assert_frame_equal(two_nil, two_one)


def test_series_probability_is_identical_for_two_nil_and_two_one():
    probs = []
    for played in (2, 3):
        frame, nodes = build_node_frame(_bo3(played), 2, [])
        probs.append(series_win_prob(dict(zip(nodes, _predict(frame))), 2, n_veto_maps=3))
    assert probs[0] == pytest.approx(probs[1], abs=1e-12)


def test_guaranteed_maps_keep_their_real_rows():
    frame, nodes = build_node_frame(_bo3(played=3), 2, [])
    by_node = dict(zip(nodes, frame.to_dict("records")))
    assert by_node[(0, 0)]["map_name"] == "de_mirage"
    assert by_node[(0, 0)]["team_a_map_winrate_long"] == 0.60
    for node in ((1, 0), (0, 1)):
        assert by_node[node]["map_name"] == "de_nuke"
        assert by_node[node]["team_a_picked_map_exact"] == 0.0


def test_unknown_decider_blanks_map_features_but_keeps_series_features():
    frame, nodes = build_node_frame(_bo3(played=3), 2, [])
    decider = dict(zip(nodes, frame.to_dict("records")))[(1, 1)]
    assert decider["map_name"] == ""
    assert decider["is_decider"] == 1.0
    for c in _MAP_SPECIFIC:
        assert np.isnan(decider[c]), c
    for c, v in SERIES_LEVEL.items():
        assert decider[c] == v, c
    assert (decider["team_a_series_score"], decider["team_b_series_score"]) == (1.0, 1.0)


def test_decider_does_not_inherit_who_picked_the_previous_map():
    """Copying series-level features from map 2 once leaked map 2's picker into the decider."""
    frame, nodes = build_node_frame(_bo3(played=2), 2, [])
    decider = dict(zip(nodes, frame.to_dict("records")))[(1, 1)]
    assert np.isnan(decider["team_a_picked_map_exact"])
    assert np.isnan(decider["map_pick_index"])


def test_bo5_positions_beyond_three_are_unknown_and_only_the_fifth_is_the_decider():
    rows = [_row(p, f"de_map{p}", 0.5 + 0.05 * p, 1.0) for p in range(1, 6)]
    frame, nodes = build_node_frame(pd.DataFrame(rows), 3, [])
    for rec, (a, b) in zip(frame.to_dict("records"), nodes):
        position = a + b + 1
        if position <= 3:
            assert rec["map_name"] == f"de_map{position}"
        else:
            assert rec["map_name"] == ""
            assert rec["is_decider"] == float(position == 5)


def test_series_without_a_guaranteed_map_is_rejected():
    with pytest.raises(ValueError, match="no played map"):
        build_node_frame(pd.DataFrame([_row(3, "de_ancient", 0.5, np.nan)]), 2, [])
