from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.efficiency.data import (
    TRANSITION_MASKED, UNITS_SPLIT, build_model_data, canonical_extensions, continuity_features, defensive_continuity,
    prior_fixes)
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import MODEL_CONFIGS
from tests.sessions.test_cfb_prior_fixes import preseason, schedule
from tests.sessions.test_cfb_win_probability_models import SEASON, synthetic


def test_trans_keeps_move_up_numbers_and_flags_them():
    out = prior_fixes(preseason([1, 2]), schedule({1: 2018, 2: 2022}), keep_transition=True).set_index(["team_id", "season"])
    assert out.loc[(2, 2023), TRANSITION_MASKED].notna().all()
    assert out["transition"].xs(2).loc[2021:2025].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert (out["transition"].xs(1) == 0).all()


def test_trans_needs_prior_and_names_follow_canonical_order():
    with pytest.raises(ValueError, match="refines prior"):
        canonical_extensions(("trans",), drives=True)
    assert canonical_extensions(("cont", "trans", "prior", "couple"), drives=True) == ("couple", "prior", "trans", "cont")
    kw = dict(units=UNITS_SPLIT, conference=True, drives=True, independents_separate=True)
    assert draws_name(2024, **kw, extensions=("couple", "fcs", "finish", "prior", "trans", "cont")) == \
        "cfb_team_week_draws_split_conf_ind_drv_cpl_fcs_fin_pfx_trx_cnt_2024"


def test_defensive_continuity_is_class_weighted_and_needs_both_rosters():
    r = pd.DataFrame({
        "season": [2023, 2023, 2023, 2023, 2024, 2024, 2023],
        "team_id": [1, 1, 1, 1, 1, 1, 2],
        "player_id": [10, 11, 12, 13, 11, 14, 20],
        "position": ["DL", "LB", "WR", "CB", "LB", "S", "DL"],
        "class_year": [4, 2, 3, 2018, 3, 1, 3]})       # 2018: a bad class value, ignored
    out = defensive_continuity(r).set_index(["season", "team_id"])["def_returning_share"]
    assert out.to_dict() == {(2024, 1): pytest.approx(2 / 6)}   # LB (2) back of DL (4) + LB (2); team 2 has no 2024 roster


def test_continuity_features_and_model_data():
    with pytest.raises(ValueError, match="def_returning_share"):
        continuity_features(pd.DataFrame({"season": [2024], "prior_off_rating": [0.1]}))
    games, eff, pre, _, _ = synthetic()
    pre = pre.assign(def_returning_share=np.linspace(0.3, 0.9, len(pre)))
    plain = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, extensions=("prior",))
    ext = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, extensions=("prior", "trans", "cont"))
    for name in ("transition", "prior_off_x_returning"):
        assert f"pass:{name}" in ext.off_names and f"pass:{name}" not in plain.off_names
    for name in ("transition", "def_returning_share", "prior_def_x_returning"):
        assert f"rush:{name}" in ext.def_names
    assert ext.z_unc.shape[1] == plain.z_unc.shape[1] + 1


def test_phase19_configs():
    base = ("couple", "fcs", "finish", "prior")
    assert MODEL_CONFIGS["Npx"]["extensions"] == (*base, "trans")
    assert MODEL_CONFIGS["Npc"]["extensions"] == (*base, "cont")
    assert MODEL_CONFIGS["Npxc"]["extensions"] == (*base, "trans", "cont")


def test_mov_diffs_use_only_earlier_fbs_games():
    from gnomepy_research.sessions.cfb_win_probability.models.win_probability import mov_diffs
    k = pd.Timestamp("2024-09-01", tz="UTC")
    games = pd.DataFrame({
        "game_id": [1, 2, 3], "season": 2024, "kickoff": [k, k + pd.Timedelta(days=7), k + pd.Timedelta(days=14)],
        "home_id": [1, 1, 2], "away_id": [2, 9, 1], "home_points": [30, 50, np.nan], "away_points": [10, 0, np.nan],
        "completed": [True, True, False], "home_classification": "fbs", "away_classification": ["fbs", "fcs", "fbs"]})
    # game 2: team 1 is +20 from game 1, the FCS side has no FBS games; game 3: team 2 lost its only FBS game by 20,
    # team 1 won its only FBS game by 20 (the FCS rout is ignored)
    np.testing.assert_allclose(mov_diffs(games, games), [0.0, 2.0, -4.0])


def test_phase20_configs():
    from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import OUTCOME_FLAGS
    assert {"mov", "mov_late"} <= set(OUTCOME_FLAGS)
    assert MODEL_CONFIGS["Npv"] == {**MODEL_CONFIGS["Np"], "mov": True}
    assert MODEL_CONFIGS["Npvl"]["mov_late"] and MODEL_CONFIGS["Npvl"]["mov"]


def test_every_extension_has_hyperparameter_fields_and_a_suffix():
    from gnomepy_research.models.cfb.efficiency.data import EXTENSIONS
    from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import EXTENSION_FIELDS
    from gnomepy_research.models.cfb.efficiency.posterior_snapshots import EXTENSION_SUFFIX
    assert set(EXTENSION_FIELDS) == set(EXTENSIONS) == set(EXTENSION_SUFFIX)


def test_mov_shrink_weights_by_games_played():
    from gnomepy_research.sessions.cfb_win_probability.models.win_probability import MOV_SHRINK_GAMES, mov_diffs
    k = pd.Timestamp("2024-09-01", tz="UTC")
    games = pd.DataFrame({
        "game_id": [1, 2, 3], "season": 2024, "kickoff": [k, k + pd.Timedelta(days=7), k + pd.Timedelta(days=14)],
        "home_id": [1, 1, 2], "away_id": [2, 9, 1], "home_points": [30, 50, np.nan], "away_points": [10, 0, np.nan],
        "completed": [True, True, False], "home_classification": "fbs", "away_classification": ["fbs", "fcs", "fbs"]})
    w = 1 / (1 + MOV_SHRINK_GAMES)                   # one FBS game each
    np.testing.assert_allclose(mov_diffs(games, games, shrink=MOV_SHRINK_GAMES), [0.0, 2.0 * w, -4.0 * w])


def test_p4_sides_and_tier_configs():
    from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import OUTCOME_FLAGS
    from gnomepy_research.sessions.cfb_win_probability.models.win_probability import p4_sides
    g = pd.DataFrame({"season": [2024, 2024, 2026, 2024, 2023],
                      "home_conference": ["SEC", "Sun Belt", "Pac-12", "FBS Independents", "Pac-12"],
                      "away_conference": ["Sun Belt", "SEC", "SEC", "Mid-American", "Mountain West"],
                      "home_team": ["A", "B", "Colorado State", "Notre Dame", "Utah"], "away_team": ["C", "D", "E", "F", "G"]})
    home, away = p4_sides(g)
    assert (home.astype(int) - away.astype(int)).tolist() == [1, -1, -1, 1, 1]
    assert {"mov_shrink", "tier"} <= set(OUTCOME_FLAGS)
    assert MODEL_CONFIGS["Npv2"] == {**MODEL_CONFIGS["Npv"], "mov_shrink": True}
    assert MODEL_CONFIGS["Npvs"] == {**MODEL_CONFIGS["Npv"], "tier": True}
    assert MODEL_CONFIGS["Npvg"]["extensions"] == ("couple", "fcs", "finish", "prior", "g5") and MODEL_CONFIGS["Npvg"]["mov"]
