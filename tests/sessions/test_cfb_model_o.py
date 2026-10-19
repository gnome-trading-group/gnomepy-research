from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT, build_model_data, coach_features, group_of_five
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name, hyper_name
from gnomepy_research.models.cfb.features.qb_status import qb_changes
from gnomepy_research.sessions.cfb_win_probability.evaluation import o_selection
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import (
    MODEL_CONFIGS, OUTCOME_FLAGS, expected_changes, resolve_config)
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import (
    PRIORS, QB_CHANGE_TERM, RESULT_HFA, RESULT_SCALE, RESULTS_TERM, TERMS_SPLIT, game_draws, results_diffs)
from tests.sessions import test_cfb_model_k as model_k
from tests.sessions.test_cfb_win_probability_models import SEASON, synthetic


def test_qb_changes_compare_with_previous_game_only():
    k = pd.Timestamp("2024-09-01", tz="UTC")
    games = pd.DataFrame({"game_id": [1, 2, 3, 4], "season": 2024, "home_id": [10, 20, 10, 10], "away_id": [20, 10, 30, 20],
                          "kickoff": [k, k + pd.Timedelta(days=7), k + pd.Timedelta(days=14), k + pd.Timedelta(days=21)]})
    starters = pd.DataFrame({"game_id": [1, 1, 2, 2, 3, 3, 4], "team_id": [10, 20, 20, 10, 10, 30, 10],
                             "qb_id": ["a", "x", "x", "b", "b", "z", "b"]})
    out = qb_changes(starters, games).set_index("game_id")
    assert out.loc[1].tolist() == [0.0, 0.0]          # first games
    assert out.loc[2].tolist() == [0.0, 1.0]          # team 20 same QB (home), team 10 a -> b (away)
    assert out.loc[3].tolist() == [0.0, 0.0]          # team 10 b -> b; team 30 first game
    assert out.loc[4].tolist() == [0.0, 0.0]          # team 20 starter unknown -> no change
    z = expected_changes(out.reset_index(), games, [2024])
    assert (z[["home_qb_change", "away_qb_change"]] == 0).all().all()


def test_results_diffs_use_only_earlier_periods():
    out = pd.DataFrame({"season": [2024, 2024, 2024], "home_id": [1, 2, 1], "away_id": [2, 3, 3]})
    periods = np.array([0, 1, 2])
    surprises = pd.DataFrame({"season": 2024, "period": [0, 1], "home_id": [1, 2], "away_id": [2, 3], "s": [0.4, -0.3]})
    rd = results_diffs(out, periods, surprises)
    # game 0: nothing earlier; game 1 (2 v 3): team 2 had -0.4 from period 0, team 3 none -> -0.4;
    # game 2 (1 v 3): team 1 +0.4, team 3 +0.3 (from period 1, as away) -> 0.1
    np.testing.assert_allclose(rd, [0.0, -0.4, 0.1])


class TestResultsTerm:
    @staticmethod
    def games():
        return pd.DataFrame({
            "game_id": [1, 2, 3], "season": SEASON, "week": [1, 4, 5], "season_type": "regular",
            "home_id": [1, 2, 1], "away_id": [2, 1, 2], "home_classification": "fbs", "away_classification": "fbs",
            "neutral_site": [False, False, True], "completed": True, "home_points": [21, 30, 10], "away_points": [17, 10, 7],
            "kickoff": pd.to_datetime(["2024-08-31", "2024-09-21", "2024-09-28"], utc=True),
        })

    def test_term_appended_with_prior_and_value(self, tmp_path):
        model_k.TestPhaseTerms.write(tmp_path)
        gd = game_draws(self.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT, results=True)
        assert gd.terms == TERMS_SPLIT + [RESULTS_TERM] and PRIORS[RESULTS_TERM] == (0.0, 0.3)
        x = gd.games.set_index("game_id").index
        first = gd.x[list(x).index(1), :, -1]
        assert np.allclose(first, 0.0)                       # no earlier games
        assert np.allclose(gd.x[list(x).index(2), :, -1], gd.x[list(x).index(2), 0, -1])  # constant across draws
        assert gd.x[list(x).index(3), 0, -1] != 0.0         # both teams have earlier results by week 5
        assert RESULT_SCALE > 0 and 0 <= RESULT_HFA < 1

    def test_qb_change_term(self, tmp_path):
        model_k.TestPhaseTerms.write(tmp_path)
        ch = pd.DataFrame({"game_id": [1, 2, 3], "home_qb_change": [0.0, 1.0, 0.0], "away_qb_change": [0.0, 0.0, 1.0]})
        gd = game_draws(self.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT, qb_change=ch)
        assert gd.terms[-1] == QB_CHANGE_TERM and PRIORS[QB_CHANGE_TERM] == (0.0, 0.5)
        got = dict(zip(gd.games["game_id"], gd.x[:, 0, -1]))
        assert got == {1: 0.0, 2: 1.0, 3: -1.0}


class TestPriorExtensions:
    def test_group_of_five_flag(self):
        games = pd.DataFrame({"season": 2024, "home_id": [1, 2, 3], "away_id": [4, 5, 6],
                              "home_team": ["A", "Notre Dame", "C"], "away_team": ["D", "E", "F"],
                              "home_conference": ["SEC", "FBS Independents", "Sun Belt"],
                              "away_conference": ["SEC", "Mid-American", "Sun Belt"]})
        pre = pd.DataFrame({"season": 2024, "team_id": [1, 2, 3, 5]})
        np.testing.assert_array_equal(group_of_five(pre, games, [2024]), [0.0, 0.0, 1.0, 1.0])

    def test_pac12_is_power_four_only_through_2023(self):
        games = pd.DataFrame({"season": [2023, 2024], "home_id": 1, "away_id": 2, "home_team": "Washington State",
                              "away_team": "B", "home_conference": "Pac-12", "away_conference": "Mountain West"})
        pre = pd.DataFrame({"season": [2023, 2024], "team_id": 1})
        np.testing.assert_array_equal(group_of_five(pre, games, [2023, 2024]), [0.0, 1.0])

    def test_coach_features(self):
        pre = pd.DataFrame({"coach_tenure": [0, 1, 2, np.nan], "coach_new": [True, False, False, True],
                            "portal_in_stars": [10.0, 5.0, 3.0, np.nan]})
        out = coach_features(pre)
        assert out["coach_year2"].tolist() == [0.0, 1.0, 0.0, 0.0]
        assert out["coach_new_x_portal"].tolist()[:3] == [10.0, 0.0, 0.0] and np.isnan(out["coach_new_x_portal"].iloc[3])

    def test_model_data_adds_flags_and_features(self):
        games, eff, pre, _, _ = synthetic()
        pre = pre.assign(coach_tenure=1, portal_in_stars=2.0)
        games = games.assign(home_team="T", away_team="U", home_conference="Sun Belt", away_conference="Sun Belt")
        plain = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
        ext = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, extensions=("g5", "coach"))
        assert ext.z_unc.shape[1] == plain.z_unc.shape[1] + 2
        assert "pass:coach_year2" in ext.off_names and "rush:coach_new_x_portal" in ext.def_names
        assert (ext.z_unc[:, -2] == 1).all()  # every synthetic team is Group of 5 (FCS rows are 1 too)


def test_naming_configs_and_ladder(tmp_path):
    kw = dict(units=UNITS_SPLIT, conference=True, drives=True, independents_separate=True)
    assert draws_name(2024, **kw, extensions=("couple", "fcs", "finish", "prior", "g5", "coach")) == \
        "cfb_team_week_draws_split_conf_ind_drv_cpl_fcs_fin_pfx_g5w_cch_2024"
    assert hyper_name("normal", 2023, **kw, extensions=("couple", "fcs", "finish", "prior", "g5")).endswith("_pfx_g5w_2019_2022")
    assert "results" in OUTCOME_FLAGS and "qb_change" not in OUTCOME_FLAGS
    assert MODEL_CONFIGS["O2"] == {**MODEL_CONFIGS["Np"], "results": True, "qb_change": True}
    assert MODEL_CONFIGS["O4"]["extensions"] == ("couple", "fcs", "finish", "prior", "g5", "coach")
    np_cfg = dict(MODEL_CONFIGS["Np"])
    assert o_selection.add_step(np_cfg, "coach")["extensions"] == ("couple", "fcs", "finish", "prior", "coach")
    assert o_selection.label(MODEL_CONFIGS["O4"]) == "Np+results+qb_change+g5+coach"
    sel = tmp_path / "O.json"
    sel.write_text(json.dumps({"config": o_selection.serializable(MODEL_CONFIGS["O3"])}))
    cfg = resolve_config("O", o_selection=sel)
    assert cfg == MODEL_CONFIGS["O3"]
    with pytest.raises(SystemExit):
        resolve_config("O", o_selection=tmp_path / "missing.json")
