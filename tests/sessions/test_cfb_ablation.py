from __future__ import annotations

import json

import numpy as np
import pytest

from gnomepy_research.models.cfb.features.ratings import FCS_TEAM
from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT, build_model_data
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import Hyperparameters
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name, drop_suffix, hyper_name
from gnomepy_research.sessions.cfb_win_probability.evaluation.ablation import (
    GROUPS, backward_eliminate, expand_groups, pick_removal)
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import FORM_TERMS, TERMS_SPLIT, game_draws
from tests.sessions import test_cfb_win_probability_models as wpm


class TestKeepTerms:
    def test_selects_and_reorders(self, tmp_path):
        wpm.TestSplitGameDraws.write(tmp_path, [1, 2, FCS_TEAM], form=True)
        g = wpm.TestSplitGameDraws.games()
        full = game_draws(g, [2024], directory=tmp_path, units=UNITS_SPLIT, form=True)
        sub = game_draws(g, [2024], directory=tmp_path, units=UNITS_SPLIT, form=True,
                         keep_terms=["venue", "pass_edge", FORM_TERMS[2]])
        assert sub.terms == ["venue", "pass_edge", FORM_TERMS[2]]
        np.testing.assert_array_equal(sub.x, full.x[..., [2, 0, 6]])

    def test_none_is_unchanged(self, tmp_path):
        wpm.TestSplitGameDraws.write(tmp_path, [1, 2, FCS_TEAM])
        g = wpm.TestSplitGameDraws.games()
        a = game_draws(g, [2024], directory=tmp_path, units=UNITS_SPLIT)
        b = game_draws(g, [2024], directory=tmp_path, units=UNITS_SPLIT, keep_terms=None)
        assert a.terms == b.terms == TERMS_SPLIT
        np.testing.assert_array_equal(a.x, b.x)

    def test_unknown_term_raises(self, tmp_path):
        wpm.TestSplitGameDraws.write(tmp_path, [1, 2, FCS_TEAM])
        with pytest.raises(ValueError, match="qb_delta_diff"):
            game_draws(wpm.TestSplitGameDraws.games(), [2024], directory=tmp_path, units=UNITS_SPLIT,
                       keep_terms=["pass_edge", "qb_delta_diff"])


class TestDropFeatures:
    def test_default_unchanged(self):
        games, eff, pre, _, _ = wpm.synthetic()
        g = wpm.with_conferences(games)
        a = build_model_data(pre, eff, g, [wpm.SEASON], units=UNITS_SPLIT, conference=True)
        b = build_model_data(pre, eff, g, [wpm.SEASON], units=UNITS_SPLIT, conference=True, drop_features=())
        assert a.off_names == b.off_names and a.def_names == b.def_names
        np.testing.assert_array_equal(a.x_off, b.x_off)
        np.testing.assert_array_equal(a.x_def, b.x_def)
        assert a.drop_features == ()

    def test_removes_base_and_conference_features(self):
        games, eff, pre, _, _ = wpm.synthetic()
        g = wpm.with_conferences(games)
        drop = ("coach_new", "coach_srs_resid", "conf_recruiting")
        full = build_model_data(pre, eff, g, [wpm.SEASON], conference=True)
        d = build_model_data(pre, eff, g, [wpm.SEASON], conference=True, drop_features=drop)
        assert not set(drop) & set(d.off_names + d.def_names)
        assert set(full.off_names) - set(d.off_names) == set(drop)
        assert d.x_off.shape[1] == full.x_off.shape[1] - 3 and d.x_def.shape[1] == full.x_def.shape[1] - 3
        keep = [full.off_names.index(n) for n in d.off_names]
        np.testing.assert_array_equal(d.x_off, full.x_off[:, keep])
        np.testing.assert_array_equal(d.z_unc, full.z_unc)

    def test_unknown_feature_raises(self):
        games, eff, pre, _, _ = wpm.synthetic()
        with pytest.raises(ValueError, match="conf_prior_off"):
            build_model_data(pre, eff, games, [wpm.SEASON], drop_features=("conf_prior_off",))

    def test_hyperparameters_record_drops_and_default_json_unchanged(self):
        games, eff, pre, _, _ = wpm.synthetic()
        d = build_model_data(pre, eff, games, [wpm.SEASON], drop_features=("portal_in_stars",))
        h = wpm.hypers(d)
        h.drop_features = list(d.drop_features)
        back = Hyperparameters.from_dict(json.loads(json.dumps(h.to_dict())))
        assert back.drop_features == ["portal_in_stars"]
        assert "drop_features" not in wpm.hypers(build_model_data(pre, eff, games, [wpm.SEASON])).to_dict()


class TestSuffix:
    def test_deterministic_and_order_free(self):
        a = drop_suffix(("coach_new", "coach_srs_resid"))
        assert a == drop_suffix(("coach_srs_resid", "coach_new"))
        assert a.startswith("_drop-") and len(a) == len("_drop-") + 8
        assert a != drop_suffix(("coach_new",))
        assert drop_suffix(()) == ""

    def test_names(self):
        s = drop_suffix(("portal_in_stars",))
        assert hyper_name("normal", 2022, UNITS_SPLIT, True, drop_features=("portal_in_stars",)) == \
            f"normal_split_conf{s}_2019_2021"
        assert draws_name(2022, UNITS_SPLIT, True, drop_features=("portal_in_stars",)) == \
            f"cfb_team_week_draws_split_conf{s}_2022"
        assert hyper_name("normal", 2022, UNITS_SPLIT, True) == "normal_split_conf_2019_2021"


class TestElimination:
    def test_pick_removal(self):
        assert pick_removal(0.60, {"a": 0.6004, "b": 0.5990, "c": 0.62}) == "b"
        assert pick_removal(0.60, {"a": 0.6004, "c": 0.62}) == "a"
        assert pick_removal(0.60, {"a": 0.6006, "c": 0.62}) is None
        assert pick_removal(0.60, {}) is None

    def test_backward_eliminate_on_score_table(self):
        table = {
            frozenset("abc"): 0.600,
            frozenset("bc"): 0.5995, frozenset("ac"): 0.6003, frozenset("ab"): 0.610,
            frozenset("c"): 0.6020, frozenset("b"): 0.6150,
        }
        kept, steps = backward_eliminate(["a", "b", "c"], lambda k: table[frozenset(k)])
        assert kept == ["b", "c"]
        dropped = [s for s in steps if s["dropped"]]
        assert [s["candidate"] for s in dropped] == ["a"]
        assert dropped[0]["delta"] == pytest.approx(-0.0005)
        assert {s["candidate"] for s in steps if s["round"] == 2} == {"b", "c"}

    def test_min_items(self):
        kept, _ = backward_eliminate(["a", "b"], lambda k: 0.6, min_items=1)
        assert len(kept) == 1
        kept, _ = backward_eliminate(["a", "b"], lambda k: 0.6, min_items=0)
        assert kept == []

    def test_groups_expand(self):
        assert set(expand_groups(["qb", "portal"])) == {"qb_prior_ppa", "qb_returning_starter", "portal_in_stars"}
        assert {"talent_offense", "talent_defense", "talent_qb", "recruiting_points"} == set(GROUPS["recruiting_talent"])
        assert expand_groups(["coach", "qb"]) == expand_groups(["qb", "coach"])
