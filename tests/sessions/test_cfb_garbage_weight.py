from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT, build_model_data
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import (
    draws_name, garbage_suffix, hyper_name, model_version, snapshot_options)
from gnomepy_research.models.cfb.features.drives import GARBAGE_WEIGHT, team_game_drives
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import MODEL_CONFIGS, OUTCOME_FLAGS
from gnomepy_research.sessions.cfb_win_probability.evaluation.garbage_weight import pick_weight, segment_log_loss
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import TERMS_SPLIT, game_draws
from tests.sessions import test_cfb_model_k as model_k
from tests.sessions.test_cfb_drives import drive, drive_hypers, hand_games
from tests.sessions.test_cfb_win_probability_models import SEASON, synthetic


def model_hypers():
    games, eff, pre, *_ = synthetic()
    return drive_hypers(build_model_data(pre, eff, games, [SEASON]))


def garbage_frame() -> pd.DataFrame:
    return pd.DataFrame([
        drive(1, 10, 20, True, 1, 0, 0, 7, "TD"),
        drive(1, 10, 20, True, 2, 7, 0, 3, "FG"),
        drive(1, 10, 20, True, 4, 40, 0, 7, "TD"),
        drive(1, 10, 20, True, 4, 47, 0, 0, "PUNT"),
    ])


class TestWeightApplied:
    @pytest.mark.parametrize("w", [0.3, 0.5, 1.0])
    def test_n_eff_and_ppd(self, w):
        home = team_game_drives(garbage_frame(), hand_games(), garbage_weight=w).set_index("team_id").loc[10]
        assert home["n_eff"] == pytest.approx(2 + 2 * w)
        assert home["ppd"] == pytest.approx((7 + 3 + w * 7) / (2 + 2 * w))

    def test_default_is_point_three(self):
        a = team_game_drives(garbage_frame(), hand_games())
        b = team_game_drives(garbage_frame(), hand_games(), garbage_weight=0.3)
        assert GARBAGE_WEIGHT == 0.3
        pd.testing.assert_frame_equal(a, b)

    def test_full_weight_is_raw_points_per_drive(self):
        home = team_game_drives(garbage_frame(), hand_games(), garbage_weight=1.0).set_index("team_id").loc[10]
        assert home["n_eff"] == 4 and home["ppd"] == pytest.approx(17 / 4)


class TestInputs:
    def inputs(self, w=0.3) -> Inputs:
        raw, games = garbage_frame(), hand_games()
        return Inputs(preseason=pd.DataFrame(), efficiency=pd.DataFrame(), games=games,
                      as_of=pd.Timestamp("2025-01-01", tz="UTC"), drives=team_game_drives(raw, games, w),
                      raw_drives=raw, garbage_weight=w)

    def test_cutoff_drives_follow_hypers_weight(self):
        cutoff = hand_games()["kickoff"].max() + pd.Timedelta(days=30)
        h = model_hypers()
        inp = self.inputs()
        assert inp.drives_for(h, cutoff)["n_eff"].iloc[0] == pytest.approx(2.6)
        assert inp.drives_for(replace(h, garbage_weight=1.0), cutoff)["n_eff"].iloc[0] == pytest.approx(4.0)
        assert inp.visible_drives(cutoff, garbage_weight=0.5)["n_eff"].iloc[0] == pytest.approx(3.0)

    def test_other_weight_without_raw_drives_raises(self):
        inp = self.inputs()
        inp.raw_drives = None
        with pytest.raises(ValueError, match="raw"):
            inp.visible_drives(pd.Timestamp("2030-01-01", tz="UTC"), garbage_weight=1.0)

    def test_loaded_weight_is_the_default(self):
        inp = self.inputs(1.0)
        cutoff = pd.Timestamp("2030-01-01", tz="UTC")
        assert inp.visible_drives(cutoff)["n_eff"].iloc[0] == pytest.approx(4.0)


class TestHyperparameters:
    def test_to_dict_omits_default_and_round_trips(self):
        h = model_hypers()
        assert h.garbage_weight == 0.3 and "garbage_weight" not in h.to_dict()
        d = replace(h, garbage_weight=1.0).to_dict()
        assert d["garbage_weight"] == 1.0
        assert type(h).from_dict(d).garbage_weight == 1.0
        assert type(h).from_dict(h.to_dict()).garbage_weight == 0.3


class TestNaming:
    def test_suffix_format(self):
        assert garbage_suffix(0.3) == ""
        assert garbage_suffix(0.5) == "_gw050"
        assert garbage_suffix(1.0) == "_gw100"
        assert garbage_suffix(0.05) == "_gw005"

    def test_defaults_unchanged(self):
        assert draws_name(2022, UNITS_SPLIT, True, drives=True, independents_separate=True) == \
            "cfb_team_week_draws_split_conf_ind_drv_2022"
        assert hyper_name("normal", 2022, UNITS_SPLIT, True, drives=True, independents_separate=True) == \
            "normal_split_conf_ind_drv_2019_2021"
        assert model_version(UNITS_SPLIT, True, drives=True) == "dyn-eff-split-conf-drv-1"

    def test_order_right_after_drv(self):
        assert draws_name(2022, UNITS_SPLIT, True, drives=True, independents_separate=True, tau_phases=True,
                          heavy_tails=True, garbage_weight=0.5) == "cfb_team_week_draws_split_conf_ind_drv_gw050_tph_ht_2022"
        name = hyper_name("normal", 2022, UNITS_SPLIT, True, projection=True, drop_features=("talent_qb",),
                          drives=True, garbage_weight=1.0)
        assert name.startswith("normal_split_conf_proj_drv_gw100_drop-") and name.endswith("_2019_2021")
        assert model_version(UNITS_SPLIT, True, drives=True, garbage_weight=1.0) == "dyn-eff-split-conf-drv-gw100-1"

    def test_weight_needs_drives(self):
        with pytest.raises(ValueError, match="drives"):
            draws_name(2022, UNITS_SPLIT, garbage_weight=1.0)

    def test_cli_and_flags(self, tmp_path):
        assert snapshot_options(["2020", "2022", "--drives", "--garbage-weight", "0.5"])["garbage_weight"] == 0.5
        assert snapshot_options(["2020", "2022"])["garbage_weight"] == 0.3
        assert "garbage_weight" in OUTCOME_FLAGS
        assert "garbage_weight" not in MODEL_CONFIGS["K"]
        model_k.TestPhaseTerms.write(tmp_path)
        src = tmp_path / f"{draws_name(SEASON, UNITS_SPLIT)}.npz"
        src.rename(tmp_path / f"{draws_name(SEASON, UNITS_SPLIT, drives=True, garbage_weight=1.0)}.npz")
        gd = game_draws(model_k.TestPhaseTerms.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT,
                        drives=True, garbage_weight=1.0)
        assert gd.terms == TERMS_SPLIT


class TestSelection:
    def test_picks_minimum(self):
        assert pick_weight({0.3: 0.552, 0.5: 0.549, 1.0: 0.551}) == 0.5
        assert pick_weight({0.3: 0.540, 0.5: 0.549, 1.0: 0.551}) == 0.3

    def test_segment_log_loss(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.2, 0.5, 0.5])
        p_open = np.array([0.85, 0.15, 0.55, np.nan])
        s = segment_log_loss(y, p, p_open)
        assert s["overall"] == pytest.approx(np.mean([-np.log(0.9), -np.log(0.8), np.log(2), np.log(2)]))
        assert s["big_favorites"] == pytest.approx(np.mean([-np.log(0.9), -np.log(0.8)]))
        assert s["toss_ups"] == pytest.approx(np.log(2))
        assert (s["n_overall"], s["n_big_favorites"], s["n_toss_ups"]) == (4, 2, 1)
