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
