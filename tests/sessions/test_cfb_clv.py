from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.evaluation.clv import clv_frame, clv_regression
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import PAIRS
from gnomepy_research.sessions.cfb_win_probability.evaluation.g_report import CUT_MODELS, CUT_PAIRS, cut_sides


def test_regression_recovers_planted_slope():
    rng = np.random.default_rng(0)
    p_open = rng.uniform(0.2, 0.8, 2000)
    p_model = np.clip(p_open + rng.normal(0, 0.08, 2000), 0.01, 0.99)
    p_close = p_open + 0.01 + 0.4 * (p_model - p_open) + rng.normal(0, 0.02, 2000)
    r = clv_regression(p_model, p_open, p_close, n_boot=200)
    assert abs(r["slope"] - 0.4) < 0.03 and abs(r["intercept"] - 0.01) < 0.003
    assert r["lo95"] < r["slope"] < r["hi95"] and r["hi95"] - r["lo95"] < 0.1
    assert 0.8 < r["corr"] < 1 and r["n"] == 2000


def test_regression_zero_slope_ci_covers_zero():
    rng = np.random.default_rng(1)
    p_open = rng.uniform(0.2, 0.8, 1500)
    p_model = p_open + rng.normal(0, 0.05, 1500)
    p_close = p_open + rng.normal(0, 0.02, 1500)
    r = clv_regression(p_model, p_open, p_close, n_boot=200)
    assert r["lo95"] < 0 < r["hi95"] and abs(r["corr"]) < 0.1


def test_clv_frame_filters_and_aligns():
    preds = pd.DataFrame({"game_id": [1, 2, 3, 4], "p_home": [0.6, 0.5, 0.9, 0.3]})
    market = pd.DataFrame({"game_id": [1, 2, 3, 5], "p_open": [0.5, np.nan, 0.5, 0.5], "p_close": [0.55, 0.5, np.nan, 0.5]})
    f = clv_frame(preds, market)
    assert f["game_id"].tolist() == [1]


def test_k_report_pairs_and_cuts():
    assert {("K", "G"), ("K", "J")} <= set(PAIRS)
    assert CUT_MODELS["K"] == ["G", "J", "K"]
    assert CUT_PAIRS["K"] == [("K", "G"), ("K", "J")]


def test_independents_cut_orients_to_independent():
    ind = "FBS Independents"
    g = pd.DataFrame({
        "home_conference": [ind, "SEC", ind, "SEC"], "away_conference": ["SEC", ind, ind, "ACC"],
        "home_team": ["Notre Dame", "Georgia", "UConn", "Georgia"], "away_team": ["Clemson", "UMass", "UMass", "Clemson"],
        "season_type": "regular", "week": 3, "p_open": 0.5, "conference_game": False,
    })
    mask, flip = cut_sides(g)["FBS independents"]
    assert mask.tolist() == [True, True, True, False]
    assert flip[mask].tolist() == [False, True, False]
