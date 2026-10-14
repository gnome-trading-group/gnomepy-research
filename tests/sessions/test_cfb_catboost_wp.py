import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name
from gnomepy_research.models.cfb.features.ratings import FCS_TEAM
from gnomepy_research.models.cfb.features.team_form import FORM_FEATURES
from gnomepy_research.sessions.cfb_win_probability.models.catboost_wp import (
    TEAM_FEATURES, fit_catboost, game_features, swap)

MATCHUPS = ["pass_matchup", "rush_matchup"]


def synthetic(n, seed=0, venue=0):
    rng = np.random.default_rng(seed)
    cols = {}
    for side in ("h", "a"):
        for f in TEAM_FEATURES:
            cols[f"{side}_{f}"] = rng.normal(size=n)
    df = pd.DataFrame(cols)
    for m, o, d in (("pass_matchup", "o_pass", "d_pass"), ("rush_matchup", "o_rush", "d_rush")):
        df[f"{m}_h"] = df[f"h_{o}"] - df[f"a_{d}"]
        df[f"{m}_a"] = df[f"a_{o}"] - df[f"h_{d}"]
    df["venue"] = venue
    df["postseason"] = 0
    df["week"] = rng.integers(1, 14, n)
    p = 1 / (1 + np.exp(-10 * (df["pass_matchup_h"] - df["pass_matchup_a"])))
    df["home_win"] = (rng.random(n) < p).astype(float)
    return df


def test_swap_is_an_involution_and_flips_label():
    df = synthetic(20, venue=1)
    s = swap(df)
    pd.testing.assert_frame_equal(swap(s), df)
    np.testing.assert_array_equal(s["home_win"], 1 - df["home_win"])
    np.testing.assert_array_equal(s["h_o_pass"], df["a_o_pass"])
    np.testing.assert_array_equal(s["pass_matchup_h"], df["pass_matchup_a"])
    np.testing.assert_array_equal(s["venue"], -df["venue"])
    assert (swap(synthetic(5, venue=0))["venue"] == 0).all()


def test_predict_is_antisymmetric_at_neutral_site():
    df = synthetic(300, venue=0)
    model = fit_catboost(df)
    test = synthetic(50, seed=5, venue=0)
    np.testing.assert_allclose(model.predict(test) + model.predict(swap(test)), 1.0, atol=1e-9)


def test_learns_obvious_signal():
    train, test = synthetic(2000, seed=1), synthetic(1000, seed=2)
    p = fit_catboost(train).predict(test)
    assert log_loss(test["home_win"], p) < 0.6
    assert roc_auc_score(test["home_win"], p) > 0.75


def test_feature_importance_covers_features():
    model = fit_catboost(synthetic(300))
    imp = model.feature_importance()
    assert "h_o_pass" in imp.index and imp.index.is_unique


def test_game_features_from_npz(tmp_path):
    rng = np.random.default_rng(1)
    arrays = {
        "regular_1__off": rng.normal(0, 0.1, (5, 6)).astype("float32"),
        "regular_1__def": rng.normal(0, 0.1, (5, 6)).astype("float32"),
        "regular_1__team_ids": np.array([1, 2, FCS_TEAM] * 2),
        "regular_1__units": np.array(["pass"] * 3 + ["rush"] * 3),
        "regular_1__rush_share": np.array([0.4, 0.6, 0.5] * 2, dtype="float32"),
        "regular_1__form": np.random.default_rng(2).normal(size=(6, 5)).astype("float32"),
    }
    np.savez(tmp_path / f"{draws_name(2024, UNITS_SPLIT)}.npz", **arrays)
    games = pd.DataFrame({
        "game_id": [10, 11], "season": 2024, "week": 1, "season_type": "regular",
        "home_id": [1, 2], "away_id": [2, 99],
        "home_classification": "fbs", "away_classification": "fbs",
        "neutral_site": [False, True], "completed": [True, False],
        "home_points": [21, 14], "away_points": [17, 28],
    }, index=[7, 3])
    gf = game_features(games, [2024], directory=tmp_path)
    off = arrays["regular_1__off"]
    np.testing.assert_allclose(gf["h_o_pass"], off[:, [0, 1]].mean(0), rtol=1e-5)
    np.testing.assert_allclose(gf["h_o_pass_sd"], off[:, [0, 1]].std(0), rtol=1e-4)
    np.testing.assert_allclose(gf["a_o_rush"], off[:, [3 + 1, 3 + 2]].mean(0), rtol=1e-5)
    np.testing.assert_allclose(gf["h_early_success"], arrays["regular_1__form"][[0, 1], 0], rtol=1e-5)
    np.testing.assert_allclose(gf["h_rush_share"], arrays["regular_1__rush_share"][[0, 1]], rtol=1e-5)
    assert gf["venue"].tolist() == [1, 0]
    assert gf["home_win"].iloc[0] == 1 and np.isnan(gf["home_win"].iloc[1])
    np.testing.assert_allclose(gf["pass_matchup_h"], gf["h_o_pass"] - gf["a_d_pass"])
    assert set(f"h_{f}" for f in FORM_FEATURES) <= set(gf.columns)

    bad = dict(arrays, regular_1__team_ids=np.array([1, 2, FCS_TEAM, 2, 1, FCS_TEAM]))
    np.savez(tmp_path / f"{draws_name(2024, UNITS_SPLIT)}.npz", **bad)
    with pytest.raises(ValueError, match="not aligned"):
        game_features(games, [2024], directory=tmp_path)
