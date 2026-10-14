from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from gnomepy_research.sessions.cfb_win_probability.models.catboost_wp import TEAM_FEATURES, swap
from gnomepy_research.sessions.cfb_win_probability.models import residual_wp
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import GameDraws
from gnomepy_research.sessions.cfb_win_probability.models.residual_wp import (
    FEATURE_SETS, INTERACTIONS, LINEAR, SYMMETRIC, attach_baseline, engineer, fit_residual, sigmoid)

PARAMS = dict(depth=3, learning_rate=0.03, l2_leaf_reg=10)


def raw_frame(n, seed=0, venue=None):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"{s}_{f}": rng.normal(size=n) for s in ("h", "a") for f in TEAM_FEATURES})
    for f in TEAM_FEATURES:
        if f.endswith("_sd"):
            for s in ("h", "a"):
                df[f"{s}_{f}"] = np.abs(df[f"{s}_{f}"]) * 0.3
    for m, o, d in (("pass_matchup", "o_pass", "d_pass"), ("rush_matchup", "o_rush", "d_rush")):
        df[f"{m}_h"] = df[f"h_{o}"] - df[f"a_{d}"]
        df[f"{m}_a"] = df[f"a_{o}"] - df[f"h_{d}"]
    df["venue"] = rng.integers(0, 2, n) if venue is None else venue
    df["postseason"] = 0
    df["week"] = rng.integers(1, 14, n)
    df["season"] = np.where(np.arange(n) < 0.75 * n, 2020, 2021)
    df["game_id"] = np.arange(n)
    df["home_win"] = 0.0
    return df


def baseline_of(df):
    return 0.4 * (df["pass_matchup_h"] - df["pass_matchup_a"]).to_numpy() + 0.3 * df["venue"].to_numpy()


def draw_y(p, seed):
    return (np.random.default_rng(seed + 100).random(len(p)) < p).astype(float)


@pytest.mark.parametrize("col", LINEAR + INTERACTIONS)
def test_engineer_symmetry(col):
    raw = raw_frame(200)
    a, b = engineer(raw)[col].to_numpy(), engineer(swap(raw))[col].to_numpy()
    expected = a if col in SYMMETRIC else -a
    np.testing.assert_allclose(b, expected, atol=1e-12)


def test_feature_sets_nested():
    assert set(FEATURE_SETS["linear"]) < set(FEATURE_SETS["interactions"]) < set(FEATURE_SETS["full"])
    assert len(FEATURE_SETS["full"]) == len(set(FEATURE_SETS["full"]))


def _split(n, seed, y_from):
    df = raw_frame(n, seed)
    b = baseline_of(df)
    df["home_win"] = draw_y(sigmoid(y_from(df, b)), seed)
    train, test = df.iloc[: n * 2 // 3], df.iloc[n * 2 // 3:]
    return train, test, b[: n * 2 // 3], b[n * 2 // 3:]


def test_zero_signal_adds_no_noise():
    train, test, b_tr, b_te = _split(9000, 1, lambda df, b: b)
    model = fit_residual(train, b_tr, FEATURE_SETS["full"], PARAMS)
    ll_f = log_loss(test["home_win"], model.predict(test, b_te))
    ll_b = log_loss(test["home_win"], sigmoid(b_te))
    assert ll_f - ll_b < 0.005


def test_finds_planted_interaction():
    train, test, b_tr, b_te = _split(9000, 2, lambda df, b: b + 2 * engineer(df)["edge_x_unc"].to_numpy())
    model = fit_residual(train, b_tr, FEATURE_SETS["interactions"], PARAMS)
    ll_f = log_loss(test["home_win"], model.predict(test, b_te))
    ll_b = log_loss(test["home_win"], sigmoid(b_te))
    assert ll_b - ll_f > 0.01


def test_predict_antisymmetric_at_neutral_site():
    df = raw_frame(3000, 3, venue=0)
    b = baseline_of(df)
    df["home_win"] = draw_y(sigmoid(b + engineer(df)["edge_x_unc"].to_numpy()), 3)
    model = fit_residual(df, b, FEATURE_SETS["full"], PARAMS)
    test = raw_frame(300, 4, venue=0)
    bt = baseline_of(test)
    np.testing.assert_allclose(model.predict(test, bt) + model.predict(swap(test), -bt), 1.0, atol=1e-12)


def test_missing_baseline_raises():
    df = raw_frame(10)
    with pytest.raises(ValueError, match="no C baseline"):
        attach_baseline(df, pd.Series(np.zeros(9), index=np.arange(9)))


def test_cross_fitted_baseline_ignores_own_season_labels(monkeypatch):
    def stub_draws(games, seasons, n_draws=None, **kw):
        g = games[games["season"].isin(seasons)].reset_index(drop=True)
        return GameDraws(games=g, x=np.zeros((len(g), 1, 1)), terms=["t"])

    def stub_fit(gd, seed=0):
        p = gd.games["home_win"].mean()
        return SimpleNamespace(predict=lambda x: pd.DataFrame({"p_home": np.full(len(x), p)}))

    monkeypatch.setattr(residual_wp, "game_draws", stub_draws)
    monkeypatch.setattr(residual_wp, "fit", stub_fit)
    rng = np.random.default_rng(0)
    games = pd.DataFrame({"game_id": np.arange(300), "season": np.repeat([2020, 2021, 2022], 100),
                          "home_win": rng.integers(0, 2, 300).astype(float)})
    before = residual_wp.c_baseline(games, [2020, 2021], [2022])
    flipped = games.assign(home_win=np.where(games["season"] == 2020, 1 - games["home_win"], games["home_win"]))
    after = residual_wp.c_baseline(flipped, [2020, 2021], [2022])
    rows = games.loc[games["season"] == 2020, "game_id"]
    np.testing.assert_array_equal(before[rows], after[rows])
    assert not np.allclose(before[games.loc[games["season"] == 2021, "game_id"]],
                           after[games.loc[games["season"] == 2021, "game_id"]])
