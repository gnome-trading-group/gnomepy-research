from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.features.ratings import FCS_TEAM
from gnomepy_research.models.cfb.efficiency.data import DEF_FEATURES, OFF_FEATURES, UNITS_SPLIT, build_model_data
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import (
    Hyperparameters,
    build_model,
    sample,
)
from gnomepy_research.models.cfb.efficiency.inference import HYPER_DIR, Inputs
from gnomepy_research.models.cfb.efficiency.gaussian_solver import _state_map, fit_cutoff_exact, solve
from gnomepy_research.models.cfb.features.rush_share import league_rush_share, rush_share
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import OutcomeModel

SEASON = 2024
N_TEAMS = 10
N_WEEKS = 12


def synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    rng_unit = np.random.default_rng(seed + 1000)
    teams = np.arange(1, N_TEAMS + 1)
    true_o = rng.normal(0, 0.12, N_TEAMS)
    true_d = rng.normal(0, 0.12, N_TEAMS)
    kick0 = pd.Timestamp("2024-08-31 16:00", tz="UTC")
    games, eff, gid = [], [], 0
    for w in range(1, N_WEEKS + 1):
        order = rng.permutation(teams)
        for a, b in zip(order[::2], order[1::2]):
            gid += 1
            kickoff = kick0 + pd.Timedelta(days=7 * (w - 1))
            games.append({"game_id": gid, "season": SEASON, "week": w, "season_type": "regular", "kickoff": kickoff,
                          "home_id": a, "away_id": b, "home_classification": "fbs", "away_classification": "fbs"})
            for off, de, venue in ((a, b, 1), (b, a, -1)):
                y = 0.1 + true_o[off - 1] - true_d[de - 1] + 0.015 * venue + rng.normal(0, 1.4 / np.sqrt(65))
                p_pass = 0.15 + true_o[off - 1] - true_d[de - 1] + 0.015 * venue + rng_unit.normal(0, 1.6 / np.sqrt(30))
                p_rush = (65 * y - 30 * p_pass) / 35
                eff.append({"game_id": gid, "season": SEASON, "week": w, "season_type": "regular", "team_id": off,
                            "opponent_id": de, "team_classification": "fbs", "opponent_classification": "fbs",
                            "venue": venue, "plays": 65, "off_ppa": y, "pass_plays": 30, "rush_plays": 35,
                            "off_pass_ppa": p_pass, "off_rush_ppa": p_rush, "usable": True,
                            "available_at": kickoff + pd.Timedelta(days=1)})
    # One game against an FCS opponent.
    games.append({"game_id": 999, "season": SEASON, "week": 1, "season_type": "regular", "kickoff": kick0,
                  "home_id": 1, "away_id": 5000, "home_classification": "fbs", "away_classification": "fcs"})
    eff.append({"game_id": 999, "season": SEASON, "week": 1, "season_type": "regular", "team_id": 1,
                "opponent_id": 5000, "team_classification": "fbs", "opponent_classification": "fcs", "venue": 1,
                "plays": 65, "off_ppa": 0.6, "pass_plays": 30, "rush_plays": 35,
                "off_pass_ppa": 0.7, "off_rush_ppa": 0.5, "usable": True, "available_at": kick0 + pd.Timedelta(days=1)})
    pre = pd.DataFrame({"season": SEASON, "team_id": teams})
    for col in set(OFF_FEATURES + DEF_FEATURES):
        pre[col] = rng.normal(size=N_TEAMS)
    pre["qb_returning_starter"] = True
    pre["coach_new"] = False
    return pd.DataFrame(games), pd.DataFrame(eff), pre, true_o, true_d


def hypers(data) -> Hyperparameters:
    return Hyperparameters(
        alpha=np.full(len(data.units), 0.1), sigma_alpha=0.02, hfa=0.015, beta_off=np.zeros(len(data.off_names)),
        beta_def=np.zeros(len(data.def_names)), gamma=np.zeros(data.z_unc.shape[1]), s_off=0.12, s_def=0.12,
        tau_off=0.005, tau_def=0.005, sigma_game=0.01, sigma_play=1.4, nu=30.0,
        off_names=data.off_names, def_names=data.def_names, likelihood="studentt",
        units=list(data.units),
    )


class TestModelData:
    def test_indices_periods_and_fcs_pooling(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, games, [SEASON])
        assert len(d.ts_keys) == N_TEAMS + 1
        assert d.ts_keys["team_id"].iloc[-1] == FCS_TEAM
        assert d.n_periods == N_WEEKS
        assert set(d.obs_period) == set(range(N_WEEKS))
        fcs_row = np.flatnonzero(d.obs_game == 999)
        assert d.obs_def[fcs_row[0]] == N_TEAMS
        assert d.x_off[-1, -1] == 1.0 and d.x_off[:-1, -1].sum() == 0.0

    def test_features_are_centred_within_season(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, games, [SEASON])
        assert np.allclose(d.x_off[:-1, : len(OFF_FEATURES)].mean(0), 0.0, atol=1e-12)

    def test_unusable_rows_are_excluded(self):
        games, eff, pre, _, _ = synthetic()
        eff.loc[0, "usable"] = False
        d = build_model_data(pre, eff, games, [SEASON])
        assert len(d.obs_y) == len(eff) - 1


class TestSplitModelData:
    def test_all_units_matches_default(self):
        games, eff, pre, _, _ = synthetic()
        a = build_model_data(pre, eff, games, [SEASON])
        b = build_model_data(pre, eff, games, [SEASON], units=("all",))
        np.testing.assert_array_equal(a.x_off, b.x_off)
        assert a.off_names == b.off_names
        assert list(a.alpha_keys["unit"]) == ["all"]
        np.testing.assert_array_equal(a.obs_alpha, np.zeros(len(a.obs_y), dtype=int))

    def test_split_doubles_states_and_observations(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
        assert len(d.ts_keys) == 2 * (N_TEAMS + 1)
        assert list(d.ts_keys["unit"].unique()) == ["pass", "rush"]
        assert len(d.obs_y) == 2 * len(eff)
        assert set(d.obs_unit) == {0, 1}
        assert (d.ts_keys["unit"].to_numpy()[d.obs_off[d.obs_unit == 0]] == "pass").all()
        assert d.x_off.shape[1] == 2 * (len(OFF_FEATURES) + 2)
        assert d.off_names[0].startswith("pass:") and d.off_names[-1] == "rush:is_fcs"

    def test_zero_play_unit_observation_is_dropped(self):
        games, eff, pre, _, _ = synthetic()
        eff.loc[0, "rush_plays"] = 0
        eff.loc[0, "off_rush_ppa"] = np.nan
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
        assert len(d.obs_y) == 2 * len(eff) - 1
        assert (d.obs_plays > 0).all() and np.isfinite(d.obs_y).all()


def test_recovers_rating_signs_and_order():
    games, eff, pre, true_o, true_d = synthetic()
    d = build_model_data(pre, eff, games, [SEASON])
    idata = sample(build_model(d, hypers=hypers(d)), draws=300, tune=300, chains=2)
    post = idata.posterior.to_dataset()
    o = post["O"].mean(("chain", "draw")).values[:N_TEAMS, -1]
    de = post["D"].mean(("chain", "draw")).values[:N_TEAMS, -1]
    # Higher O = better offense, higher D = better defense.
    assert np.corrcoef(o, true_o)[0, 1] > 0.8
    assert np.corrcoef(de, true_d)[0, 1] > 0.8


def test_exact_gaussian_solver_matches_mcmc():
    games, eff, pre, _, _ = synthetic(seed=1)
    d = build_model_data(pre, eff, games, [SEASON])
    h = hypers(d)
    h.likelihood = "normal"
    mean, _, lay = solve(d, h)
    ts = np.arange(len(d.ts_keys))
    c_off, B_off = _state_map(d, h, lay, ts, np.full(len(ts), d.n_periods - 1), offense=True)
    exact = c_off + B_off @ mean

    idata = sample(build_model(d, hypers=h, likelihood="normal"), draws=1000, tune=500, chains=2)
    mcmc = idata.posterior.to_dataset()["O"].mean(("chain", "draw")).values[:, -1]
    assert layout_size_matches(lay, d)
    np.testing.assert_allclose(exact, mcmc, atol=0.01)


def test_exact_solver_matches_mcmc_split():
    games, eff, pre, _, _ = synthetic(seed=1)
    d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
    h = hypers(d)
    h.likelihood = "normal"
    h.alpha = np.array([0.15, 0.05])
    mean, _, lay = solve(d, h)
    ts = np.arange(len(d.ts_keys))
    c_off, B_off = _state_map(d, h, lay, ts, np.full(len(ts), d.n_periods - 1), offense=True)
    exact = c_off + B_off @ mean
    idata = sample(build_model(d, hypers=h, likelihood="normal"), draws=1000, tune=500, chains=2)
    mcmc = idata.posterior.to_dataset()["O"].mean(("chain", "draw")).values[:, -1]
    assert layout_size_matches(lay, d)
    np.testing.assert_allclose(exact, mcmc, atol=0.01)


def layout_size_matches(lay: dict, d) -> bool:
    n_ts = len(d.ts_keys)
    return lay["size"] == len(d.alpha_keys) + 2 * n_ts + 2 * n_ts * (d.n_periods - 1)


class TestOutcomeModel:
    @staticmethod
    def model():
        rng = np.random.default_rng(0)
        return OutcomeModel(gamma=np.column_stack([rng.normal(8, 0.5, 200), rng.normal(7, 0.5, 200),
                                                   rng.normal(0.3, 0.05, 200), rng.normal(-1, 0.5, 200)]))

    def test_neutral_site_swap_gives_complement(self):
        rng = np.random.default_rng(1)
        od, dd = rng.normal(0, 0.1, (5, 40)), rng.normal(0, 0.1, (5, 40))
        x = np.stack([od, dd, np.zeros_like(od), np.zeros_like(od)], axis=-1)
        swapped = np.stack([-od, -dd, np.zeros_like(od), np.zeros_like(od)], axis=-1)
        m = self.model()
        np.testing.assert_allclose(m.predict(x)["p_home"] + m.predict(swapped)["p_home"], 1.0, atol=1e-12)

    def test_probabilities_valid_and_interval_brackets_mean(self):
        rng = np.random.default_rng(2)
        x = np.stack([rng.normal(0, 0.2, (50, 40)), rng.normal(0, 0.2, (50, 40)),
                      np.ones((50, 40)), np.zeros((50, 40))], axis=-1)
        p = self.model().predict(x)
        assert ((p["p_home"] > 0) & (p["p_home"] < 1)).all()
        assert (p["p_lo"] <= p["p_home"]).all() and (p["p_home"] <= p["p_hi"]).all()

    def test_home_field_raises_home_probability(self):
        z = np.zeros((1, 10, 4))
        home = z.copy()
        home[..., 2] = 1.0
        m = self.model()
        assert m.predict(z)["p_home"].iloc[0] == 0.5
        assert m.predict(home)["p_home"].iloc[0] > 0.5


def test_old_scalar_alpha_json_still_loads():
    d = json.loads((HYPER_DIR / "normal_2019_2024.json").read_text())["hyperparameters"]
    h = Hyperparameters.from_dict(d)
    assert h.alpha.shape == (1,) and h.units == ["all"]


def test_split_recovers_unit_baselines_and_ratings():
    games, eff, pre, true_o, true_d = synthetic()
    rng_t = np.random.default_rng(7)
    fbs = (eff["opponent_classification"] == "fbs").to_numpy()
    oi = (eff["team_id"].to_numpy() - 1).clip(0, N_TEAMS - 1)
    di = (eff["opponent_id"].to_numpy() - 1).clip(0, N_TEAMS - 1)
    base = np.where(fbs, true_o[oi] - true_d[di], 0.0) + 0.015 * eff["venue"].to_numpy()
    eff["off_pass_ppa"] = 0.15 + base + rng_t.normal(0, 0.03, len(eff))
    eff["off_rush_ppa"] = 0.05 + base + rng_t.normal(0, 0.03, len(eff))
    d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
    h = hypers(d)
    h.alpha = np.array([0.15, 0.05])
    idata = sample(build_model(d, hypers=h), draws=300, tune=300, chains=2)
    post = idata.posterior.to_dataset()
    o = post["O"].mean(("chain", "draw")).values[:, -1]
    n = N_TEAMS + 1
    assert np.corrcoef(o[:N_TEAMS], true_o)[0, 1] > 0.7
    assert np.corrcoef(o[n:n + N_TEAMS], true_o)[0, 1] > 0.7
    a = post["alpha_season"].mean(("chain", "draw")).values
    assert a[0] > a[1]


def test_rush_share_is_shrunk_toward_league():
    e = pd.DataFrame({"season": 2024, "team_id": [1, 1, 2], "rush_plays": [50, 50, 10], "pass_plays": [10, 10, 50]})
    s = rush_share(e, 2024, shrink_plays=150)
    lg = league_rush_share(e, 2024)
    assert lg == pytest.approx(110 / 180)
    assert lg < s[1] < 100 / 120
    assert 10 / 60 < s[2] < lg


def test_rush_share_with_no_rows_is_empty_and_league_half():
    e = pd.DataFrame(columns=["season", "team_id", "rush_plays", "pass_plays"])
    assert rush_share(e, 2024).empty and league_rush_share(e, 2024) == 0.5


def test_split_cutoff_posterior_shapes():
    games, eff, pre, _, _ = synthetic(seed=2)
    inputs = Inputs(preseason=pre, efficiency=eff, games=games, as_of=pd.Timestamp("2025-01-01", tz="UTC"))
    d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
    h = hypers(d)
    h.likelihood = "normal"
    h.alpha = np.array([0.15, 0.05])
    cutoff = d.periods["cutoff"].iloc[-1]
    post, _ = fit_cutoff_exact(inputs, SEASON, cutoff, h, n_draws=200)
    assert post.off.shape == (200, 2 * (N_TEAMS + 1))
    assert list(pd.unique(post.units)) == ["pass", "rush"]
    assert post.alpha.shape == (200, 2)
    s = post.summary()
    assert set(s["unit"]) == {"pass", "rush"} and len(s) == 2 * (N_TEAMS + 1)
