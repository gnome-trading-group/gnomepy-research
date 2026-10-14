from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT, build_model_data
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import Hyperparameters, build_model, sample
from gnomepy_research.models.cfb.efficiency.gaussian_solver import _state_map, fit_cutoff_exact, solve
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name, hyper_name, model_version, snapshot_options
from gnomepy_research.models.cfb.features.ratings import FCS_TEAM
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import MODEL_CONFIGS, OUTCOME_FLAGS, resolve_config
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import (
    MARGIN_EDGE_TERMS, MARGIN_PRIORS, MARGIN_EDGE_SCALE, PHASE_TERMS, PRIORS, TERMS_SPLIT, game_draws)
from tests.sessions.test_cfb_win_probability_models import N_TEAMS, SEASON, conf_hypers, hypers, synthetic

INDEPENDENTS = "FBS Independents"
CONF = {t: (INDEPENDENTS if t in (1, 2) else "A" if t <= 5 else "B") for t in range(1, N_TEAMS + 1)}


def with_conferences(games: pd.DataFrame) -> pd.DataFrame:
    return games.assign(home_conference=games["home_id"].map(CONF), away_conference=games["away_id"].map(CONF))


class TestIndependentsSeparate:
    def test_independents_get_distinct_groups(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, with_conferences(games), [SEASON], conference=True, independents_separate=True)
        assert list(d.conf_keys["conference"]) == ["A", "B", f"{INDEPENDENTS}:1", f"{INDEPENDENTS}:2", "none"]
        names = d.conf_keys["conference"].to_numpy()[d.ts_conf]
        teams = d.ts_keys["team_id"].to_numpy()
        assert names[teams == 1][0] != names[teams == 2][0]
        assert names[teams == 1][0] == f"{INDEPENDENTS}:1"
        assert names[teams == 3][0] == "A" and names[teams == FCS_TEAM][0] == "none"
        assert d.independents_separate

    def test_default_unchanged(self):
        games, eff, pre, _, _ = synthetic()
        games = with_conferences(games)
        a = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, conference=True)
        b = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, conference=True, independents_separate=False)
        np.testing.assert_array_equal(a.x_off, b.x_off)
        np.testing.assert_array_equal(a.x_def, b.x_def)
        np.testing.assert_array_equal(a.ts_conf, b.ts_conf)
        pd.testing.assert_frame_equal(a.conf_keys, b.conf_keys)
        assert "FBS Independents:1" not in set(a.conf_keys["conference"]) and not a.independents_separate

    def test_pooled_by_default(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, with_conferences(games), [SEASON], conference=True)
        assert INDEPENDENTS in set(d.conf_keys["conference"])

    def test_conference_mean_is_own_value_for_independents(self):
        games, eff, pre, _, _ = synthetic()
        games = with_conferences(games)
        a = build_model_data(pre, eff, games, [SEASON], conference=True, independents_separate=True)
        idx = a.off_names.index("conf_talent_offense")
        col = a.x_off[:N_TEAMS, idx]
        raw = pre["talent_offense"].to_numpy().copy()
        raw[2:5] = raw[2:5].mean()
        raw[5:] = raw[5:].mean()
        np.testing.assert_allclose(col, (raw - raw.mean()) / raw.std(ddof=1))

    def test_hyperparameters_flag_dropped_when_off(self):
        games, eff, pre, _, _ = synthetic()
        h = hypers(build_model_data(pre, eff, games, [SEASON]))
        assert h.independents_separate is False and "independents_separate" not in h.to_dict()
        h.independents_separate = True
        assert Hyperparameters.from_dict(h.to_dict()).independents_separate is True


def test_naming_order_and_cli():
    name = hyper_name("normal", 2023, UNITS_SPLIT, conference=True, projection=True, drives=True,
                      independents_separate=True)
    assert name == "normal_split_conf_ind_proj_drv_2019_2022"
    assert draws_name(2024, UNITS_SPLIT, conference=True, independents_separate=True) == \
        "cfb_team_week_draws_split_conf_ind_2024"
    assert model_version(UNITS_SPLIT, conference=True, drives=True, independents_separate=True) == \
        "dyn-eff-split-conf-ind-drv-1"
    assert hyper_name("normal", 2023, UNITS_SPLIT, conference=True) == "normal_split_conf_2019_2022"
    assert snapshot_options(["2020", "2024", "--independents-separate"])["independents_separate"]
    assert not snapshot_options(["2020", "2024"])["independents_separate"]
    assert {"independents_separate", "phase"} <= set(OUTCOME_FLAGS)


class TestPhaseTerms:
    @staticmethod
    def write(tmp_path, ids=(1, 2, FCS_TEAM)):
        rng = np.random.default_rng(1)
        arrays = {}
        for key in ("regular_1", "regular_4", "regular_5", "postseason_1"):
            arrays |= {
                f"{key}__off": rng.normal(0, 0.1, (5, 6)).astype("float32"),
                f"{key}__def": rng.normal(0, 0.1, (5, 6)).astype("float32"),
                f"{key}__team_ids": np.array([*ids, *ids]),
                f"{key}__units": np.array(["pass"] * 3 + ["rush"] * 3),
                f"{key}__rush_share": np.array([0.4, 0.6, 0.5] * 2, dtype="float32"),
            }
        np.savez(tmp_path / f"{draws_name(SEASON, UNITS_SPLIT)}.npz", **arrays)

    @staticmethod
    def games(home=(1, 1, 1, 1), away=(2, 2, 2, 2)):
        return pd.DataFrame({
            "game_id": [1, 2, 3, 4], "season": SEASON, "week": [1, 4, 5, 1],
            "season_type": ["regular", "regular", "regular", "postseason"],
            "home_id": home, "away_id": away, "home_classification": "fbs", "away_classification": "fbs",
            "neutral_site": True, "completed": True, "home_points": 21, "away_points": 17,
        })

    def draws(self, tmp_path, games=None, phase=True):
        self.write(tmp_path)
        return game_draws(self.games() if games is None else games, [SEASON], directory=tmp_path,
                          units=UNITS_SPLIT, phase=phase)

    def test_terms_order_and_priors(self, tmp_path):
        gd = self.draws(tmp_path)
        assert PHASE_TERMS == ["pass_edge_early", "rush_edge_early"]
        assert gd.terms == TERMS_SPLIT + PHASE_TERMS
        assert all(PRIORS[t] == (0.0, 5.0) for t in PHASE_TERMS)
        assert gd.x.shape[-1] == 6

    def test_zero_outside_weeks_1_to_4(self, tmp_path):
        gd = self.draws(tmp_path)
        wk = gd.games.set_index("game_id")
        early = np.array([wk.loc[g, "season_type"] == "regular" and wk.loc[g, "week"] <= 4 for g in gd.games["game_id"]])
        assert early.sum() == 2
        for j, edge in ((4, 0), (5, 1)):
            np.testing.assert_allclose(gd.x[early][..., j], gd.x[early][..., edge])
            assert (gd.x[~early][..., j] == 0).all()
        assert np.abs(gd.x[early][..., 4]).sum() > 0

    def test_antisymmetric_under_swap(self, tmp_path):
        g = self.games()
        sw = self.games(home=(2, 2, 2, 2), away=(1, 1, 1, 1))
        a, b = self.draws(tmp_path, g), self.draws(tmp_path, sw)
        np.testing.assert_allclose(a.x[..., 4:], -b.x[..., 4:], rtol=1e-5, atol=1e-7)

    def test_default_has_no_phase_columns(self, tmp_path):
        gd = self.draws(tmp_path, phase=False)
        assert gd.terms == TERMS_SPLIT and gd.x.shape[-1] == 4

    def test_phase_requires_split_units(self, tmp_path):
        with pytest.raises(ValueError, match="split"):
            game_draws(self.games(), [SEASON], directory=tmp_path, phase=True)


def test_phase_terms_are_margin_edges():
    assert set(PHASE_TERMS) <= MARGIN_EDGE_TERMS
    for t in PHASE_TERMS:
        assert MARGIN_PRIORS[t] == (0.0, PRIORS[t][1] * MARGIN_EDGE_SCALE)


class TestIndependentsSolver:
    @staticmethod
    def data(seed=1):
        games, eff, pre, _, _ = synthetic(seed=seed)
        games = with_conferences(games)
        return games, eff, pre, build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, conference=True,
                                                 independents_separate=True)

    def test_solver_matches_mcmc(self):
        _, _, _, d = self.data()
        h = conf_hypers(d)
        h.alpha = np.array([0.15, 0.05])
        h.independents_separate = True
        mean, _, lay = solve(d, h)
        ts = np.arange(len(d.ts_keys))
        c_off, B_off = _state_map(d, h, lay, ts, np.full(len(ts), d.n_periods - 1), offense=True)
        exact = c_off + B_off @ mean
        idata = sample(build_model(d, hypers=h, likelihood="normal"), draws=1000, tune=500, chains=2)
        post = idata.posterior.to_dataset()
        np.testing.assert_allclose(exact, post["O"].mean(("chain", "draw")).values[:, -1], atol=0.01)

    def test_cutoff_fit_exact(self):
        games, eff, pre, d = self.data(seed=2)
        h = conf_hypers(d)
        h.independents_separate = True
        inputs = Inputs(preseason=pre, efficiency=eff, games=games, as_of=pd.Timestamp("2025-01-01", tz="UTC"))
        post, data = fit_cutoff_exact(inputs, SEASON, d.periods["cutoff"].iloc[-1], h, n_draws=50)
        assert data.independents_separate
        assert {f"{INDEPENDENTS}:1", f"{INDEPENDENTS}:2"} <= set(data.conf_keys["conference"])


def test_model_k_config(tmp_path):
    k = MODEL_CONFIGS["K"]
    g = MODEL_CONFIGS["G"]
    assert {kk: v for kk, v in k.items() if kk in g} == g
    assert k["drives"] and k["independents_separate"] and k["phase"]
    assert "keep_terms" not in k and "drop_features" not in k and "j_features" not in k
    assert resolve_config("K", tmp_path / "nope.json") == k


def test_game_draws_loads_k_draws(tmp_path):
    TestPhaseTerms.write(tmp_path)
    src = tmp_path / f"{draws_name(SEASON, UNITS_SPLIT)}.npz"
    name = draws_name(SEASON, UNITS_SPLIT, True, drives=True, independents_separate=True)
    assert name == f"cfb_team_week_draws_split_conf_ind_drv_{SEASON}"
    src.rename(tmp_path / f"{name}.npz")
    g = TestPhaseTerms.games()
    gd = game_draws(g, [SEASON], directory=tmp_path, units=UNITS_SPLIT, conference=True, drives=True,
                    independents_separate=True, phase=True)
    assert gd.terms == TERMS_SPLIT + PHASE_TERMS
    with pytest.raises(FileNotFoundError):
        game_draws(g, [SEASON], directory=tmp_path, units=UNITS_SPLIT, conference=True, drives=True, phase=True)
