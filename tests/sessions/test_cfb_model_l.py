from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.efficiency.data import UNITS_ALL, UNITS_SPLIT, build_model_data, period_phase, step_phases
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import (
    TAU_PHASE_PRIORS, Hyperparameters, build_model, hyperparameters_from, sample)
from gnomepy_research.models.cfb.efficiency.gaussian_solver import _state_map, fit_cutoff_exact, solve
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name, hyper_name, model_version, snapshot_options
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import OUTCOME_FLAGS
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import PRIORS, TERMS_SPLIT, game_draws
from tests.sessions.test_cfb_model_k import TestPhaseTerms
from tests.sessions.test_cfb_win_probability_models import SEASON, hypers, synthetic

PHASED_OFF = np.array([0.02, 0.01, 0.005, 0.04])
PHASED_DEF = np.array([0.015, 0.008, 0.004, 0.03])


def phased(h: Hyperparameters, off=PHASED_OFF, de=PHASED_DEF) -> Hyperparameters:
    h.tau_phases, h.tau_off, h.tau_def = True, np.asarray(off, float), np.asarray(de, float)
    return h


def with_bowl(games: pd.DataFrame, eff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append a postseason week replaying week 12's pairings two weeks later."""
    g12, e12 = games[games["week"] == 12], eff[eff["week"] == 12]
    shift = pd.Timedelta(days=14)
    gid = {g: g + 10_000 for g in g12["game_id"]}
    bg = g12.assign(season_type="postseason", week=1, kickoff=g12["kickoff"] + shift, game_id=g12["game_id"].map(gid))
    be = e12.assign(season_type="postseason", week=1, available_at=e12["available_at"] + shift,
                    game_id=e12["game_id"].map(gid))
    return pd.concat([games, bg], ignore_index=True), pd.concat([eff, be], ignore_index=True)


def bowl_data(seed=1, units=UNITS_ALL):
    games, eff, pre, o, de = synthetic(seed=seed)
    games, eff = with_bowl(games, eff)
    return games, eff, pre, build_model_data(pre, eff, games, [SEASON], units=units)


class TestPhaseAssignment:
    def test_period_phase(self):
        st = np.array(["regular"] * 7 + ["postseason"])
        wk = np.array([0, 1, 4, 5, 9, 10, 15, 1])
        assert period_phase(st, wk).tolist() == [0, 0, 0, 1, 1, 2, 2, 3]

    def test_step_phase_uses_destination_period_and_own_season_calendar(self):
        periods = pd.DataFrame({
            "season": [2020] * 4 + [2021] * 6,
            "period": [0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
            "season_type": ["regular", "regular", "regular", "postseason"] + ["regular"] * 5 + ["postseason"],
            "week": [4, 5, 10, 1, 1, 4, 5, 9, 10, 1],
        })
        sp = step_phases(periods, np.array([2020, 2021, 2020]), 6)
        assert sp.shape == (3, 5)
        assert sp[0].tolist() == [1, 2, 3, 3, 3]
        assert sp[1].tolist() == [0, 1, 1, 2, 3]
        assert sp[2].tolist() == sp[0].tolist()

    def test_synthetic_data_steps(self):
        _, _, _, d = bowl_data()
        sp = step_phases(d.periods, d.ts_keys["season"].to_numpy(), d.n_periods)
        assert sp.shape == (len(d.ts_keys), d.n_periods - 1)
        assert sp[0].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3]


class TestDefaultUnchanged:
    def test_hyperparameters_default_dict(self):
        games, eff, pre, _, _ = synthetic()
        h = hypers(build_model_data(pre, eff, games, [SEASON]))
        assert h.tau_phases is False and "tau_phases" not in h.to_dict()
        assert isinstance(h.to_dict()["tau_off"], float)

    def test_phased_round_trip(self):
        games, eff, pre, _, _ = synthetic()
        h = phased(hypers(build_model_data(pre, eff, games, [SEASON])))
        d = h.to_dict()
        assert d["tau_phases"] is True and d["tau_off"] == PHASED_OFF.tolist()
        back = Hyperparameters.from_dict(d)
        assert back.tau_phases and isinstance(back.tau_off, np.ndarray)
        np.testing.assert_array_equal(back.tau_def, PHASED_DEF)

    def test_free_model_default_has_scalar_tau(self):
        games, eff, pre, _, _ = synthetic()
        m = build_model(build_model_data(pre, eff, games, [SEASON]))
        assert "tau_phase" not in m.coords
        assert m["tau_off"].eval().shape == ()

    def test_free_model_phased_has_vector_tau_with_priors(self):
        games, eff, pre, _, _ = synthetic()
        m = build_model(build_model_data(pre, eff, games, [SEASON]), tau_phases=True)
        assert list(m.coords["tau_phase"]) == ["early", "mid", "late", "post"]
        assert m["tau_off"].eval().shape == (4,)
        assert TAU_PHASE_PRIORS == (0.03, 0.03, 0.03, 0.06)

    @pytest.mark.parametrize("units", [UNITS_ALL, UNITS_SPLIT])
    def test_equal_phase_taus_match_scalar_solver(self, units):
        _, _, _, d = bowl_data(units=units)
        h0 = hypers(d)
        h0.likelihood = "normal"
        h1 = phased(hypers(d), np.full(4, h0.tau_off), np.full(4, h0.tau_def))
        h1.likelihood = "normal"
        m0, _, lay = solve(d, h0)
        m1, _, _ = solve(d, h1)
        np.testing.assert_allclose(m0, m1, rtol=1e-10, atol=1e-12)
        ts = np.arange(len(d.ts_keys))
        k = np.arange(len(ts)) % d.n_periods
        _, B0 = _state_map(d, h0, lay, ts, k, offense=True)
        _, B1 = _state_map(d, h1, lay, ts, k, offense=True)
        np.testing.assert_array_equal(B0, B1)


def test_state_map_puts_step_tau_per_column():
    _, _, _, d = bowl_data()
    h = phased(hypers(d))
    mean, _, lay = solve(d, h)
    n_s = d.n_periods - 1
    _, B = _state_map(d, h, lay, np.array([2]), np.array([n_s]), offense=True)
    row = B[0, lay["eps_off"] + 2 * n_s: lay["eps_off"] + 3 * n_s]
    sp = step_phases(d.periods, d.ts_keys["season"].to_numpy(), d.n_periods)[2]
    np.testing.assert_array_equal(row, PHASED_OFF[sp])
    _, B = _state_map(d, h, lay, np.array([2]), np.array([3]), offense=True)
    row = B[0, lay["eps_off"] + 2 * n_s: lay["eps_off"] + 3 * n_s]
    np.testing.assert_array_equal(row, np.r_[PHASED_OFF[sp[:3]], np.zeros(n_s - 3)])


def test_exact_solver_matches_mcmc_with_tau_phases():
    _, _, _, d = bowl_data()
    h = phased(hypers(d), PHASED_OFF * 3, PHASED_DEF * 3)
    h.likelihood = "normal"
    mean, _, lay = solve(d, h)
    ts = np.arange(len(d.ts_keys))
    c, B = _state_map(d, h, lay, ts, np.full(len(ts), d.n_periods - 1), offense=True)
    idata = sample(build_model(d, hypers=h, likelihood="normal"), draws=1000, tune=500, chains=2)
    post = idata.posterior.to_dataset()
    np.testing.assert_allclose(c + B @ mean, post["O"].mean(("chain", "draw")).values[:, -1], atol=0.01)


def test_cutoff_fit_exact_with_tau_phases():
    games, eff, pre, d = bowl_data(seed=2)
    h = phased(hypers(d))
    h.likelihood = "normal"
    inputs = Inputs(preseason=pre, efficiency=eff, games=games, as_of=pd.Timestamp("2025-06-01", tz="UTC"))
    post, data = fit_cutoff_exact(inputs, SEASON, d.periods["cutoff"].iloc[-1], h, n_draws=50)
    assert post.off.shape == (50, len(data.ts_keys)) and np.isfinite(post.off).all()


def drifting(seed=0, n_teams=40, early=0.08, late=0.002):
    """Ratings that move a lot over weeks 1-4 and barely after; 13 regular weeks."""
    rng = np.random.default_rng(seed)
    teams = np.arange(1, n_teams + 1)
    o = rng.normal(0, 0.1, n_teams)
    de = rng.normal(0, 0.1, n_teams)
    kick0 = pd.Timestamp("2024-08-31 16:00", tz="UTC")
    games, eff, gid = [], [], 0
    for w in range(1, 14):
        tau = early if w <= 5 else late
        if w > 1:
            o = o + rng.normal(0, tau, n_teams)
            de = de + rng.normal(0, tau, n_teams)
        order = rng.permutation(teams)
        kickoff = kick0 + pd.Timedelta(days=7 * (w - 1))
        for a, b in zip(order[::2], order[1::2]):
            gid += 1
            games.append({"game_id": gid, "season": SEASON, "week": w, "season_type": "regular", "kickoff": kickoff,
                          "home_id": a, "away_id": b, "home_classification": "fbs", "away_classification": "fbs"})
            for off, df_, venue in ((a, b, 1), (b, a, -1)):
                y = 0.1 + o[off - 1] - de[df_ - 1] + 0.015 * venue + rng.normal(0, 1.4 / np.sqrt(65))
                eff.append({"game_id": gid, "season": SEASON, "week": w, "season_type": "regular", "team_id": off,
                            "opponent_id": df_, "team_classification": "fbs", "opponent_classification": "fbs",
                            "venue": venue, "plays": 65, "off_ppa": y, "pass_plays": 30, "rush_plays": 35,
                            "off_pass_ppa": y, "off_rush_ppa": y, "usable": True,
                            "available_at": kickoff + pd.Timedelta(days=1)})
    _, _, pre, _, _ = synthetic()
    pre = pd.concat([pre.iloc[:1].assign(team_id=t) for t in teams], ignore_index=True)
    return pd.DataFrame(games), pd.DataFrame(eff), pre


def test_synthetic_recovery_early_tau_exceeds_late():
    games, eff, pre = drifting()
    d = build_model_data(pre, eff, games, [SEASON])
    idata = sample(build_model(d, likelihood="normal", store_states=False, tau_phases=True),
                   draws=300, tune=500, chains=2)
    h = hyperparameters_from(idata, d, "normal", tau_phases=True)
    assert h.tau_phases and h.tau_off.shape == (4,)
    assert h.tau_off[0] > 2 * h.tau_off[2] and h.tau_def[0] > 2 * h.tau_def[2]


class TestExcludeTerms:
    def draws(self, tmp_path, **kw):
        TestPhaseTerms.write(tmp_path)
        return game_draws(TestPhaseTerms.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT, **kw)

    def test_removes_column_and_prior(self, tmp_path):
        full = self.draws(tmp_path)
        gd = self.draws(tmp_path, exclude_terms=["bowl_edge"])
        assert "bowl_edge" in full.terms and gd.terms == [t for t in TERMS_SPLIT if t != "bowl_edge"]
        np.testing.assert_array_equal(gd.x, full.x[..., [0, 1, 2]])
        assert [PRIORS[t] for t in gd.terms] == [PRIORS[t] for t in ("pass_edge", "rush_edge", "venue")]

    def test_applied_before_keep_terms(self, tmp_path):
        gd = self.draws(tmp_path, phase=True, exclude_terms=["bowl_edge"], keep_terms=["rush_edge", "pass_edge_early"])
        assert gd.terms == ["rush_edge", "pass_edge_early"]
        with pytest.raises(ValueError, match="keep_terms"):
            self.draws(tmp_path, exclude_terms=["bowl_edge"], keep_terms=["bowl_edge"])

    def test_unknown_raises(self, tmp_path):
        with pytest.raises(ValueError, match="exclude_terms"):
            self.draws(tmp_path, exclude_terms=["nope"])

    def test_tau_phases_draws_name(self, tmp_path):
        TestPhaseTerms.write(tmp_path)
        src = tmp_path / f"{draws_name(SEASON, UNITS_SPLIT)}.npz"
        src.rename(tmp_path / f"{draws_name(SEASON, UNITS_SPLIT, tau_phases=True)}.npz")
        gd = game_draws(TestPhaseTerms.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT, tau_phases=True)
        assert gd.terms == TERMS_SPLIT


def test_naming_cli_and_flags():
    drop = ("talent_qb",)
    name = hyper_name("normal", 2023, UNITS_SPLIT, conference=True, projection=True, drop_features=drop, drives=True,
                      independents_separate=True, tau_phases=True)
    assert name.startswith("normal_split_conf_ind_proj_drv_tph_drop-") and name.endswith("_2019_2022")
    assert draws_name(2024, UNITS_SPLIT, True, drives=True, independents_separate=True, tau_phases=True) == \
        "cfb_team_week_draws_split_conf_ind_drv_tph_2024"
    assert model_version(UNITS_SPLIT, True, drives=True, independents_separate=True, tau_phases=True) == \
        "dyn-eff-split-conf-ind-drv-tph-1"
    assert hyper_name("normal", 2023, UNITS_SPLIT, conference=True) == "normal_split_conf_2019_2022"
    assert snapshot_options(["2020", "2024", "--tau-phases"])["tau_phases"]
    assert not snapshot_options(["2020", "2024"])["tau_phases"]
    assert {"tau_phases", "exclude_terms"} <= set(OUTCOME_FLAGS)
