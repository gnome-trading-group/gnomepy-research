from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from gnomepy_research.models.cfb.efficiency import posterior_snapshots
from gnomepy_research.models.cfb.efficiency.data import UNITS_ALL, UNITS_SPLIT, build_model_data
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import (
    Hyperparameters, build_model, hyperparameters_from)
from gnomepy_research.models.cfb.efficiency.gaussian_solver import fit_cutoff_exact
from gnomepy_research.models.cfb.efficiency.inference import Inputs, fit_cutoff, fit_hyperparameters
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import (
    draws_name, hyper_name, model_version, snapshot_options, snapshot_season)
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import OUTCOME_FLAGS
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import TERMS_SPLIT, game_draws
from tests.sessions import test_cfb_model_k as model_k
from tests.sessions.test_cfb_win_probability_models import SEASON, hypers, synthetic

AS_OF = pd.Timestamp("2025-06-01", tz="UTC")


def heavy(h: Hyperparameters, nu_team: float = 3.0) -> Hyperparameters:
    return replace(h, heavy_tails=True, nu_team=nu_team)


def rv_op(model: pm.Model, name: str) -> str:
    return type(model[name].owner.op).__name__


def synthetic_inputs(seed=0):
    games, eff, pre, _, _ = synthetic(seed=seed)
    return games, eff, pre, Inputs(preseason=pre, efficiency=eff, games=games, as_of=AS_OF)


class TestDefaultUnchanged:
    def test_hyperparameters_default(self):
        games, eff, pre, _, _ = synthetic()
        h = hypers(build_model_data(pre, eff, games, [SEASON]))
        assert h.heavy_tails is False and h.nu_team == float("inf")
        d = h.to_dict()
        assert "heavy_tails" not in d and "nu_team" not in d
        back = Hyperparameters.from_dict(d)
        assert back.heavy_tails is False and back.nu_team == float("inf")

    def test_free_and_fixed_models_keep_normal_start(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, games, [SEASON])
        free = build_model(d)
        assert "nu_team" not in free.named_vars
        assert rv_op(free, "z0_off") == "NormalRV" and rv_op(free, "z0_def") == "NormalRV"
        fixed = build_model(d, hypers=hypers(d))
        assert rv_op(fixed, "z0_off") == "NormalRV"


class TestHeavyModel:
    def test_round_trip(self):
        games, eff, pre, _, _ = synthetic()
        h = heavy(hypers(build_model_data(pre, eff, games, [SEASON])), 7.5)
        d = h.to_dict()
        assert d["heavy_tails"] is True and d["nu_team"] == 7.5
        back = Hyperparameters.from_dict(d)
        assert back.heavy_tails and back.nu_team == 7.5

    @pytest.mark.parametrize("units", [UNITS_ALL, UNITS_SPLIT])
    def test_free_model_has_nu_team_and_t_start(self, units):
        games, eff, pre, _, _ = synthetic()
        m = build_model(build_model_data(pre, eff, games, [SEASON], units=units), heavy_tails=True)
        assert rv_op(m, "nu_team") == "GammaRV"
        assert rv_op(m, "z0_off") == "StudentTRV" and rv_op(m, "z0_def") == "StudentTRV"
        assert rv_op(m, "eps_off") == "NormalRV"
        a, b = m["nu_team"].owner.inputs[-2:]
        assert (float(a.eval()), float(b.eval())) == (2.0, pytest.approx(5.0))

    def test_fixed_model_uses_hyper_nu_team(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, games, [SEASON])
        m = build_model(d, hypers=heavy(hypers(d), 4.0), heavy_tails=False)
        assert "nu_team" not in m.named_vars
        assert rv_op(m, "z0_off") == "StudentTRV"
        assert np.unique(m["z0_off"].owner.inputs[2].eval()).tolist() == [4.0]

    def test_hyperparameters_from_reads_nu_team(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, games, [SEASON])
        with build_model(d, store_states=False, heavy_tails=True):
            prior = pm.sample_prior_predictive(draws=20, random_seed=0).prior
        h = hyperparameters_from(SimpleNamespace(posterior=prior), d, "studentt", heavy_tails=True)
        assert h.heavy_tails and h.nu_team == pytest.approx(float(prior["nu_team"].mean()))
        assert "nu_team" in h.to_dict()
        assert not hyperparameters_from(SimpleNamespace(posterior=prior), d, "studentt").heavy_tails

    def test_fit_hyperparameters_heavy(self):
        _, _, _, inputs = synthetic_inputs()
        h, diag = fit_hyperparameters(inputs, [SEASON], "normal", draws=40, tune=40, heavy_tails=True)
        assert h.heavy_tails and np.isfinite(h.nu_team) and h.nu_team > 0
        assert diag["hyper"]["min_ess_bulk"] >= 0


def test_exact_solver_refuses_heavy_tails():
    games, eff, pre, inputs = synthetic_inputs()
    d = build_model_data(pre, eff, games, [SEASON])
    h = heavy(hypers(d))
    h.likelihood = "normal"
    with pytest.raises(ValueError, match="heavy"):
        fit_cutoff_exact(inputs, SEASON, d.periods["cutoff"].iloc[-1], h)


class TestSnapshotDispatch:
    def run(self, tmp_path, monkeypatch, h):
        games, eff, pre, inputs = synthetic_inputs()
        calls = []

        def fake_mcmc(inputs_, season, cutoff, hy, **kw):
            calls.append(("mcmc", kw))
            return fit_cutoff_exact(inputs_, season, cutoff, replace(hy, heavy_tails=False), n_draws=50)

        def fake_exact(inputs_, season, cutoff, hy, **kw):
            calls.append(("exact", kw))
            return fit_cutoff_exact(inputs_, season, cutoff, hy, n_draws=50)

        monkeypatch.setattr(posterior_snapshots, "ensure_hyperparameters", lambda *a, **k: "x")
        monkeypatch.setattr(posterior_snapshots, "load_hyperparameters", lambda name: h)
        monkeypatch.setattr(posterior_snapshots, "fit_cutoff", fake_mcmc)
        monkeypatch.setattr(posterior_snapshots, "fit_cutoff_exact", fake_exact)
        out, path = snapshot_season(inputs, SEASON, "normal", out_dir=tmp_path, heavy_tails=h.heavy_tails)
        return calls, out, path

    def test_heavy_tails_uses_mcmc(self, tmp_path, monkeypatch):
        games, eff, pre, _ = synthetic_inputs()
        h = heavy(hypers(build_model_data(pre, eff, games, [SEASON])))
        h.likelihood = "normal"
        calls, out, path = self.run(tmp_path, monkeypatch, h)
        assert calls and all(c == ("mcmc", {"draws": 300, "tune": 300, "chains": 2, "seed": 0}) for c in calls)
        assert path.name == f"cfb_team_week_draws_ht_{SEASON}.npz"
        assert set(out["model_version"]) == {"dyn-eff-ht-1"}

    def test_default_normal_uses_exact(self, tmp_path, monkeypatch):
        games, eff, pre, _ = synthetic_inputs()
        h = hypers(build_model_data(pre, eff, games, [SEASON]))
        h.likelihood = "normal"
        calls, _, path = self.run(tmp_path, monkeypatch, h)
        assert calls and all(c == ("exact", {"seed": 0}) for c in calls)
        assert path.name == f"cfb_team_week_draws_{SEASON}.npz"


def test_naming_cli_and_flags(tmp_path):
    name = hyper_name("normal", 2023, UNITS_SPLIT, conference=True, projection=True, drop_features=("talent_qb",),
                      drives=True, independents_separate=True, tau_phases=True, heavy_tails=True)
    assert name.startswith("normal_split_conf_ind_proj_drv_tph_ht_drop-") and name.endswith("_2019_2022")
    assert draws_name(2024, UNITS_SPLIT, True, drives=True, independents_separate=True, heavy_tails=True) == \
        "cfb_team_week_draws_split_conf_ind_drv_ht_2024"
    assert model_version(UNITS_SPLIT, True, drives=True, tau_phases=True, heavy_tails=True) == \
        "dyn-eff-split-conf-drv-tph-ht-1"
    assert hyper_name("normal", 2023, UNITS_SPLIT, True, tau_phases=True) == "normal_split_conf_tph_2019_2022"
    assert snapshot_options(["2020", "2024", "--heavy-tails"])["heavy_tails"]
    assert not snapshot_options(["2020", "2024"])["heavy_tails"]
    assert "heavy_tails" in OUTCOME_FLAGS
    model_k.TestPhaseTerms.write(tmp_path)
    src = tmp_path / f"{draws_name(SEASON, UNITS_SPLIT)}.npz"
    src.rename(tmp_path / f"{draws_name(SEASON, UNITS_SPLIT, heavy_tails=True)}.npz")
    gd = game_draws(model_k.TestPhaseTerms.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT, heavy_tails=True)
    assert gd.terms == TERMS_SPLIT


def planted(seed=3, n_teams=40, weeks=4, sd=0.1, star_sd=4.0):
    """Ratings ~ N(0, sd) except team 1's offense at +star_sd sd; static over `weeks` regular weeks."""
    rng = np.random.default_rng(seed)
    teams = np.arange(1, n_teams + 1)
    o, de = rng.normal(0, sd, n_teams), rng.normal(0, sd, n_teams)
    o[0] = star_sd * sd
    kick0 = pd.Timestamp("2024-08-31 16:00", tz="UTC")
    games, eff, gid = [], [], 0
    for w in range(1, weeks + 1):
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
    return pd.DataFrame(games), pd.DataFrame(eff), pre, o, de


def recovery(seed: int) -> dict:
    """Star offense posterior mean and ordinary-team RMSE (other offenses, all defenses) per prior."""
    games, eff, pre, o, de = planted(seed=seed)
    inputs = Inputs(preseason=pre, efficiency=eff, games=games, as_of=AS_OF)
    d = build_model_data(pre, eff, games, [SEASON])
    base = replace(hypers(d), s_off=0.1, s_def=0.1, likelihood="normal")
    out = {"truth": o[0]}
    for label, h in (("normal", base), ("heavy", heavy(base, 3.0))):
        post, _ = fit_cutoff(inputs, SEASON, d.periods["cutoff"].iloc[-1], h, draws=300, tune=300, chains=2)
        assert post.diagnostics["max_rhat"] < 1.05 and post.diagnostics["divergences"] == 0
        col = {t: i for i, t in enumerate(post.team_ids)}
        idx = [col[t] for t in range(1, len(o) + 1)]
        om, dm = post.off.mean(0)[idx], post.de.mean(0)[idx]
        out[label] = (om[0], np.sqrt(np.mean(np.r_[om[1:] - o[1:], dm - de] ** 2)))
    return out


def test_synthetic_recovery_heavy_tails_keeps_planted_star():
    t0 = time.time()
    runs = [recovery(seed) for seed in range(3)]
    for r in runs:
        print(f"star truth {r['truth']:.3f} normal {r['normal'][0]:.3f} heavy {r['heavy'][0]:.3f}; "
              f"rmse ordinary normal {r['normal'][1]:.4f} heavy {r['heavy'][1]:.4f}")
    err_n, err_h = (np.array([abs(r[k][0] - r["truth"]) for r in runs]) for k in ("normal", "heavy"))
    assert (err_h < err_n).all() and err_h.mean() < 0.75 * err_n.mean()
    rmse_n, rmse_h = (np.mean([r[k][1] for r in runs]) for k in ("normal", "heavy"))
    print(f"mean ordinary rmse normal {rmse_n:.4f} heavy {rmse_h:.4f}; {time.time() - t0:.1f}s")
    assert abs(rmse_h - rmse_n) < 0.1 * rmse_n
