from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.efficiency import posterior_snapshots
from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT, build_model_data
from gnomepy_research.models.cfb.efficiency.gaussian_solver import fit_cutoff_exact
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import (
    HEAVY_TAILS_MCMC, snapshot_options, snapshot_season, snapshot_seasons)
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import MODEL_CONFIGS
from gnomepy_research.sessions.cfb_win_probability.evaluation.l_report import cut_table
from gnomepy_research.sessions.cfb_win_probability.evaluation.m_report import (
    favorite_bias, final_spread, m_cuts, mcmc_diagnostics, nu_row)
from tests.sessions import test_cfb_l_report as l_report_test
from tests.sessions.test_cfb_heavy_tails import heavy, synthetic_inputs
from tests.sessions.test_cfb_win_probability_models import SEASON, hypers


class TestMcmcSettings:
    def test_cli_defaults_match_heavy_tails_mcmc(self):
        assert snapshot_options(["2020", "2024"])["mcmc"] == HEAVY_TAILS_MCMC == {"draws": 300, "tune": 300, "chains": 2}

    def test_cli_overrides(self):
        o = snapshot_options(["2020", "2024", "--heavy-tails", "--mcmc-draws", "500", "--mcmc-tune", "450",
                              "--mcmc-chains", "3"])
        assert o["mcmc"] == {"draws": 500, "tune": 450, "chains": 3}

    def test_snapshot_seasons_threads_mcmc(self, tmp_path, monkeypatch):
        seen = []

        def fake_season(inputs, season, likelihood, **kw):
            seen.append((season, kw["mcmc"]))
            return pd.DataFrame({"season": [season]}), tmp_path / f"{season}.npz"

        monkeypatch.setattr(posterior_snapshots, "snapshot_season", fake_season)
        mcmc = {"draws": 500, "tune": 500, "chains": 2}
        snapshot_seasons(None, 2021, 2022, "normal", out_dir=tmp_path, heavy_tails=True, mcmc=mcmc)
        assert seen == [(2021, mcmc), (2022, mcmc)]
        seen.clear()
        snapshot_seasons(None, 2021, 2021, "normal", out_dir=tmp_path)
        assert seen == [(2021, None)]

    def test_snapshot_season_uses_override(self, tmp_path, monkeypatch):
        games, eff, pre, inputs = synthetic_inputs()
        h = heavy(hypers(build_model_data(pre, eff, games, [SEASON])))
        h.likelihood = "normal"
        calls = []

        def fake_mcmc(inputs_, season, cutoff, hy, **kw):
            calls.append(kw)
            return fit_cutoff_exact(inputs_, season, cutoff, replace(hy, heavy_tails=False), n_draws=50)

        monkeypatch.setattr(posterior_snapshots, "ensure_hyperparameters", lambda *a, **k: "x")
        monkeypatch.setattr(posterior_snapshots, "load_hyperparameters", lambda name: h)
        monkeypatch.setattr(posterior_snapshots, "fit_cutoff", fake_mcmc)
        snapshot_season(inputs, SEASON, "normal", out_dir=tmp_path, heavy_tails=True,
                        mcmc={"draws": 500, "tune": 500, "chains": 2})
        assert calls and all(c == {"draws": 500, "tune": 500, "chains": 2, "seed": 0} for c in calls)


def test_model_m_config():
    m, k = MODEL_CONFIGS["M"], MODEL_CONFIGS["K"]
    assert m == {**k, "heavy_tails": True}
    assert "garbage_weight" not in m and "calibrator" not in m
    assert m["units"] == UNITS_SPLIT


def report_games() -> pd.DataFrame:
    g = l_report_test.games()
    return g.assign(season=[2023, 2023, 2023, 2024, 2024, 2024], home_win=[1, 0, 0, 1, 0, 0],
                    p_K=[0.8, 0.5, 0.3, 0.5, 0.6, 0.4], p_M=[0.9, 0.5, 0.2, 0.5, 0.6, 0.4])


def test_m_cuts_selection_and_sides():
    g = report_games()
    c = m_cuts(g)
    assert list(c) == ["big favorites", "favorites", "toss-ups", "postseason", "regular neutral-site",
                       "regular home-site", "weeks 1-4", "weeks 10+", "P4 vs G5"]
    assert c["favorites"][0].all() and c["favorites"][1].tolist() == [False, False, True, True, False, True]
    t = cut_table(g, ["K", "M"], [("M", "K")], n_boot=50, cuts=m_cuts(g))
    assert list(t["cut"]) == list(c) and "M-K" in t


def test_favorite_bias():
    g = report_games()
    b = favorite_bias(g, ["K", "M"]).set_index("source")
    assert (b["n"] == 2).all() and np.isclose(b.loc["K", "win_rate"], 1.0)
    assert np.isclose(b.loc["K", "mean_p"], (0.8 + 0.7) / 2) and np.isclose(b.loc["M", "bias"], 1 - 0.85)
    assert np.isclose(b.loc["open", "bias"], 1 - 0.85)


def test_mcmc_diagnostics_one_row_per_cutoff():
    post = pd.DataFrame({"season": [2020] * 4 + [2021] * 2, "season_type": ["regular"] * 6,
                         "week": [1, 1, 2, 2, 1, 1], "team_id": [1, 2, 1, 2, 1, 2],
                         "max_rhat": [1.01, 1.01, 1.07, 1.07, 1.02, 1.02], "divergences": [0, 0, 3, 3, 0, 0]})
    d = mcmc_diagnostics(post).set_index("season")
    assert d.loc[2020, "cutoffs"] == 2 and d.loc[2020, "rhat_gt_1.05"] == 1
    assert np.isclose(d.loc[2020, "max_rhat"], 1.07) and d.loc[2020, "divergences"] == 3
    assert d.loc[2021, "rhat_gt_1.05"] == 0 and d.loc[2021, "cutoffs_with_div"] == 0


def test_final_spread_per_season():
    rows = []
    for season, scale in ((2023, 1.0), (2024, 2.0)):
        for team, r in ((1, 0.1), (2, 0.4), (3, -0.2)):
            for w in (1, 12):
                for unit, mult in (("pass", 1.0), ("rush", 9.0)):
                    rows.append(dict(season=season, team_id=team, unit=unit, season_type="regular", week=w,
                                     offense_mean=scale * r * mult * (w / 12), defense_mean=0.0))
    s = final_spread(pd.DataFrame(rows), [2023, 2024])
    assert np.isclose(s.loc[2023, "sd"], np.std([0.1, 0.4, -0.2], ddof=1))
    assert np.isclose(s.loc[2024, "sd"], 2 * np.std([0.1, 0.4, -0.2], ddof=1))
    assert np.isclose(s.loc[2024, "max"], 0.8) and np.isclose(s.loc[2023, "min"], -0.2)


def test_nu_rows():
    raw = {"hyperparameters": {"nu_team": 11.5, "s_off": 0.06, "s_def": 0.04},
           "diagnostics": {"hyper": {"max_rhat": 1.02, "worst_rhat_param": "nu_team", "min_ess_bulk": 140.0,
                                     "divergences": 1}}}
    r = nu_row(2023, raw)
    assert r["season"] == 2023 and r["nu_team"] == 11.5 and r["max_rhat"] == 1.02 and r["divergences"] == 1
