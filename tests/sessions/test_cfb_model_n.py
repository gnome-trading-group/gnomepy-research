from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_solve

from gnomepy_research.models.cfb.efficiency.data import (
    UNITS_ALL, UNITS_SPLIT, build_model_data, canonical_extensions, season_periods)
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import (
    Hyperparameters, _extension_values, build_model, sample)
from gnomepy_research.models.cfb.efficiency.gaussian_solver import _state_map, fit_cutoff_exact, solve
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import (
    draws_name, hyper_name, model_version, snapshot_options)
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import MODEL_CONFIGS, OUTCOME_FLAGS
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import (
    FINISH_TERM, PRIORS, TERMS_SPLIT, game_draws)
from tests.sessions import test_cfb_model_k as model_k
from tests.sessions.test_cfb_drives import TRUE_PPD, drive_hypers, exact_last_states
from tests.sessions.test_cfb_win_probability_models import N_TEAMS, SEASON, synthetic

ALL_EXT = ("couple", "fcs", "finish")


def coupled_drives(eff, true_o, true_d, kappa=(0.0,), fin_o=None, fin_d=None, noise=0.5, seed=0):
    """ppd = alpha + lam (o - d) + hfa v + fin_o - fin_d + sum_u kappa_u * (realized PPA noise of unit u) + noise.
    kappa has one entry per unit: ("all",) uses off_ppa, split uses (pass, rush)."""
    rng = np.random.default_rng(seed + 900)
    t = TRUE_PPD
    cols = ["off_ppa"] if len(kappa) == 1 else ["off_pass_ppa", "off_rush_ppa"]
    rows = []
    for r in eff.itertuples():
        fbs_o, fbs_d = r.team_classification == "fbs", r.opponent_classification == "fbs"
        o = true_o[r.team_id - 1] if fbs_o else 0.0
        de = true_d[r.opponent_id - 1] if fbs_d else 0.0
        mean = {"off_ppa": 0.1 + o - de + 0.015 * r.venue, "off_pass_ppa": 0.15 + o - de + 0.015 * r.venue}
        mean["off_rush_ppa"] = (65 * mean["off_ppa"] - 30 * mean["off_pass_ppa"]) / 35
        shock = sum(k * (getattr(r, c) - mean[c]) for k, c in zip(kappa, cols))
        f = (fin_o[r.team_id - 1] if fbs_o and fin_o is not None else 0.0) - \
            (fin_d[r.opponent_id - 1] if fbs_d and fin_d is not None else 0.0)
        ppd = t["alpha"] + t["lam"] * (o - de) + t["hfa"] * r.venue + f + shock + rng.normal(0, noise)
        rows.append({"game_id": r.game_id, "season": r.season, "season_type": r.season_type, "week": r.week,
                     "team_id": r.team_id, "opponent_id": r.opponent_id, "team_classification": r.team_classification,
                     "opponent_classification": r.opponent_classification, "venue": r.venue, "ppd": ppd,
                     "n_eff": t["n"], "available_at": r.available_at})
    return pd.DataFrame(rows)


def ext_hypers(d, kappa=None, s_fcs=0.0, s_fcs_ppd=0.0, s_fin=0.0) -> Hyperparameters:
    h = drive_hypers(d)
    h.extensions = list(d.extensions)
    h.kappa_ppd = list(kappa if kappa is not None else [0.0] * len(d.units))
    h.s_fcs, h.s_fcs_ppd, h.s_fin_off, h.s_fin_def = s_fcs, s_fcs_ppd, s_fin, s_fin
    return h


def state_sd(d, h) -> np.ndarray:
    """Posterior sd of every offense state at the last period from the exact solver."""
    _, chol, lay = solve(d, h)
    ts = np.arange(len(d.ts_keys))
    _, B = _state_map(d, h, lay, ts, np.full(len(ts), d.n_periods - 1), offense=True)
    return np.sqrt(np.einsum("ij,ji->i", B, cho_solve(chol, B.T)))


class TestExtensionData:
    def test_canonical_order_and_validation(self):
        assert canonical_extensions(["finish", "fcs", "couple"], drives=True) == ALL_EXT
        assert canonical_extensions(["fcs"], drives=False) == ("fcs",)
        with pytest.raises(ValueError):
            canonical_extensions(["couple"], drives=False)
        with pytest.raises(ValueError):
            canonical_extensions(["finish"], drives=False)
        with pytest.raises(ValueError):
            canonical_extensions(["bogus"], drives=True)

    def test_default_has_no_extension_arrays(self):
        games, eff, pre, o, de = synthetic()
        d = build_model_data(pre, eff, games, [SEASON], drives=coupled_drives(eff, o, de))
        assert d.extensions == () and len(d.obs_fcs) == 0 and d.drv_obs.shape[0] == 0 and d.n_base == 0

    @pytest.mark.parametrize("units", [UNITS_ALL, UNITS_SPLIT])
    def test_drive_rows_link_to_same_game_offense_and_unit(self, units):
        games, eff, pre, o, de = synthetic()
        d = build_model_data(pre, eff, games, [SEASON], units=units, drives=coupled_drives(eff, o, de),
                             extensions=ALL_EXT)
        assert d.drv_obs.shape == (len(d.drv_y), len(units)) and (d.drv_obs >= 0).all()
        for u in range(len(units)):
            rows = d.drv_obs[:, u]
            np.testing.assert_array_equal(d.obs_game[rows], d.drv_game)
            np.testing.assert_array_equal(d.obs_off[rows], d.drv_off[:, u])
            np.testing.assert_array_equal(d.obs_unit[rows], u)
        assert d.n_base == N_TEAMS + 1
        np.testing.assert_array_equal(d.drv_base_off, d.drv_off[:, 0])
        np.testing.assert_array_equal(d.ts_keys["team_id"].to_numpy()[d.drv_base_off],
                                      d.ts_keys["team_id"].to_numpy()[d.drv_off[:, -1]])

    def test_fcs_flags(self):
        games, eff, pre, o, de = synthetic()
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, drives=coupled_drives(eff, o, de),
                             extensions=("fcs",))
        np.testing.assert_array_equal(d.obs_fcs == 1, d.obs_game == 999)
        np.testing.assert_array_equal(d.drv_fcs == 1, d.drv_game == 999)
        no_drv = build_model_data(pre, eff, games, [SEASON], extensions=("fcs",))
        assert no_drv.obs_fcs.sum() == 1 and len(no_drv.drv_fcs) == 0


class TestHyperparameters:
    def test_default_dict_unchanged_and_round_trip(self):
        games, eff, pre, o, de = synthetic()
        plain = build_model_data(pre, eff, games, [SEASON], drives=coupled_drives(eff, o, de))
        keys = drive_hypers(plain).to_dict().keys()
        assert not {"extensions", "kappa_ppd", "s_fcs", "s_fcs_ppd", "s_fin_off", "s_fin_def"} & set(keys)
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, drives=coupled_drives(eff, o, de),
                             extensions=("couple", "finish"))
        h = ext_hypers(d, kappa=[1.5, 0.5], s_fin=0.3)
        out = h.to_dict()
        assert out["extensions"] == ["couple", "finish"] and out["kappa_ppd"] == [1.5, 0.5]
        assert "s_fcs" not in out and "s_fcs_ppd" not in out
        back = Hyperparameters.from_dict(out)
        assert back.extensions == ["couple", "finish"] and back.kappa_ppd == [1.5, 0.5] and back.s_fin_def == 0.3

    def test_extension_values_from_posterior_means(self):
        games, eff, pre, o, de = synthetic()
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, drives=coupled_drives(eff, o, de),
                             extensions=ALL_EXT)
        means = {"kappa_ppd": np.array([1.2, 0.8]), "s_fcs": np.array(0.1), "s_fcs_ppd": np.array(0.4),
                 "s_fin_off": np.array(0.3), "s_fin_def": np.array(0.2)}
        v = _extension_values(lambda k: means[k], d)
        assert v == {"extensions": list(ALL_EXT), "kappa_ppd": [1.2, 0.8], "s_fcs": 0.1, "s_fcs_ppd": 0.4,
                     "s_fin_off": 0.3, "s_fin_def": 0.2}
        assert _extension_values(lambda k: means[k], build_model_data(pre, eff, games, [SEASON])) == {}

    def test_free_model_variables(self):
        games, eff, pre, o, de = synthetic()
        drv = coupled_drives(eff, o, de)
        plain = {v.name for v in build_model(build_model_data(pre, eff, games, [SEASON], drives=drv)).free_RVs}
        assert not {"kappa_ppd", "s_fcs", "s_fin_off", "z_fin_off"} & plain
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, drives=drv, extensions=ALL_EXT)
        names = {v.name for v in build_model(d).free_RVs}
        assert {"kappa_ppd", "s_fcs", "s_fcs_ppd", "s_fin_off", "s_fin_def", "z_fin_off", "z_fin_def"} <= names

    def test_mismatched_extensions_raise(self):
        games, eff, pre, o, de = synthetic()
        d = build_model_data(pre, eff, games, [SEASON], drives=coupled_drives(eff, o, de), extensions=("couple",))
        h = drive_hypers(d)
        with pytest.raises(ValueError):
            solve(d, h)
        with pytest.raises(ValueError):
            build_model(d, hypers=h, likelihood="normal")


def model_k_hypers(d):
    h = drive_hypers(d)
    h.drives = False
    return h


def test_coupling_removes_double_counting():
    """When a game's ppd is its PPA noise times kappa plus little else, it carries no new information about
    team states: the coupled posterior sd matches PPA-only, the independent model is overconfident."""
    games, eff, pre, o, de = synthetic(seed=2)
    lam = TRUE_PPD["lam"]
    drv = coupled_drives(eff, o, de, kappa=(lam,), noise=0.05, seed=2)
    ppa_only = build_model_data(pre, eff, games, [SEASON])
    indep = build_model_data(pre, eff, games, [SEASON], drives=drv)
    coupled = build_model_data(pre, eff, games, [SEASON], drives=drv, extensions=("couple",))
    sd0 = state_sd(ppa_only, model_k_hypers(ppa_only))[:N_TEAMS]
    h_ind = drive_hypers(indep)
    h_ind.s_game_ppd, h_ind.s_drive_ppd = 0.05, 0.0
    h_cpl = ext_hypers(coupled, kappa=[lam])
    h_cpl.s_game_ppd, h_cpl.s_drive_ppd = 0.05, 0.0
    sd_ind, sd_cpl = state_sd(indep, h_ind)[:N_TEAMS], state_sd(coupled, h_cpl)[:N_TEAMS]
    assert np.all(sd_ind < 0.8 * sd0)
    np.testing.assert_allclose(sd_cpl, sd0, rtol=0.05)


def test_fcs_noise_shrinks_fcs_game_influence():
    games, eff, pre, o, de = synthetic()
    fcs_row = eff["game_id"] == 999

    def team1_offense(bump: float, s_fcs: float) -> float:
        e = eff.copy()
        e.loc[fcs_row, "off_ppa"] += bump
        d = build_model_data(pre, e, games, [SEASON], extensions=("fcs",))
        h = model_k_hypers(d)
        h.extensions, h.s_fcs = ["fcs"], s_fcs
        return exact_last_states(d, h)[0][0]

    plain = team1_offense(1.0, 0.0) - team1_offense(0.0, 0.0)
    noisy = team1_offense(1.0, 0.3) - team1_offense(0.0, 0.3)
    assert 0 < noisy < 0.5 * plain


def finishing_world(seed=0, n_teams=30, weeks=10, fin_sd=0.6):
    rng = np.random.default_rng(seed)
    teams = np.arange(1, n_teams + 1)
    o, de = rng.normal(0, 0.1, n_teams), rng.normal(0, 0.1, n_teams)
    fo, fd = rng.normal(0, fin_sd, n_teams), rng.normal(0, fin_sd, n_teams)
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
    eff = pd.DataFrame(eff)
    return pd.DataFrame(games), eff, pre, o, de, fo, fd


def test_finishing_effects_are_recovered_and_unconfound_offense():
    """Finishing is only seen by drives, where it competes with lam * O, so recovery is partial; separating it
    should also clean the per-play offense estimate."""
    fin_r, o_gain = [], []
    for seed in range(3):
        games, eff, pre, o, de, fo, fd = finishing_world(seed=seed)
        drv = coupled_drives(eff, o, de, fin_o=fo, fin_d=fd, noise=0.5, seed=seed)
        d = build_model_data(pre, eff, games, [SEASON], drives=drv, extensions=("finish",))
        inputs = Inputs(pre, eff, games, as_of=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime(), drives=drv)
        cutoff = season_periods(games, [SEASON])["cutoff"].iloc[-1]
        post, _ = fit_cutoff_exact(inputs, SEASON, cutoff, ext_hypers(d, s_fin=0.6), n_draws=300)
        plain, _ = fit_cutoff_exact(inputs, SEASON, cutoff,
                                    drive_hypers(build_model_data(pre, eff, games, [SEASON], drives=drv)), n_draws=300)
        n = len(fo)
        assert post.fin_off.shape == post.off.shape and plain.fin_off is None
        fin_r += [np.corrcoef(post.fin_off.mean(0)[:n], fo)[0, 1], np.corrcoef(post.fin_def.mean(0)[:n], fd)[0, 1]]
        o_gain.append(np.corrcoef(post.off.mean(0)[:n], o)[0, 1] - np.corrcoef(plain.off.mean(0)[:n], o)[0, 1])
    assert np.mean(fin_r) > 0.5 and min(fin_r) > 0.3
    assert np.mean(o_gain) > 0


@pytest.mark.parametrize("units", [UNITS_ALL, UNITS_SPLIT])
def test_exact_solver_matches_mcmc_with_all_extensions(units):
    games, eff, pre, o, de = synthetic(seed=1)
    rng = np.random.default_rng(7)
    fo, fd = rng.normal(0, 0.4, N_TEAMS), rng.normal(0, 0.4, N_TEAMS)
    kappa = (2.0,) if units == UNITS_ALL else (1.5, 1.0)
    drv = coupled_drives(eff, o, de, kappa=kappa, fin_o=fo, fin_d=fd, seed=1)
    d = build_model_data(pre, eff, games, [SEASON], units=units, drives=drv, extensions=ALL_EXT)
    h = ext_hypers(d, kappa=list(kappa), s_fcs=0.1, s_fcs_ppd=0.5, s_fin=0.4)
    exact_o, exact_d = exact_last_states(d, h)
    mean, _, lay = solve(d, h)
    exact_fin = h.s_fin_off * mean[lay["z_fin_off"]:lay["z_fin_off"] + d.n_base]
    idata = sample(build_model(d, hypers=h, likelihood="normal"), draws=1000, tune=500, chains=2)
    post = idata.posterior.to_dataset()
    np.testing.assert_allclose(exact_o, post["O"].mean(("chain", "draw")).values[:, -1], atol=0.01)
    np.testing.assert_allclose(exact_d, post["D"].mean(("chain", "draw")).values[:, -1], atol=0.01)
    np.testing.assert_allclose(exact_fin, post["FIN_O"].mean(("chain", "draw")).values, atol=0.03)


class TestFinishTerm:
    @staticmethod
    def write(tmp_path):
        model_k.TestPhaseTerms.write(tmp_path)
        src = tmp_path / f"{draws_name(SEASON, UNITS_SPLIT)}.npz"
        with np.load(src) as z:
            arrays = {k: z[k] for k in z.files}
        rng = np.random.default_rng(3)
        for key in ("regular_1", "regular_4", "regular_5", "postseason_1"):
            f_off, f_def = rng.normal(0, 0.5, (5, 3)), rng.normal(0, 0.5, (5, 3))
            arrays[f"{key}__fin_off"] = np.hstack([f_off, f_off]).astype("float32")
            arrays[f"{key}__fin_def"] = np.hstack([f_def, f_def]).astype("float32")
        src.unlink()
        name = draws_name(SEASON, UNITS_SPLIT, drives=True, extensions=("finish",))
        np.savez(tmp_path / f"{name}.npz", **arrays)
        return arrays

    def test_finish_edge_values_and_prior(self, tmp_path):
        arrays = self.write(tmp_path)
        gd = game_draws(model_k.TestPhaseTerms.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT, drives=True,
                        extensions=("finish",))
        assert gd.terms == TERMS_SPLIT + [FINISH_TERM] and PRIORS[FINISH_TERM] == (1.0, 1.0)
        for i, row in gd.games.iterrows():
            key = f"{row['season_type']}_{row['week']}"
            fo, fd = arrays[f"{key}__fin_off"], arrays[f"{key}__fin_def"]
            expect = (fo[:, 0] - fd[:, 1]) - (fo[:, 1] - fd[:, 0])
            np.testing.assert_allclose(gd.x[i, :, -1], expect, rtol=1e-6)

    def test_missing_finish_arrays_raise(self, tmp_path):
        model_k.TestPhaseTerms.write(tmp_path)
        src = tmp_path / f"{draws_name(SEASON, UNITS_SPLIT)}.npz"
        src.rename(tmp_path / f"{draws_name(SEASON, UNITS_SPLIT, drives=True, extensions=('finish',))}.npz")
        with pytest.raises(ValueError, match="finishing"):
            game_draws(model_k.TestPhaseTerms.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT, drives=True,
                       extensions=("finish",))


def test_naming_cli_and_configs():
    name = hyper_name("normal", 2023, UNITS_SPLIT, True, drives=True, independents_separate=True,
                      extensions=("finish", "couple", "fcs"))
    assert name == "normal_split_conf_ind_drv_cpl_fcs_fin_2019_2022"
    assert hyper_name("normal", 2023, UNITS_SPLIT, True, drop_features=("talent_qb",), drives=True,
                      heavy_tails=True, extensions=("couple",)).startswith("normal_split_conf_drv_ht_cpl_drop-")
    assert hyper_name("normal", 2023, extensions=("fcs",)) == "normal_fcs_2019_2022"
    assert draws_name(2024, UNITS_SPLIT, True, drives=True, independents_separate=True, extensions=ALL_EXT) == \
        "cfb_team_week_draws_split_conf_ind_drv_cpl_fcs_fin_2024"
    assert model_version(UNITS_SPLIT, True, drives=True, extensions=ALL_EXT) == "dyn-eff-split-conf-drv-cpl-fcs-fin-1"
    assert draws_name(2024, UNITS_SPLIT, True, drives=True) == "cfb_team_week_draws_split_conf_drv_2024"
    with pytest.raises(ValueError):
        hyper_name("normal", 2023, UNITS_SPLIT, extensions=("couple",))
    assert snapshot_options(["2020", "2024", "--drives", "--extensions", "fcs,couple"])["extensions"] == ("couple", "fcs")
    assert snapshot_options(["2020", "2024"])["extensions"] == ()
    with pytest.raises(ValueError):
        snapshot_options(["2020", "2024", "--extensions", "finish"])
    assert "extensions" in OUTCOME_FLAGS
    k = {key: v for key, v in MODEL_CONFIGS["K"].items()}
    assert MODEL_CONFIGS["N1"] == {**k, "extensions": ("couple",)}
    assert MODEL_CONFIGS["N2"] == {**k, "extensions": ("couple", "fcs")}
    assert MODEL_CONFIGS["N"] == {**k, "extensions": ALL_EXT}


def test_selection_ladder_rule():
    from gnomepy_research.sessions.cfb_win_probability.evaluation import n_selection as ns
    assert ns.ladder_step((), "couple") == ("couple",)
    assert ns.ladder_step(("couple",), "finish") == ("couple", "finish")
    assert ns.ladder_step(("finish",), "couple") == ("couple", "finish")
    assert ns.keep(0.5670, 0.5660) and ns.keep(0.5679, 0.5660) and ns.keep(0.5600, 0.5660)
    assert not ns.keep(0.5681, 0.5660)
    assert ns.candidate(("fcs",)) == {**MODEL_CONFIGS["K"], "extensions": ("fcs",)}
    assert ns.label(()) == "K" and ns.label(("couple", "fcs")) == "K+couple+fcs"
    assert ns.snapshot_kwargs(ns.candidate(ALL_EXT))["extensions"] == ALL_EXT


def test_disagreement_table_buckets_on_base_model():
    from gnomepy_research.sessions.cfb_win_probability.evaluation.n_report import disagreement_table
    g = pd.DataFrame({"home_win": [1, 0, 1, 1], "p_open": [0.6, 0.6, 0.9, 0.5],
                      "p_K": [0.6, 0.45, 0.5, 0.52], "p_N": [0.7, 0.5, 0.8, 0.5]})
    t = disagreement_table(g, ["K", "N"]).set_index("|logit open - logit K|")
    assert t["n"].tolist() == [2, 0, 1, 1]
    ll = lambda y, p: -np.log(p) if y else -np.log(1 - p)
    assert t.loc[">1", "K total"] == pytest.approx(ll(1, 0.5) - ll(1, 0.9), abs=0.01)
    assert t.loc["<0.25", "N mean"] == pytest.approx(((ll(1, 0.7) - ll(1, 0.6)) + 0.0) / 2, abs=1e-4)
