from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.efficiency.data import UNITS_ALL, UNITS_SPLIT, build_model_data, season_periods
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import Hyperparameters, build_model, sample
from gnomepy_research.models.cfb.efficiency.gaussian_solver import _state_map, fit_cutoff_exact, solve
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name, hyper_name, model_version
from gnomepy_research.models.cfb.features.drives import GARBAGE_WEIGHT, team_game_drives
from tests.sessions.test_cfb_win_probability_models import N_TEAMS, SEASON, hypers, synthetic

TRUE_PPD = {"alpha": 2.2, "lam": 12.0, "hfa": 0.2, "s_game": 0.5, "s_drive": 2.5, "n": 12.0}


def drive(game_id, off, de, home, period, off_score, def_score, points, result="PUNT", n_plays=6):
    return {"drive_id": f"{game_id}-{off}-{period}-{off_score}-{points}", "game_id": game_id, "season": SEASON,
            "offense": str(off), "defense": str(de), "is_home_offense": home, "start_period": period,
            "start_offense_score": off_score, "start_defense_score": def_score, "off_points": points,
            "def_points": 0, "drive_result": result, "n_plays": n_plays, "offense_id": off, "defense_id": de}


def hand_games():
    k = pd.Timestamp("2024-09-07 16:00", tz="UTC")
    return pd.DataFrame([
        {"game_id": 1, "season": SEASON, "season_type": "regular", "week": 2, "kickoff": k, "neutral_site": False,
         "home_id": 10, "away_id": 20, "home_classification": "fbs", "away_classification": "fcs"},
        {"game_id": 2, "season": SEASON, "season_type": "regular", "week": 3, "kickoff": k + pd.Timedelta(days=7),
         "neutral_site": True, "home_id": 10, "away_id": 30, "home_classification": "fbs", "away_classification": "fbs"},
    ])


class TestTeamGameDrives:
    def test_kneels_overtime_and_garbage(self):
        d = pd.DataFrame([
            drive(1, 10, 20, True, 1, 0, 0, 7, "TD"),
            drive(1, 10, 20, True, 2, 7, 0, 0, "END OF HALF", n_plays=1),
            drive(1, 10, 20, True, 2, 7, 0, 3, "END OF HALF", n_plays=8),
            drive(1, 10, 20, True, 4, 40, 0, 7, "TD"),
            drive(1, 10, 20, True, 5, 47, 0, 7, "TD"),
            drive(1, 20, 10, False, 1, 0, 7, 0),
        ])
        out = team_game_drives(d, hand_games()).set_index("team_id")
        home = out.loc[10]
        assert home["n_eff"] == pytest.approx(2 + GARBAGE_WEIGHT)
        assert home["ppd"] == pytest.approx((7 + 3 + GARBAGE_WEIGHT * 7) / (2 + GARBAGE_WEIGHT))
        assert home["venue"] == 1 and out.loc[20, "venue"] == -1
        assert home["team_classification"] == "fbs" and home["opponent_classification"] == "fcs"
        assert out.loc[20, "team_classification"] == "fcs" and out.loc[20, "ppd"] == 0.0
        assert home["available_at"] == hand_games()["kickoff"].iloc[0] + pd.Timedelta(days=1)
        assert (home["season_type"], home["week"], home["opponent_id"]) == ("regular", 2, 20)

    def test_garbage_threshold_is_strict_and_neutral_venue(self):
        d = pd.DataFrame([drive(2, 10, 30, True, 3, 28, 0, 7, "TD"), drive(2, 10, 30, True, 3, 29, 0, 0)])
        out = team_game_drives(d, hand_games())
        assert out["n_eff"].iloc[0] == pytest.approx(1 + GARBAGE_WEIGHT)
        assert out["ppd"].iloc[0] == pytest.approx(7 / (1 + GARBAGE_WEIGHT))
        assert out["venue"].iloc[0] == 0


def synthetic_drives(eff: pd.DataFrame, true_o, true_d, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 500)
    t = TRUE_PPD
    rows = []
    for r in eff.itertuples():
        o = true_o[r.team_id - 1] if r.team_classification == "fbs" else 0.0
        de = true_d[r.opponent_id - 1] if r.opponent_classification == "fbs" else 0.0
        sd = np.sqrt(t["s_game"] ** 2 + t["s_drive"] ** 2 / t["n"])
        ppd = t["alpha"] + t["lam"] * (o - de) + t["hfa"] * r.venue + rng.normal(0, sd)
        rows.append({"game_id": r.game_id, "season": r.season, "season_type": r.season_type, "week": r.week,
                     "team_id": r.team_id, "opponent_id": r.opponent_id, "team_classification": r.team_classification,
                     "opponent_classification": r.opponent_classification, "venue": r.venue, "ppd": ppd,
                     "n_eff": t["n"], "available_at": r.available_at})
    return pd.DataFrame(rows)


def drive_hypers(d, likelihood="normal") -> Hyperparameters:
    h = hypers(d)
    h.likelihood = likelihood
    h.alpha = np.full(len(d.units), 0.1)
    h.drives, h.alpha_ppd, h.sigma_alpha_ppd = True, TRUE_PPD["alpha"], 0.1
    h.lam_ppd, h.hfa_ppd, h.s_game_ppd, h.s_drive_ppd = (TRUE_PPD["lam"], TRUE_PPD["hfa"], TRUE_PPD["s_game"],
                                                         TRUE_PPD["s_drive"])
    return h


def exact_last_states(d, h):
    mean, _, lay = solve(d, h)
    ts = np.arange(len(d.ts_keys))
    k = np.full(len(ts), d.n_periods - 1)
    c_off, B_off = _state_map(d, h, lay, ts, k, offense=True)
    c_def, B_def = _state_map(d, h, lay, ts, k, offense=False)
    return c_off + B_off @ mean, c_def + B_def @ mean


class TestDriveModelData:
    def test_default_has_no_drive_observations(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, games, [SEASON])
        assert d.drives is False and len(d.drv_y) == 0

    def test_all_units_weights_are_one(self):
        games, eff, pre, o, de = synthetic()
        drv = synthetic_drives(eff, o, de)
        d = build_model_data(pre, eff, games, [SEASON], drives=drv)
        assert d.drives and len(d.drv_y) == len(drv)
        assert d.drv_off.shape == (len(drv), 1) and np.all(d.drv_off_w == 1.0) and np.all(d.drv_def_w == 1.0)

    def test_split_weights_are_season_to_date_pass_shares(self):
        games, eff, pre, o, de = synthetic()
        eff.loc[eff["team_id"] == 3, "pass_plays"] = 45
        eff.loc[eff["team_id"] == 3, "rush_plays"] = 15
        drv = synthetic_drives(eff, o, de)
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, drives=drv)
        units = d.ts_keys["unit"].to_numpy()
        assert (units[d.drv_off[:, 0]] == "pass").all() and (units[d.drv_off[:, 1]] == "rush").all()
        team = d.ts_keys["team_id"].to_numpy()[d.drv_off[:, 0]]
        np.testing.assert_allclose(d.drv_off_w[team == 3, 0], 0.75)
        np.testing.assert_allclose(d.drv_off_w[team == 4, 0], 30 / 65)
        np.testing.assert_allclose(d.drv_off_w.sum(1), 1.0)
        opp = d.ts_keys["team_id"].to_numpy()[d.drv_def[:, 0]]
        faced = eff[eff["opponent_id"] == 3]
        share = faced["pass_plays"].sum() / (faced["pass_plays"] + faced["rush_plays"]).sum()
        np.testing.assert_allclose(d.drv_def_w[opp == 3, 0], share)

    def test_team_without_efficiency_rows_gets_half(self):
        games, eff, pre, o, de = synthetic()
        drv = synthetic_drives(eff, o, de)
        d = build_model_data(pre, eff[eff["team_id"] != 2], games, [SEASON], units=UNITS_SPLIT, drives=drv)
        team = d.ts_keys["team_id"].to_numpy()[d.drv_off[:, 0]]
        np.testing.assert_allclose(d.drv_off_w[team == 2], 0.5)


def test_drive_channel_improves_recovery():
    gains = []
    for seed in range(3):
        games, eff, pre, o, de = synthetic(seed=seed)
        few = eff[eff["week"] <= 4]
        drv = synthetic_drives(few, o, de, seed)
        base = build_model_data(pre, few, games, [SEASON])
        with_drv = build_model_data(pre, few, games, [SEASON], drives=drv)
        o0, d0 = exact_last_states(base, hypers(base))
        o1, d1 = exact_last_states(with_drv, drive_hypers(with_drv))
        c0 = np.corrcoef(o0[:N_TEAMS], o)[0, 1] + np.corrcoef(d0[:N_TEAMS], de)[0, 1]
        c1 = np.corrcoef(o1[:N_TEAMS], o)[0, 1] + np.corrcoef(d1[:N_TEAMS], de)[0, 1]
        gains.append(c1 - c0)
    assert np.mean(gains) > 0.05 and min(gains) > 0


@pytest.mark.parametrize("units", [UNITS_ALL, UNITS_SPLIT])
def test_exact_solver_matches_mcmc_with_drives(units):
    games, eff, pre, o, de = synthetic(seed=1)
    drv = synthetic_drives(eff, o, de, 1)
    d = build_model_data(pre, eff, games, [SEASON], units=units, drives=drv)
    h = drive_hypers(d)
    exact_o, exact_d = exact_last_states(d, h)
    idata = sample(build_model(d, hypers=h, likelihood="normal"), draws=1000, tune=500, chains=2)
    post = idata.posterior.to_dataset()
    np.testing.assert_allclose(exact_o, post["O"].mean(("chain", "draw")).values[:, -1], atol=0.01)
    np.testing.assert_allclose(exact_d, post["D"].mean(("chain", "draw")).values[:, -1], atol=0.01)


def test_free_drive_model_has_drive_hyperparameters_and_default_has_none():
    games, eff, pre, o, de = synthetic()
    names = {v.name for v in build_model(build_model_data(pre, eff, games, [SEASON])).free_RVs}
    assert not any("ppd" in n for n in names)
    d = build_model_data(pre, eff, games, [SEASON], drives=synthetic_drives(eff, o, de))
    names = {v.name for v in build_model(d).free_RVs}
    assert {"alpha_ppd", "sigma_alpha_ppd", "lam_ppd", "hfa_ppd", "s_game_ppd", "s_drive_ppd", "z_alpha_ppd"} <= names


def test_drives_point_in_time_cutoff_fit():
    games, eff, pre, o, de = synthetic()
    drv = synthetic_drives(eff, o, de)
    inputs = Inputs(pre, eff, games, as_of=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime(), drives=drv)
    cutoff = season_periods(games, [SEASON])["cutoff"].iloc[4]
    assert (inputs.visible_drives(cutoff)["available_at"] < cutoff).all()
    d0 = build_model_data(pre, eff, games, [SEASON], drives=drv)
    post, d = fit_cutoff_exact(inputs, SEASON, cutoff, drive_hypers(d0))
    assert d.drives and len(d.drv_y) == (drv["available_at"] < cutoff).sum() < len(drv)
    assert set(d.drv_game) == set(drv.loc[drv["available_at"] < cutoff, "game_id"])
    no_drv = Inputs(pre, eff, games, as_of=inputs.as_of)
    with pytest.raises(ValueError):
        fit_cutoff_exact(no_drv, SEASON, cutoff, drive_hypers(d0))
    assert post.off.shape[1] == N_TEAMS + 1


def test_drive_hyperparameters_round_trip_and_default_dict_unchanged():
    games, eff, pre, o, de = synthetic()
    d = build_model_data(pre, eff, games, [SEASON])
    plain = hypers(d).to_dict()
    assert not any("ppd" in k for k in plain) and "drives" not in plain
    h = drive_hypers(d)
    back = Hyperparameters.from_dict(h.to_dict())
    assert back.drives and back.lam_ppd == TRUE_PPD["lam"] and back.s_drive_ppd == TRUE_PPD["s_drive"]


def test_drive_naming_order():
    drop = ("talent_qb",)
    name = hyper_name("normal", 2023, UNITS_SPLIT, conference=True, projection=True, drop_features=drop, drives=True)
    assert name.startswith("normal_split_conf_proj_drv_drop-") and name.endswith("_2019_2022")
    assert draws_name(2024, UNITS_SPLIT, conference=True, drives=True) == "cfb_team_week_draws_split_conf_drv_2024"
    assert model_version(UNITS_SPLIT, conference=True, drives=True) == "dyn-eff-split-conf-drv-1"
    assert hyper_name("normal", 2023, UNITS_SPLIT, conference=True) == "normal_split_conf_2019_2022"
    ind = hyper_name("normal", 2023, UNITS_SPLIT, conference=True, projection=True, drop_features=drop, drives=True,
                     independents_separate=True)
    assert ind.startswith("normal_split_conf_ind_proj_drv_drop-")
