from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.efficiency.data import UNCERTAINTY_FLAGS, UNITS_ALL, UNITS_SPLIT, build_model_data
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import Hyperparameters
from gnomepy_research.models.cfb.efficiency.gaussian_solver import fit_cutoff_exact, solve
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name, hyper_name, model_version
from gnomepy_research.models.cfb.efficiency.projection import (
    def_continuity,
    feature_names,
    fit_unit,
    project,
    unit_ridge_ratings,
)
from gnomepy_research.models.cfb.features.source import SourceData
from tests.sessions.test_cfb_win_probability_models import SEASON, N_TEAMS, hypers, synthetic


def planted_unit_eff(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 12
    po, pd_, ro, rd = (rng.normal(0, 0.15, n) for _ in range(4))
    rows, gid = [], 0
    for _ in range(14):
        order = rng.permutation(n)
        for a, b in zip(order[::2], order[1::2]):
            gid += 1
            for off, de, venue in ((a, b, 1), (b, a, -1)):
                rows.append({"game_id": gid, "team_id": off + 1, "opponent_id": de + 1, "team_classification": "fbs",
                             "opponent_classification": "fbs", "venue": venue, "usable": True,
                             "plays": 65, "off_ppa": 9.9, "pass_plays": 30, "rush_plays": 35,
                             "off_pass_ppa": 0.2 + po[off] - pd_[de] + rng.normal(0, 0.05),
                             "off_rush_ppa": 0.1 + ro[off] - rd[de] + rng.normal(0, 0.05)})
    return pd.DataFrame(rows), po, pd_, ro, rd


def test_unit_ridge_recovers_planted_unit_ratings():
    eff, po, pd_, ro, rd = planted_unit_eff()
    p = unit_ridge_ratings(eff, "pass").set_index("team_id").loc[np.arange(1, 13)]
    r = unit_ridge_ratings(eff, "rush").set_index("team_id").loc[np.arange(1, 13)]
    assert np.corrcoef(p["off_rating"], po)[0, 1] > 0.9
    assert np.corrcoef(p["def_rating"], pd_)[0, 1] > 0.9
    assert np.corrcoef(r["off_rating"], ro)[0, 1] > 0.9
    assert np.corrcoef(r["def_rating"], rd)[0, 1] > 0.9


def test_unit_ridge_skips_rows_without_unit_plays():
    eff, *_ = planted_unit_eff()
    bad = eff.iloc[:1].assign(pass_plays=0, off_pass_ppa=50.0)
    a = unit_ridge_ratings(eff, "pass")
    b = unit_ridge_ratings(pd.concat([eff, bad], ignore_index=True), "pass")
    np.testing.assert_allclose(a["off_rating"], b["off_rating"])


def roster_rows(season: int, team_id: int, players: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame({"season": season, "team_id": team_id, "player_id": [p for p, _ in players],
                         "position": [pos for _, pos in players], "available_at": pd.Timestamp("2020-01-01", tz="UTC")})


def test_def_continuity_hand_built():
    prev = roster_rows(2023, 1, [("a", "LB"), ("b", "CB"), ("c", "DT"), ("q", "QB")])
    cur = roster_rows(2024, 1, [("a", "LB"), ("q", "QB"), ("z", "S")])
    moved = roster_rows(2024, 2, [("b", "CB")])
    recruits = pd.DataFrame({"athlete_id": ["a", "b", "c"], "rating": [0.98, 0.90, 0.80]})
    src = SourceData(*([pd.DataFrame()] * 5), recruits, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                     pd.concat([prev, cur, moved], ignore_index=True))
    out = def_continuity(src, 2024)
    assert out.loc[1] == pytest.approx(0.98 / (0.98 + 0.90 + 0.80))


def proj_table(seed: int = 0, slope: float = 1.0) -> pd.DataFrame:
    """Planted: pass offense carries last season's rating forward in proportion to returning passing share."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(2019, 2025):
        n = 120
        t = pd.DataFrame({"season": s, "team_id": np.arange(n)})
        for c in ("returning_pct_passing_ppa", "returning_pct_receiving_ppa", "returning_pct_rushing_ppa",
                  "def_continuity"):
            t[c] = rng.uniform(0, 1, n)
        for c in ("qb_prior_ppa", "portal_in_stars", "portal_out_stars", "talent", "blue_chip_ratio",
                  "conf_recruiting", "conf_prior_off", "conf_talent_offense", "conf_prior_def", "conf_talent_defense"):
            t[c] = rng.normal(size=n)
        t["qb_returning_starter"] = rng.uniform(size=n) < 0.5
        t["coach_new"] = rng.uniform(size=n) < 0.2
        for u in ("pass", "rush"):
            for sd in ("off", "def"):
                t[f"last_{u}_{sd}"] = rng.normal(0, 0.15, n)
                t[f"target_{u}_{sd}"] = 0.3 * t[f"last_{u}_{sd}"] + rng.normal(0, 0.05, n)
        t["target_pass_off"] = slope * t["last_pass_off"] * t["returning_pct_passing_ppa"] + rng.normal(0, 0.02, n)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def test_planted_interaction_sign_recovered():
    table = proj_table()
    m, _, _ = fit_unit(table, 2024, "pass", "off")
    coef = dict(zip(feature_names("pass", "off"), m.coef_))
    inter = coef["last_pass_off*returning_pct_passing_ppa"]
    assert inter > 0.03
    assert inter > abs(coef["last_pass_off*returning_pct_receiving_ppa"]) * 3
    neg, _, _ = fit_unit(proj_table(slope=-1.0), 2024, "pass", "off")
    assert dict(zip(feature_names("pass", "off"), neg.coef_))["last_pass_off*returning_pct_passing_ppa"] < -0.03


def test_rolling_origin_ignores_target_season_and_later_targets():
    table = proj_table()
    a = project(table, 2022)
    pert = table.copy()
    later = pert["season"] >= 2022
    for c in [c for c in pert.columns if c.startswith("target_")]:
        pert.loc[later, c] = 99.0
    b = project(pert, 2022)
    pd.testing.assert_frame_equal(a, b)


def test_projection_without_training_pairs_is_nan_and_sd_widens_with_low_share():
    table = proj_table()
    assert project(table, 2019)["proj"].isna().all()
    p = project(table, 2021)
    po = p[(p["unit"] == "rush") & (p["side"] == "off")]
    assert po["proj"].notna().all()
    assert np.corrcoef(po["returning_share"], po["proj_sd"])[0, 1] == pytest.approx(-1.0)
    assert set(p.columns) >= {"season", "team_id", "unit", "side", "proj", "proj_sd", "returning_share"}


def synthetic_projection(pre: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for t in pre["team_id"]:
        for u in ("pass", "rush"):
            for sd in ("off", "def"):
                share = 0.2 if t == 1 else 0.8
                rows.append({"season": SEASON, "team_id": t, "unit": u, "side": sd, "proj": rng.normal(0, 0.1),
                             "proj_sd": 0.1, "returning_share": share})
    return pd.DataFrame(rows)


class TestProjectionModelData:
    def test_default_unchanged(self):
        games, eff, pre, _, _ = synthetic()
        for units in (UNITS_ALL, UNITS_SPLIT):
            a = build_model_data(pre, eff, games, [SEASON], units=units)
            b = build_model_data(pre, eff, games, [SEASON], units=units, projection=None)
            np.testing.assert_array_equal(a.x_off, b.x_off)
            np.testing.assert_array_equal(a.z_unc, b.z_unc)
            assert a.off_names == b.off_names and a.z_unc.shape[1] == len(UNCERTAINTY_FLAGS)
            assert not a.projection

    def test_proj_column_per_unit_block(self):
        games, eff, pre, _, _ = synthetic()
        proj = synthetic_projection(pre)
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, projection=proj)
        base = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
        assert d.projection
        assert len(d.off_names) == len(base.off_names) + 2
        assert "pass:proj" in d.off_names and "rush:proj" in d.def_names
        keys = d.ts_keys
        for u in ("pass", "rush"):
            for side, x, names in (("off", d.x_off, d.off_names), ("def", d.x_def, d.def_names)):
                rows = ((keys["unit"] == u) & (keys["team_id"] != -1)).to_numpy()
                col = x[rows, names.index(f"{u}:proj")]
                raw = proj[(proj["unit"] == u) & (proj["side"] == side)].set_index("team_id")["proj"]
                raw = raw.loc[keys.loc[rows, "team_id"]].to_numpy()
                np.testing.assert_allclose(col, (raw - raw.mean()) / raw.std(ddof=1))
                other = "rush" if u == "pass" else "pass"
                assert (x[rows, names.index(f"{other}:proj")] == 0).all()

    def test_low_continuity_flag(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, projection=synthetic_projection(pre))
        assert d.z_unc.shape[1] == len(UNCERTAINTY_FLAGS) + 1
        flag = d.z_unc[:, -1]
        keys = d.ts_keys
        assert (flag[(keys["team_id"] == 1).to_numpy()] == 1).all()
        assert (flag[keys["team_id"].between(2, N_TEAMS).to_numpy()] == 0).all()
        assert (flag[(keys["team_id"] == -1).to_numpy()] == 1).all()

    def test_projection_requires_split_units(self):
        games, eff, pre, _, _ = synthetic()
        with pytest.raises(ValueError):
            build_model_data(pre, eff, games, [SEASON], projection=synthetic_projection(pre))


def proj_hypers(d) -> Hyperparameters:
    h = hypers(d)
    h.likelihood, h.projection = "normal", True
    return h


def test_projection_hypers_gamma_shape_solve_and_round_trip():
    games, eff, pre, _, _ = synthetic()
    proj = synthetic_projection(pre)
    d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, projection=proj)
    h = proj_hypers(d)
    assert len(h.gamma) == 4 and len(h.beta_off) == len(d.off_names)
    solve(d, h)
    back = Hyperparameters.from_dict(json.loads(json.dumps(h.to_dict())))
    assert back.projection and len(back.gamma) == 4
    plain = hypers(build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT))
    assert "projection" not in plain.to_dict()
    cutoff = pd.Timestamp("2024-09-20", tz="UTC")
    inputs = Inputs(pre, eff, games.assign(home_points=0, away_points=0), cutoff, projection=proj)
    post, data = fit_cutoff_exact(inputs, SEASON, d.periods["cutoff"].iloc[3], h, n_draws=50)
    assert data.projection and post.off.shape[1] == len(d.ts_keys)
    with pytest.raises(ValueError):
        fit_cutoff_exact(Inputs(pre, eff, games, cutoff), SEASON, d.periods["cutoff"].iloc[3], h, n_draws=50)


def test_projection_naming():
    assert hyper_name("normal", 2023, UNITS_SPLIT, conference=True, projection=True) == "normal_split_conf_proj_2019_2022"
    assert draws_name(2024, UNITS_SPLIT, conference=True, projection=True) == "cfb_team_week_draws_split_conf_proj_2024"
    assert model_version(UNITS_SPLIT, conference=True, projection=True) == "dyn-eff-split-conf-proj-1"
    assert hyper_name("normal", 2023, UNITS_SPLIT, conference=True) == "normal_split_conf_2019_2022"
