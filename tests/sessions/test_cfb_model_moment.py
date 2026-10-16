from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import MODEL_CONFIGS
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import (
    N_PHASES, N_TIERS, STRETCH_TERM, GameDraws, MomentModel, fit_moment, matchup_tier, moment_design, moment_phase)

TERMS = ["pass_edge", "rush_edge", "venue", STRETCH_TERM]


def draws(pass_edges: list[list[float]], venue: float = 1.0) -> GameDraws:
    p = np.array(pass_edges)
    x = np.stack([p, np.zeros_like(p), np.full_like(p, venue), p * np.abs(p)], axis=-1)
    games = pd.DataFrame({"season": 2024, "week": 6, "season_type": "regular", "home_win": 1.0,
                          "home_conference": "SEC", "away_conference": "SEC", "home_team": "A", "away_team": "B"},
                         index=range(len(p)))
    return GameDraws(games=games, x=x, terms=TERMS)


def test_moment_design_puts_the_stretch_on_the_mean_edge():
    x, v = moment_design(draws([[0.1, 0.3, 0.1, 0.3], [0.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(x[0], [0.2, 0.0, 1.0, 0.04])     # per-draw average would be 0.05
    np.testing.assert_allclose(v, [0.01, 0.0])


def test_matchup_tier_and_phase():
    g = pd.DataFrame({
        "season": [2024, 2024, 2024, 2023, 2026, 2024, 2024],
        "home_conference": ["SEC", "SEC", "Sun Belt", "Pac-12", "Pac-12", "FBS Independents", "Big Ten"],
        "away_conference": ["SEC", "Sun Belt", "Mid-American", "Pac-12", "Mountain West", "Mid-American", "Big 12"],
        "home_team": ["A", "B", "C", "D", "Colorado State", "Notre Dame", "E"],
        "away_team": ["F", "G", "H", "I", "Boise State", "J", "K"],
        "week": [3, 5, 9, 10, 7, 1, 15], "season_type": ["regular"] * 6 + ["postseason"]})
    assert matchup_tier(g).tolist() == [0, 1, 2, 0, 2, 1, 0]
    assert moment_phase(g).tolist() == [0, 1, 1, 2, 1, 0, 2]


def test_moment_predict_limits_symmetry_and_interval():
    gd = draws([[0.1, 0.3, 0.1, 0.3]], venue=0.0)
    gamma = np.array([[8.0, 8.0, 0.3, 10.0]])
    eta = 8.0 * 0.2 + 10.0 * 0.04
    sharp = MomentModel(gamma=gamma, kappa=np.zeros((1, N_PHASES, N_TIERS))).predict(gd)
    np.testing.assert_allclose(sharp["p_home"], 1 / (1 + np.exp(-eta)))
    np.testing.assert_allclose(sharp[["p_lo", "p_hi"]].to_numpy().ravel(), np.repeat(sharp["p_home"].to_numpy(), 2))
    blurred = MomentModel(gamma=gamma, kappa=np.full((1, N_PHASES, N_TIERS), 20.0)).predict(gd)
    assert 0.5 < blurred["p_home"].iloc[0] < sharp["p_home"].iloc[0]
    assert blurred["p_lo"].iloc[0] < blurred["p_home"].iloc[0] < blurred["p_hi"].iloc[0]
    swapped = draws([[-0.1, -0.3, -0.1, -0.3]], venue=0.0)
    np.testing.assert_allclose(MomentModel(gamma=gamma, kappa=np.full((1, N_PHASES, N_TIERS), 20.0)).predict(swapped)["p_home"],
                               1 - blurred["p_home"])


def test_phase17_configs():
    assert MODEL_CONFIGS["Npm"] == {**MODEL_CONFIGS["Np"], "learner": "moment"}
    assert MODEL_CONFIGS["Npmt"]["moment_tier"] and MODEL_CONFIGS["Km"]["learner"] == "moment"
    assert "moment_tier" not in MODEL_CONFIGS["Npm"]
    assert MODEL_CONFIGS["Np21"] == {**MODEL_CONFIGS["Np"], "outcome_first": 2021}


def test_fit_moment_recovers_a_positive_edge_slope():
    rng = np.random.default_rng(0)
    g, s = 300, 20
    edge = rng.normal(0, 0.15, g)
    x_pass = edge[:, None] + rng.normal(0, 0.05, (g, s))
    x = np.stack([x_pass, np.zeros_like(x_pass), np.ones_like(x_pass), x_pass * np.abs(x_pass)], axis=-1)
    y = (rng.random(g) < 1 / (1 + np.exp(-8 * edge))).astype(float)
    games = pd.DataFrame({"season": 2024, "week": rng.integers(1, 14, g), "season_type": "regular", "home_win": y})
    model = fit_moment(GameDraws(games=games, x=x, terms=TERMS), draws=200, tune=200)
    assert model.gamma.shape[1] == len(TERMS) and model.kappa.shape[1:] == (N_PHASES, N_TIERS)
    assert model.gamma[:, 0].mean() > 2
