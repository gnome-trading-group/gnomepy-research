from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import MODEL_CONFIGS, draw_kwargs
from gnomepy_research.sessions.cfb_win_probability.models import margin_filter as mf
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import KF_SCALE, KF_TERM, PRIORS, game_draws
from tests.sessions import test_cfb_model_k as model_k
from tests.sessions.test_cfb_model_o import TestResultsTerm
from tests.sessions.test_cfb_win_probability_models import SEASON

K = pd.Timestamp("2024-09-01", tz="UTC")


def schedule(rows):
    """rows: (game_id, week, home, away, home_pts, away_pts, completed)."""
    return pd.DataFrame([{"game_id": g, "season": 2024, "week": w, "season_type": "regular", "kickoff": K + pd.Timedelta(days=7 * w),
                          "home_id": h, "away_id": a, "home_points": hp, "away_points": ap, "completed": c, "neutral_site": True}
                         for g, w, h, a, hp, ap, c in rows])


def test_kalman_update_matches_the_closed_form_and_never_sees_its_own_game():
    games = schedule([(1, 1, 1, 2, 10, 0, True), (2, 2, 1, 2, np.nan, np.nan, False)])
    out = mf.run_filter(games, {1: (0.0, 36.0), 2: (0.0, 36.0)}).set_index("game_id")
    assert out.loc[1, "kf_diff"] == 0.0 and out.loc[1, "kf_var"] == 72.0          # pre-game: priors only
    total = 36 + 36 + mf.SIGMA ** 2
    gain = 36 / total * 10
    np.testing.assert_allclose(out.loc[2, "kf_diff"], 2 * gain)                   # both teams moved by the same gain
    post_var = 36 - 36 ** 2 / total + mf.Q                                          # one week of drift added
    np.testing.assert_allclose(out.loc[2, "kf_var"], 2 * post_var)


def test_margin_cap_fcs_and_unplayed_games():
    games = schedule([(1, 1, 1, 2, 70, 0, True), (2, 2, 1, 9, 50, 3, True), (3, 3, 1, 2, np.nan, np.nan, False)])
    out = mf.run_filter(games, {1: (0.0, 36.0), 2: (0.0, 36.0)}).set_index("game_id")
    capped = 36 / (72 + mf.SIGMA ** 2) * mf.CAP * 2
    assert out.loc[2, "kf_diff"] == pytest.approx(capped / 2 - mf.FCS_MEAN)      # team 9 (no prior) sits at FCS_MEAN
    assert len(out) == 3 and np.isfinite(out.loc[3, "kf_diff"])


def test_prior_map_recovers_coefficients_and_uses_earlier_seasons_only():
    rng = np.random.default_rng(0)
    n = 400
    x = pd.DataFrame({"season": np.repeat([2022, 2023], n // 2), "team_id": np.arange(n), "s": rng.normal(0, 0.25, n),
                      "p4": rng.random(n) < 0.5})
    x["srs"] = 26 * x.s + 3 * x.p4 + rng.normal(0, 1, n) * np.where(x.p4, 6.0, 8.0)
    m = mf.fit_prior_map(x, tiered=True)
    assert m["a"] == pytest.approx(26, abs=2.5) and m["b"] == pytest.approx(3, abs=1.5)
    assert m["tau_g5"] > m["tau_p4"]
    flat = mf.fit_prior_map(x, tiered=False)
    assert flat["b"] == 0.0 and flat["tau_p4"] == flat["tau_g5"]
    srs = x.set_index(["season", "team_id"])["srs"]
    x2 = x.copy()
    x2.loc[x2.season == 2023, "srs"] = 1000.0                                    # poison the predicted season
    priors, used = mf.season_priors(x.drop(columns="srs"), x2.set_index(["season", "team_id"])["srs"], 2023, tiered=True)
    assert used["a"] == pytest.approx(mf.fit_prior_map(x[x.season == 2022], tiered=True)["a"])
    assert abs(np.mean([m for m, _ in priors.values()])) < 1e-9
    assert srs is not None


def test_kf_term_wiring_and_configs(tmp_path):
    model_k.TestPhaseTerms.write(tmp_path)
    kf = pd.DataFrame({"game_id": [1, 2, 3], "kf_diff": [5.0, -10.0, 0.0], "kf_var": 50.0})
    gd = game_draws(TestResultsTerm.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT, kf=kf)
    assert gd.terms[-1] == KF_TERM and PRIORS[KF_TERM] == (0.0, 1.0)
    got = dict(zip(gd.games["game_id"], gd.x[:, 0, -1]))
    assert got == {1: 5.0 / KF_SCALE, 2: -10.0 / KF_SCALE, 3: 0.0}
    assert MODEL_CONFIGS["Npvk"] == {**MODEL_CONFIGS["Npv"], "kf": "tier"}
    assert MODEL_CONFIGS["Npvk0"]["kf"] == "flat"
    assert draw_kwargs(MODEL_CONFIGS["Npvk"])["extensions"] == ("couple", "fcs", "finish", "prior")
