from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import (
    CALIBRATOR_SEASONS, MODEL_CONFIGS, apply_calibrator, load_calibrator)
from gnomepy_research.sessions.cfb_win_probability.models.tail_calibrator import (
    TailCalibrator, bootstrap_ci, fit_tail_calibrator)


def synthetic(n=6000, b=0.08, c=0.0, post_frac=0.0, seed=0):
    """Truth: logit(p_true) = z + b z|z| + c post z, so the model's p = sigmoid(z) is under-confident in the tails."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.6, n)
    post = (rng.random(n) < post_frac).astype(float)
    eta = z + b * z * np.abs(z) + c * post * z
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return 1 / (1 + np.exp(-z)), y, post


class TestTailCalibrator:
    def test_swap_symmetry(self):
        cal = TailCalibrator(a=1.1, b=0.07, c=-0.3)
        p = np.array([0.03, 0.2, 0.5, 0.71, 0.97])
        for post in (np.zeros(5), np.ones(5)):
            assert np.allclose(cal.apply(1 - p, post), 1 - cal.apply(p, post))
        assert cal.apply(np.array([0.5]), np.array([1.0]))[0] == pytest.approx(0.5)

    def test_identity_at_default(self):
        p = np.array([0.1, 0.6, 0.9])
        assert np.allclose(TailCalibrator(1.0, 0.0, 0.0).apply(p, np.zeros(3)), p)

    def test_recovers_positive_tail_bend(self):
        p, y, post = synthetic(b=0.08)
        cal = fit_tail_calibrator(p, y, post)
        assert cal.b > 0.03
        assert cal.apply(np.array([0.9]), np.array([0.0]))[0] > 0.9

    def test_well_calibrated_input_stays_near_identity(self):
        p, y, post = synthetic(n=20000, b=0.0, post_frac=0.2)
        cal = fit_tail_calibrator(p, y, post)
        grid = np.linspace(0.02, 0.98, 49)
        for flag in (0.0, 1.0):
            assert np.abs(cal.apply(grid, np.full(49, flag)) - grid).max() < 0.03

    def test_postseason_term_only_on_postseason_rows(self):
        cal = TailCalibrator(a=1.0, b=0.0, c=-0.5)
        p = np.array([0.8, 0.8, 0.3, 0.3])
        post = np.array([0.0, 1.0, 0.0, 1.0])
        out = cal.apply(p, post)
        assert out[0] == pytest.approx(0.8) and out[2] == pytest.approx(0.3)
        assert 0.5 < out[1] < 0.8 and 0.3 < out[3] < 0.5

    def test_recovers_postseason_shrink(self):
        p, y, post = synthetic(n=12000, b=0.0, c=-0.4, post_frac=0.3, seed=2)
        cal = fit_tail_calibrator(p, y, post)
        assert cal.c < -0.2

    def test_bootstrap_ci_brackets_estimate(self):
        p, y, post = synthetic(n=2000, b=0.08, post_frac=0.1, seed=3)
        cal = fit_tail_calibrator(p, y, post)
        ci = bootstrap_ci(p, y, post, n=40, seed=0)
        for k in ("a", "b", "c"):
            assert ci[k][0] <= getattr(cal, k) <= ci[k][1]


def preds_frame(seasons, n=400, seed=0):
    rng = np.random.default_rng(seed)
    p, y, _ = synthetic(n=n, b=0.08, seed=seed)
    return pd.DataFrame({"game_id": np.arange(n), "season": rng.choice(seasons, n), "week": 5,
                         "season_type": np.where(rng.random(n) < 0.1, "postseason", "regular"),
                         "home_win": y, "p_home": p, "p_bayes": p, "p_bayes_lo": p * 0.9, "p_bayes_hi": p + (1 - p) * 0.1})


class TestKc:
    def test_config_is_k_plus_tail(self):
        kc = MODEL_CONFIGS["Kc"]
        assert kc["calibrator"] == "tail" and kc["calibrator_base"] == "K"
        assert {k: v for k, v in kc.items() if k not in ("calibrator", "calibrator_base")} == MODEL_CONFIGS["K"]

    @pytest.mark.parametrize("seasons", [[2022], [2021, 2023], [2020], [2022, 2023, 2024]])
    def test_refuses_overlapping_test_seasons(self, tmp_path, seasons):
        preds_frame(CALIBRATOR_SEASONS).to_parquet(tmp_path / "predictions_2021_2022_K.parquet")
        with pytest.raises(SystemExit, match="2021"):
            load_calibrator(MODEL_CONFIGS["Kc"], seasons, results=tmp_path)

    def test_raises_when_oos_predictions_missing(self, tmp_path):
        with pytest.raises(SystemExit, match="predictions_2021_2022_K.parquet"):
            load_calibrator(MODEL_CONFIGS["Kc"], [2023, 2024], results=tmp_path)

    def test_fits_on_2021_2022_file(self, tmp_path):
        preds_frame(CALIBRATOR_SEASONS).to_parquet(tmp_path / "predictions_2021_2022_K.parquet")
        cal, info = load_calibrator(MODEL_CONFIGS["Kc"], [2023, 2024], results=tmp_path, n_boot=10)
        assert isinstance(cal, TailCalibrator)
        assert info["n"] == 400 and info["source"] == "predictions_2021_2022_K.parquet"
        assert set(info["ci"]) == {"a", "b", "c"}

    def test_rejects_file_with_wrong_seasons(self, tmp_path):
        preds_frame([2021, 2023]).to_parquet(tmp_path / "predictions_2021_2022_K.parquet")
        with pytest.raises(SystemExit, match="seasons"):
            load_calibrator(MODEL_CONFIGS["Kc"], [2024], results=tmp_path)

    def test_uncalibrated_model_needs_no_file(self, tmp_path):
        assert load_calibrator(MODEL_CONFIGS["K"], [2021], results=tmp_path) == (None, None)

    def test_apply_calibrator_updates_all_probability_columns(self):
        df = preds_frame([2023])
        cal = TailCalibrator(a=1.0, b=0.1, c=-0.5)
        out = apply_calibrator(df, cal)
        post = (df["season_type"] == "postseason").to_numpy().astype(float)
        for col in ("p_home", "p_bayes", "p_bayes_lo", "p_bayes_hi"):
            assert np.allclose(out[col], cal.apply(df[col].to_numpy(), post))
        assert np.allclose(out["p_home"], out["p_bayes"])
        assert np.allclose(df["p_home"], preds_frame([2023])["p_home"])
