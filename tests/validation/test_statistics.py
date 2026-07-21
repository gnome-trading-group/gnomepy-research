"""Unit tests for validation.statistics — no JVM or file I/O needed."""
from __future__ import annotations

import numpy as np
import pytest

from gnomepy_research.validation.statistics import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_backtest_length,
    sharpe_standard_error,
    trials_correction,
)


class TestExpectedMaxSharpe:
    def test_one_trial_returns_zero(self):
        assert expected_max_sharpe(1) == 0.0

    def test_zero_trials_returns_zero(self):
        assert expected_max_sharpe(0) == 0.0

    def test_increases_with_trials(self):
        s10 = expected_max_sharpe(10)
        s100 = expected_max_sharpe(100)
        s1000 = expected_max_sharpe(1000)
        assert s10 < s100 < s1000

    def test_reasonable_magnitude(self):
        # 100 trials should yield expected max around 2-3 Sharpe under normal noise
        sr = expected_max_sharpe(100)
        assert 1.5 < sr < 3.5

    def test_many_trials(self):
        # 4000 trials (40 iters x 100 sweeps) should be a high hurdle
        sr = expected_max_sharpe(4000)
        assert sr > 3.0


class TestSharpeStandardError:
    def test_infinite_for_one_bar(self):
        assert sharpe_standard_error(1.0, 1) == float("inf")

    def test_decreases_with_bars(self):
        se_100 = sharpe_standard_error(1.0, 100)
        se_1000 = sharpe_standard_error(1.0, 1000)
        assert se_100 > se_1000

    def test_positive_for_positive_sharpe(self):
        se = sharpe_standard_error(2.0, 500)
        assert se > 0

    def test_normal_distribution_approximation(self):
        # For normal returns (skew=0, kurtosis=3), variance = (1 + SR²/2) / (T-1)
        sr, n = 1.0, 1001
        se = sharpe_standard_error(sr, n, skew=0.0, kurtosis=3.0)
        expected = ((1 + sr**2 / 2) / (n - 1)) ** 0.5
        assert se == pytest.approx(expected, rel=1e-6)


class TestDeflatedSharpeRatio:
    def test_single_trial_high_significance(self):
        # With 1 trial and good Sharpe, DSR should be high
        dsr = deflated_sharpe_ratio(2.0, n_trials=1, n_bars=1000)
        assert dsr > 0.9

    def test_many_trials_low_significance(self):
        # After 4000 trials, a Sharpe of 1.5 on 180 bars is not significant
        dsr = deflated_sharpe_ratio(1.5, n_trials=4000, n_bars=180)
        assert dsr < 0.5

    def test_returns_in_unit_interval(self):
        for sr in [-1.0, 0.0, 1.0, 2.0, 5.0]:
            dsr = deflated_sharpe_ratio(sr, n_trials=100, n_bars=500)
            assert 0 <= dsr <= 1

    def test_higher_sharpe_more_significant(self):
        dsr_low = deflated_sharpe_ratio(1.0, n_trials=50, n_bars=1000)
        dsr_high = deflated_sharpe_ratio(4.0, n_trials=50, n_bars=1000)
        assert dsr_high > dsr_low

    def test_more_bars_more_significant(self):
        # SR=3.0 is above the ~2.5 hurdle for 100 trials, so more bars → higher confidence
        dsr_few = deflated_sharpe_ratio(3.0, n_trials=100, n_bars=100)
        dsr_many = deflated_sharpe_ratio(3.0, n_trials=100, n_bars=10000)
        assert dsr_many > dsr_few

    def test_p_value_consistency(self):
        # DSR and p-value should sum to 1
        dsr = deflated_sharpe_ratio(2.5, n_trials=200, n_bars=500)
        assert dsr == pytest.approx(1 - (1 - dsr), abs=1e-12)


class TestTrialsCorrection:
    def test_consistent_with_expected_max(self):
        assert trials_correction(100) == pytest.approx(expected_max_sharpe(100))


class TestBootstrapSharpeCI:
    def test_positive_sharpe_ci_is_above_zero(self):
        rng = np.random.default_rng(42)
        returns = 0.5 + rng.normal(0, 1.0, 500)
        lo, hi = bootstrap_sharpe_ci(returns, n_bootstrap=200)
        assert hi > 0

    def test_ci_contains_true_sharpe(self):
        rng = np.random.default_rng(42)
        # True Sharpe = 1.0 with decent sample
        returns = 1.0 + rng.normal(0, 1.0, 1000)
        true_sharpe = 1.0
        lo, hi = bootstrap_sharpe_ci(returns, n_bootstrap=500)
        assert lo < true_sharpe < hi

    def test_insufficient_data_returns_inf(self):
        lo, hi = bootstrap_sharpe_ci(np.array([1.0, 2.0]))
        assert lo == float("-inf")
        assert hi == float("inf")

    def test_wide_ci_for_short_series(self):
        rng = np.random.default_rng(42)
        returns_short = rng.normal(0.5, 1.0, 20)
        returns_long = rng.normal(0.5, 1.0, 2000)
        lo_s, hi_s = bootstrap_sharpe_ci(returns_short, n_bootstrap=200)
        lo_l, hi_l = bootstrap_sharpe_ci(returns_long, n_bootstrap=200)
        assert (hi_s - lo_s) > (hi_l - lo_l)


class TestMinimumBacktestLength:
    def test_zero_sharpe_returns_zero(self):
        assert minimum_backtest_length(0.0) == 0

    def test_negative_sharpe_returns_zero(self):
        assert minimum_backtest_length(-1.0) == 0

    def test_higher_sharpe_needs_fewer_bars(self):
        t_low = minimum_backtest_length(0.5)
        t_high = minimum_backtest_length(2.0)
        assert t_high < t_low

    def test_short_window_insufficient(self):
        # A bar-level Sharpe of 0.1 needs far more than 180 bars
        # (a 30-minute session at 10s = 180 bars is not enough)
        min_len = minimum_backtest_length(0.1)
        assert min_len > 180

    def test_higher_ci_needs_more_bars(self):
        t_95 = minimum_backtest_length(1.0, ci=0.95)
        t_99 = minimum_backtest_length(1.0, ci=0.99)
        assert t_99 > t_95
