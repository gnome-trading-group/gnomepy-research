"""Unit tests for rolling performance analysis — no JVM or file I/O needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.reporting.backtest.rolling_performance import (
    compute_alpha_decay,
    compute_rolling_sharpe,
    detect_regimes,
    pnl_by_regime,
)


def _pnl(returns, freq="10s", start="2026-01-01"):
    idx = pd.date_range(start, periods=len(returns), freq=freq)
    return pd.Series(np.cumsum(returns), index=idx, dtype=float)


def _market_df(n=200, start="2026-01-01", with_spread=True):
    idx = pd.date_range(start, periods=n, freq="1s")
    mid = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.1, n))
    df = pd.DataFrame({"mid_price": mid}, index=idx)
    df["exchange_id"] = 1
    df["security_id"] = 1
    if with_spread:
        df["bid_price_0"] = mid - 0.5
        df["ask_price_0"] = mid + 0.5
    return df


class TestComputeRollingSharpe:
    def test_positive_drift_produces_positive_rolling_sharpe(self):
        rng = np.random.default_rng(42)
        returns = 0.5 + rng.normal(0, 1.0, 500)
        pnl = _pnl(returns)
        rolling = compute_rolling_sharpe(pnl, window="1min", bar="10s")
        assert not rolling.empty
        assert rolling.dropna().mean() > 0

    def test_negative_drift_produces_negative_rolling_sharpe(self):
        rng = np.random.default_rng(42)
        returns = -0.5 + rng.normal(0, 1.0, 500)
        pnl = _pnl(returns)
        rolling = compute_rolling_sharpe(pnl, window="1min", bar="10s")
        assert not rolling.empty
        assert rolling.dropna().mean() < 0

    def test_empty_pnl_returns_empty(self):
        result = compute_rolling_sharpe(pd.Series(dtype=float))
        assert result.empty

    def test_output_aligned_to_bar_index(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.3, 1.0, 200)
        pnl = _pnl(returns)
        rolling = compute_rolling_sharpe(pnl, window="30s", bar="10s")
        assert rolling.index.dtype.kind == "M"


class TestComputeAlphaDecay:
    def test_returns_one_value_per_window(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.3, 1.0, 360)
        pnl = _pnl(returns)
        decay = compute_alpha_decay(pnl, window="1min", bar="10s")
        assert len(decay) == 60

    def test_empty_pnl_returns_empty(self):
        result = compute_alpha_decay(pd.Series(dtype=float))
        assert result.empty

    def test_declining_edge_visible(self):
        rng = np.random.default_rng(42)
        strong = 2.0 + rng.normal(0, 1.0, 180)
        weak = 0.1 + rng.normal(0, 1.0, 180)
        returns = np.concatenate([strong, weak])
        pnl = _pnl(returns)
        decay = compute_alpha_decay(pnl, window="1min", bar="10s")
        assert not decay.empty
        assert decay.iloc[0] > decay.iloc[-1]


class TestDetectRegimes:
    def test_returns_series_of_strings(self):
        mkt = _market_df(n=300)
        regimes = detect_regimes(mkt)
        assert not regimes.empty
        assert regimes.dtype == object

    def test_only_valid_labels(self):
        mkt = _market_df(n=300)
        regimes = detect_regimes(mkt)
        valid = {"low_vol_tight", "low_vol_wide", "high_vol_tight", "high_vol_wide"}
        assert set(regimes.unique()).issubset(valid)

    def test_produces_multiple_regimes(self):
        mkt = _market_df(n=600)
        regimes = detect_regimes(mkt)
        assert len(set(regimes.unique())) >= 2

    def test_empty_market_returns_empty(self):
        result = detect_regimes(pd.DataFrame())
        assert result.empty

    def test_no_spread_columns_still_works(self):
        mkt = _market_df(n=200, with_spread=False)
        regimes = detect_regimes(mkt)
        assert not regimes.empty


class TestPnlByRegime:
    def test_regime_pnls_sum_to_total(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.2, 1.0, 360)
        pnl = _pnl(returns)
        mkt = _market_df(n=3600, start="2026-01-01")
        regimes = detect_regimes(mkt)
        stats = pnl_by_regime(pnl, regimes)
        if stats:
            total = sum(r["final_pnl"] for r in stats.values())
            pnl_sum = float(pnl.resample("10s").last().dropna().diff().dropna().sum())
            assert abs(total - pnl_sum) < abs(pnl_sum) * 0.01 or abs(total - pnl_sum) < 1e-6

    def test_pct_time_sums_to_one(self):
        rng = np.random.default_rng(42)
        pnl = _pnl(rng.normal(0.2, 1.0, 360))
        mkt = _market_df(n=3600)
        regimes = detect_regimes(mkt)
        stats = pnl_by_regime(pnl, regimes)
        if stats:
            total_pct = sum(r["pct_time"] for r in stats.values())
            assert total_pct == pytest.approx(1.0, abs=0.05)

    def test_empty_inputs_return_empty(self):
        result = pnl_by_regime(pd.Series(dtype=float), pd.Series(dtype=str))
        assert result == {}
