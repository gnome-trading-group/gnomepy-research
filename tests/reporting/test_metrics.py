"""Unit tests for gnomepy_research.reporting.metrics — no JVM needed.

All tests use synthetic in-memory DataFrames that mirror the schema of
``BacktestResults.execution_records_df()`` and ``market_records_df()``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gnomepy_research.reporting import build_curves, compute_sharpe


# ---------------------------------------------------------------------------
# Helpers to build synthetic DataFrames
# ---------------------------------------------------------------------------

def _ts(offset_ns: int) -> pd.Timestamp:
    """Return a Timestamp at a given nanosecond offset from a fixed origin."""
    return pd.Timestamp("2024-01-15 10:00:00") + pd.Timedelta(nanoseconds=offset_ns)


def _market_df(rows: list[dict]) -> pd.DataFrame:
    """Build a market_records_df-shaped DataFrame.

    Each row dict should contain: ts (int ns offset), exchange_id, security_id,
    mid_price, and optionally best_bid_price / best_ask_price.
    """
    records = []
    for r in rows:
        records.append({
            "exchange_id": r["exchange_id"],
            "security_id": r["security_id"],
            "mid_price": r["mid_price"],
            "best_bid_price": r.get("best_bid_price", r["mid_price"] - 0.5),
            "best_ask_price": r.get("best_ask_price", r["mid_price"] + 0.5),
        })
    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex([_ts(r["ts"]) for r in rows], name="timestamp")
    return df


def _exec_df(rows: list[dict]) -> pd.DataFrame:
    """Build an execution_records_df-shaped DataFrame.

    Each row dict should contain: ts (int ns offset), exchange_id,
    security_id, side ("BID" or "ASK"), filled_qty, fill_price, fee.
    """
    records = []
    for r in rows:
        records.append({
            "exchange_id": r["exchange_id"],
            "security_id": r["security_id"],
            "side": r["side"],
            "filled_qty": r["filled_qty"],
            "fill_price": r["fill_price"],
            "fee": r.get("fee", 0.0),
        })
    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex([_ts(r["ts"]) for r in rows], name="timestamp_event")
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuyThenSell:
    """Buy 1 @ 100, sell 1 @ 110 → realized profit of 10 minus fees."""

    @pytest.fixture()
    def curves(self):
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 105.0},
            {"ts": 200, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
            {"ts": 300, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
        ])
        execs = _exec_df([
            {"ts": 50,  "exchange_id": 1, "security_id": 1, "side": "BID",
             "filled_qty": 1.0, "fill_price": 100.0, "fee": 0.5},
            {"ts": 250, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "filled_qty": 1.0, "fill_price": 110.0, "fee": 0.5},
        ])
        return build_curves(mkt, execs)

    def test_final_pnl(self, curves):
        # profit = (110 - 100)*1 = 10, fees = 1.0, net = 9.0
        assert curves.pnl.iloc[-1] == pytest.approx(9.0)

    def test_position_returns_to_zero(self, curves):
        assert curves.position.iloc[-1] == pytest.approx(0.0)

    def test_fees(self, curves):
        assert curves.fees.iloc[-1] == pytest.approx(1.0)

    def test_volume(self, curves):
        assert curves.volume.iloc[-1] == pytest.approx(2.0)

    def test_fill_count(self, curves):
        assert curves.fill_count == 2

    def test_pnl_moves_with_price(self, curves):
        # At ts=100 (mid=105), position=1 bought at 100, fee=0.5
        # cash = -100 - 0.5 = -100.5, pnl = -100.5 + 1*105 = 4.5
        pnl_at_100 = curves.pnl.iloc[1]
        assert pnl_at_100 == pytest.approx(4.5)


class TestFlipThroughZero:
    """Long 1, then sell 2 → net short 1. Cash + inventory stays consistent."""

    @pytest.fixture()
    def curves(self):
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 200, "exchange_id": 1, "security_id": 1, "mid_price": 95.0},
        ])
        execs = _exec_df([
            {"ts": 50,  "exchange_id": 1, "security_id": 1, "side": "BID",
             "filled_qty": 1.0, "fill_price": 100.0, "fee": 0.0},
            {"ts": 150, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "filled_qty": 2.0, "fill_price": 100.0, "fee": 0.0},
        ])
        return build_curves(mkt, execs)

    def test_position_flipped(self, curves):
        assert curves.position.iloc[-1] == pytest.approx(-1.0)

    def test_pnl_at_end(self, curves):
        # cash after buy:  -100
        # cash after sell:  -100 + 200 = 100
        # position = -1, mid = 95 → pnl = 100 + (-1)*95 = 5
        assert curves.pnl.iloc[-1] == pytest.approx(5.0)


class TestMultipleSymbols:
    """Two symbols traded independently — curves stay independent, totals sum."""

    @pytest.fixture()
    def curves(self):
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 0,   "exchange_id": 2, "security_id": 2, "mid_price": 50.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
            {"ts": 100, "exchange_id": 2, "security_id": 2, "mid_price": 55.0},
        ])
        execs = _exec_df([
            {"ts": 50, "exchange_id": 1, "security_id": 1, "side": "BID",
             "filled_qty": 1.0, "fill_price": 100.0, "fee": 0.0},
            {"ts": 50, "exchange_id": 2, "security_id": 2, "side": "BID",
             "filled_qty": 2.0, "fill_price": 50.0, "fee": 0.0},
        ])
        return build_curves(mkt, execs)

    def test_position_per_symbol(self, curves):
        assert len(curves.position_by_symbol.columns) == 2

    def test_pnl_sums_correctly(self, curves):
        # sym1: cash=-100, pos=1, mid=110 → 10
        # sym2: cash=-100, pos=2, mid=55  → 10
        assert curves.pnl.iloc[-1] == pytest.approx(20.0)

    def test_pnl_by_symbol(self, curves):
        # Each symbol contributes 10.
        assert curves.pnl_by_symbol[(1, 1)].iloc[-1] == pytest.approx(10.0)
        assert curves.pnl_by_symbol[(2, 2)].iloc[-1] == pytest.approx(10.0)

    def test_by_symbol_sums_to_total(self, curves):
        total_from_parts = curves.pnl_by_symbol.sum(axis=1)
        pd.testing.assert_series_equal(
            total_from_parts.reset_index(drop=True),
            curves.pnl.reset_index(drop=True),
        )


class TestEmptyFills:
    """No fills at all — curves should be all zeros, no crashes."""

    @pytest.fixture()
    def curves(self):
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 105.0},
        ])
        execs = _exec_df([])
        return build_curves(mkt, execs)

    def test_pnl_zero(self, curves):
        assert (curves.pnl == 0.0).all()

    def test_position_zero(self, curves):
        assert (curves.position == 0.0).all()

    def test_fill_count_zero(self, curves):
        assert curves.fill_count == 0


class TestDuplicateTimestamps:
    """Duplicate timestamps in market data must not inflate PnL."""

    @pytest.fixture()
    def curves(self):
        # 3 market records at ts=0, 3 at ts=100 — all same symbol.
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.5},
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 101.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 110.5},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 111.0},
        ])
        execs = _exec_df([
            {"ts": 50, "exchange_id": 1, "security_id": 1, "side": "BID",
             "filled_qty": 1.0, "fill_price": 100.0, "fee": 0.0},
        ])
        return build_curves(mkt, execs)

    def test_no_data_loss(self, curves):
        # Output should have same number of rows as market_df (6).
        assert len(curves.pnl) == 6

    def test_pnl_not_multiplied(self, curves):
        # At ts=100 with mid=111 (last dup): cash=-100, pos=1 → pnl=11.
        # Must NOT be 3x that.
        assert curves.pnl.iloc[-1] == pytest.approx(11.0)

    def test_pnl_varies_across_dups(self, curves):
        # The 3 records at ts=100 have mid 110, 110.5, 111 → distinct PnL.
        assert curves.pnl.iloc[3] == pytest.approx(10.0)
        assert curves.pnl.iloc[4] == pytest.approx(10.5)
        assert curves.pnl.iloc[5] == pytest.approx(11.0)


class TestEmptyMarket:
    """No market data — everything should be empty, no crashes."""

    def test_no_crash(self):
        mkt = _market_df([])
        execs = _exec_df([
            {"ts": 50, "exchange_id": 1, "security_id": 1, "side": "BID",
             "filled_qty": 1.0, "fill_price": 100.0, "fee": 0.0},
        ])
        curves = build_curves(mkt, execs)
        assert curves.pnl.empty or (curves.pnl == 0.0).all()
        assert curves.fill_count == 0


class TestComputeSharpe:
    """Sharpe ratio and stability metrics."""

    def test_positive_sharpe(self):
        # Rising PnL with noise → positive Sharpe.
        import numpy as np
        rng = np.random.RandomState(42)
        idx = pd.date_range("2024-01-01", periods=100, freq="10s")
        returns = 0.5 + rng.normal(0, 1.0, 100)  # positive drift + noise
        pnl = pd.Series(np.cumsum(returns), index=idx, dtype=float)
        result = compute_sharpe(pnl, bar="10s", n_buckets=5)
        assert result["sharpe"] > 0
        assert result["sortino"] > 0
        assert result["n_bars"] > 0
        assert len(result["sharpe_per_bucket"]) == 5

    def test_negative_sharpe(self):
        # Falling PnL with noise → negative Sharpe.
        import numpy as np
        rng = np.random.RandomState(42)
        idx = pd.date_range("2024-01-01", periods=100, freq="10s")
        returns = -1.0 + rng.normal(0, 0.5, 100)  # negative drift + noise
        pnl = pd.Series(np.cumsum(returns), index=idx, dtype=float)
        result = compute_sharpe(pnl, bar="10s")
        assert result["sharpe"] < 0

    def test_flat_pnl(self):
        # Flat PnL → zero std → Sharpe = 0.
        idx = pd.date_range("2024-01-01", periods=100, freq="10s")
        pnl = pd.Series(5.0, index=idx, dtype=float)
        result = compute_sharpe(pnl, bar="10s")
        assert result["sharpe"] == 0.0

    def test_empty_pnl(self):
        result = compute_sharpe(pd.Series(dtype=float), bar="10s")
        assert result["sharpe"] == 0.0
        assert result["n_bars"] == 0

    def test_stability_all_positive(self):
        # Strong positive drift + small noise → all buckets positive.
        import numpy as np
        rng = np.random.RandomState(42)
        idx = pd.date_range("2024-01-01", periods=500, freq="10s")
        returns = 2.0 + rng.normal(0, 0.3, 500)
        pnl = pd.Series(np.cumsum(returns), index=idx, dtype=float)
        result = compute_sharpe(pnl, bar="10s", n_buckets=5)
        assert result["pct_positive_buckets"] == 1.0
        assert result["sharpe_min"] > 0
