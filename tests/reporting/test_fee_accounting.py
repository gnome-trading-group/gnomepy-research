"""Fee accounting verification tests for build_curves — no JVM needed.

All fees are expressed as human-readable floats (fill_qty * fill_price * rate)
matching the final scaled values produced by BacktestResults.fills_df().
"""
from __future__ import annotations

import pandas as pd
import pytest

from gnomepy_research.reporting import build_curves


TAKER_RATE = 0.001      # 10 bps
MAKER_RATE = 0.0002     # 2 bps
MAKER_REBATE = -0.0001  # -1 bp


def _ts(offset_ns: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-15 10:00:00") + pd.Timedelta(nanoseconds=offset_ns)


def _market_df(rows: list[dict]) -> pd.DataFrame:
    records = [
        {
            "exchange_id": r["exchange_id"],
            "security_id": r["security_id"],
            "mid_price": r["mid_price"],
        }
        for r in rows
    ]
    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex([_ts(r["ts"]) for r in rows], name="timestamp")
    return df


def _exec_df(rows: list[dict]) -> pd.DataFrame:
    records = [
        {
            "exchange_id": r["exchange_id"],
            "security_id": r["security_id"],
            "side": r["side"],
            "fill_qty": r["fill_qty"],
            "fill_price": r["fill_price"],
            "fee": r.get("fee", 0.0),
        }
        for r in rows
    ]
    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex([_ts(r["ts"]) for r in rows], name="timestamp_event")
    return df


class TestTakerFeeRoundTrip:
    """Buy 1 @ 100 then sell 1 @ 110, both taker fills at 10 bps.

    fee_buy  = 1.0 * 100.0 * 0.001 = 0.10
    fee_sell = 1.0 * 110.0 * 0.001 = 0.11
    total_fees = 0.21
    gross_pnl  = 110 - 100 = 10.0
    net_pnl    = 10.0 - 0.21 = 9.79
    """

    @pytest.fixture()
    def curves(self):
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
            {"ts": 200, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
        ])
        execs = _exec_df([
            {"ts": 50,  "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 1.0, "fill_price": 100.0, "fee": 1.0 * 100.0 * TAKER_RATE},
            {"ts": 150, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "fill_qty": 1.0, "fill_price": 110.0, "fee": 1.0 * 110.0 * TAKER_RATE},
        ])
        return build_curves(mkt, execs)

    def test_final_pnl(self, curves):
        assert curves.pnl.iloc[-1] == pytest.approx(9.79)

    def test_total_fees(self, curves):
        assert curves.fees.iloc[-1] == pytest.approx(0.21)

    def test_position_flat(self, curves):
        assert curves.position.iloc[-1] == pytest.approx(0.0)


class TestMakerFeeRoundTrip:
    """Buy 1 @ 100 then sell 1 @ 110, both maker fills at 2 bps.

    fee_buy  = 1.0 * 100.0 * 0.0002 = 0.020
    fee_sell = 1.0 * 110.0 * 0.0002 = 0.022
    total_fees = 0.042
    net_pnl    = 10.0 - 0.042 = 9.958
    """

    @pytest.fixture()
    def curves(self):
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 200, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
        ])
        execs = _exec_df([
            {"ts": 50,  "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 1.0, "fill_price": 100.0, "fee": 1.0 * 100.0 * MAKER_RATE},
            {"ts": 150, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "fill_qty": 1.0, "fill_price": 110.0, "fee": 1.0 * 110.0 * MAKER_RATE},
        ])
        return build_curves(mkt, execs)

    def test_final_pnl(self, curves):
        assert curves.pnl.iloc[-1] == pytest.approx(9.958)

    def test_total_fees(self, curves):
        assert curves.fees.iloc[-1] == pytest.approx(0.042)

    def test_maker_fee_lower_than_taker(self, curves):
        # Maker fees should always be smaller than taker fees on same notional.
        assert curves.fees.iloc[-1] < 0.21


class TestMakerRebate:
    """Buy 1 @ 100 then sell 1 @ 110 with a maker rebate of -1 bp.

    fee_buy  = 1.0 * 100.0 * (-0.0001) = -0.010  (rebate received)
    fee_sell = 1.0 * 110.0 * (-0.0001) = -0.011
    total_fees = -0.021
    gross_pnl  = 10.0
    net_pnl    = 10.0 - (-0.021) = 10.021  (rebate adds to profit)
    """

    @pytest.fixture()
    def curves(self):
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 200, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
        ])
        execs = _exec_df([
            {"ts": 50,  "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 1.0, "fill_price": 100.0, "fee": 1.0 * 100.0 * MAKER_REBATE},
            {"ts": 150, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "fill_qty": 1.0, "fill_price": 110.0, "fee": 1.0 * 110.0 * MAKER_REBATE},
        ])
        return build_curves(mkt, execs)

    def test_rebate_boosts_pnl_above_gross(self, curves):
        assert curves.pnl.iloc[-1] == pytest.approx(10.021)

    def test_cum_fees_negative(self, curves):
        # Negative cumulative fees indicates net rebate received.
        assert curves.fees.iloc[-1] == pytest.approx(-0.021)


class TestPartialFillFeeAccumulation:
    """Single order filled in 4 equal partial fills at taker rate.

    Each fill: qty=0.25 @ 100.0, fee = 0.25 * 100.0 * 0.001 = 0.025
    After n fills: cum_fees = n * 0.025
    After 4 fills: total_fees = 0.10
    """

    @pytest.fixture()
    def curves(self):
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 200, "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 300, "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 400, "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
        ])
        fill_fee = 0.25 * 100.0 * TAKER_RATE  # 0.025
        execs = _exec_df([
            {"ts": 50,  "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 0.25, "fill_price": 100.0, "fee": fill_fee},
            {"ts": 150, "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 0.25, "fill_price": 100.0, "fee": fill_fee},
            {"ts": 250, "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 0.25, "fill_price": 100.0, "fee": fill_fee},
            {"ts": 350, "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 0.25, "fill_price": 100.0, "fee": fill_fee},
        ])
        return build_curves(mkt, execs)

    def test_total_fees_after_all_partials(self, curves):
        assert curves.fees.iloc[-1] == pytest.approx(0.10)

    def test_position_accumulated(self, curves):
        assert curves.position.iloc[-1] == pytest.approx(1.0)

    def test_fees_grow_monotonically(self, curves):
        # Fees should only increase as each partial fill lands.
        fees = curves.fees.tolist()
        assert fees == sorted(fees)

    def test_intermediate_fee_after_two_fills(self, curves):
        # After ts=200 (2 fills complete), cum_fees should be 0.05.
        assert curves.fees.iloc[2] == pytest.approx(0.05)


class TestMarketMakerSpreadCapture:
    """Market maker: buy at bid 99.0 (maker), sell at ask 101.0 (maker).

    Spread = 2.0, maker_rate = 0.0002
    fee_buy  = 1.0 * 99.0  * 0.0002 = 0.0198
    fee_sell = 1.0 * 101.0 * 0.0002 = 0.0202
    total_fees = 0.04
    gross_pnl  = 2.0
    net_pnl    = 2.0 - 0.04 = 1.96
    """

    @pytest.fixture()
    def curves(self):
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 200, "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
        ])
        execs = _exec_df([
            {"ts": 50,  "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 1.0, "fill_price": 99.0, "fee": 1.0 * 99.0 * MAKER_RATE},
            {"ts": 150, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "fill_qty": 1.0, "fill_price": 101.0, "fee": 1.0 * 101.0 * MAKER_RATE},
        ])
        return build_curves(mkt, execs)

    def test_net_pnl(self, curves):
        assert curves.pnl.iloc[-1] == pytest.approx(1.96)

    def test_total_fees(self, curves):
        assert curves.fees.iloc[-1] == pytest.approx(0.04)

    def test_net_pnl_less_than_spread(self, curves):
        assert curves.pnl.iloc[-1] < 2.0


class TestFeeRateConsistency:
    """Fee/notional ratio must equal the stated rate for every fill."""

    def test_taker_rate_constant_across_sizes(self):
        fills = [
            (0.1,  100.0),
            (0.5,  200.0),
            (2.0,  50.0),
            (10.0, 1000.0),
        ]
        for qty, price in fills:
            fee = qty * price * TAKER_RATE
            ratio = fee / (qty * price)
            assert ratio == pytest.approx(TAKER_RATE), (
                f"Rate mismatch for qty={qty}, price={price}: got {ratio}"
            )

    def test_maker_rate_constant_across_sizes(self):
        fills = [
            (0.1,  100.0),
            (1.0,  60000.0),
            (5.0,  0.50),
        ]
        for qty, price in fills:
            fee = qty * price * MAKER_RATE
            ratio = fee / (qty * price)
            assert ratio == pytest.approx(MAKER_RATE)

    def test_pnl_accounts_for_fee_correctly(self):
        """PnL = gross_pnl - total_fees regardless of fill size."""
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
            {"ts": 200, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
        ])
        qty = 3.0
        buy_price = 100.0
        sell_price = 110.0
        fee_buy = qty * buy_price * TAKER_RATE
        fee_sell = qty * sell_price * TAKER_RATE

        execs = _exec_df([
            {"ts": 50,  "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": qty, "fill_price": buy_price, "fee": fee_buy},
            {"ts": 150, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "fill_qty": qty, "fill_price": sell_price, "fee": fee_sell},
        ])
        curves = build_curves(mkt, execs)

        gross_pnl = qty * (sell_price - buy_price)
        total_fees = fee_buy + fee_sell
        assert curves.pnl.iloc[-1] == pytest.approx(gross_pnl - total_fees)


class TestHighFrequencyFeeAccumulation:
    """100 tiny buy fills then a single sell — verifies no floating-point drift.

    Each buy: qty=0.01 @ 100.0, fee = 0.01 * 100.0 * 0.001 = 0.001
    Total buy fees = 100 * 0.001 = 0.10
    Sell: qty=1.0 @ 110.0, fee = 1.0 * 110.0 * 0.001 = 0.11
    Total fees = 0.21
    Gross PnL = (110 - 100) * 1 = 10.0
    Net PnL = 10.0 - 0.21 = 9.79
    """

    @pytest.fixture()
    def curves(self):
        n = 100
        fill_fee = 0.01 * 100.0 * TAKER_RATE  # 0.001

        mkt_rows = [
            {"ts": 0, "exchange_id": 1, "security_id": 1, "mid_price": 100.0},
        ]
        mkt_rows += [
            {"ts": (i + 1) * 10, "exchange_id": 1, "security_id": 1, "mid_price": 100.0}
            for i in range(n)
        ]
        mkt_rows.append({"ts": n * 10 + 100, "exchange_id": 1, "security_id": 1, "mid_price": 110.0})
        mkt_rows.append({"ts": n * 10 + 200, "exchange_id": 1, "security_id": 1, "mid_price": 110.0})

        exec_rows = [
            {"ts": i * 10 + 5, "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 0.01, "fill_price": 100.0, "fee": fill_fee}
            for i in range(n)
        ]
        sell_fee = 1.0 * 110.0 * TAKER_RATE
        exec_rows.append(
            {"ts": n * 10 + 150, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "fill_qty": 1.0, "fill_price": 110.0, "fee": sell_fee}
        )

        return build_curves(_market_df(mkt_rows), _exec_df(exec_rows))

    def test_total_fees_no_drift(self, curves):
        expected = 100 * (0.01 * 100.0 * TAKER_RATE) + 1.0 * 110.0 * TAKER_RATE
        assert curves.fees.iloc[-1] == pytest.approx(expected, rel=1e-9)

    def test_net_pnl_no_drift(self, curves):
        total_fees = 100 * (0.01 * 100.0 * TAKER_RATE) + 1.0 * 110.0 * TAKER_RATE
        assert curves.pnl.iloc[-1] == pytest.approx(10.0 - total_fees, rel=1e-9)

    def test_fill_count(self, curves):
        assert curves.fill_count == 101


class TestFeeAlwaysReducesPnl:
    """Fee must reduce PnL on both buy and sell sides."""

    def test_fee_on_buy_reduces_pnl(self):
        # Market tick must come AFTER the fill so merge_asof can pick up the fill state.
        # no_fee:   cash=-100,   pos=1, mid=110 → pnl = -100   + 110 = 10.0
        # with_fee: cash=-100.5, pos=1, mid=110 → pnl = -100.5 + 110 = 9.5
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 110.0},
        ])
        no_fee = _exec_df([
            {"ts": 50, "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 1.0, "fill_price": 100.0, "fee": 0.0},
        ])
        with_fee = _exec_df([
            {"ts": 50, "exchange_id": 1, "security_id": 1, "side": "BID",
             "fill_qty": 1.0, "fill_price": 100.0, "fee": 0.5},
        ])
        pnl_no_fee = build_curves(mkt, no_fee).pnl.iloc[-1]
        pnl_with_fee = build_curves(mkt, with_fee).pnl.iloc[-1]
        assert pnl_with_fee < pnl_no_fee
        assert pnl_no_fee - pnl_with_fee == pytest.approx(0.5)

    def test_fee_on_sell_reduces_pnl(self):
        # no_fee:   cash=+100,  pos=-1, mid=90 → pnl = 100  - 90 = 10.0
        # with_fee: cash=+99.5, pos=-1, mid=90 → pnl = 99.5 - 90 = 9.5
        mkt = _market_df([
            {"ts": 0,   "exchange_id": 1, "security_id": 1, "mid_price": 90.0},
            {"ts": 100, "exchange_id": 1, "security_id": 1, "mid_price": 90.0},
        ])
        no_fee = _exec_df([
            {"ts": 50, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "fill_qty": 1.0, "fill_price": 100.0, "fee": 0.0},
        ])
        with_fee = _exec_df([
            {"ts": 50, "exchange_id": 1, "security_id": 1, "side": "ASK",
             "fill_qty": 1.0, "fill_price": 100.0, "fee": 0.5},
        ])
        pnl_no_fee = build_curves(mkt, no_fee).pnl.iloc[-1]
        pnl_with_fee = build_curves(mkt, with_fee).pnl.iloc[-1]
        assert pnl_with_fee < pnl_no_fee
        assert pnl_no_fee - pnl_with_fee == pytest.approx(0.5)
