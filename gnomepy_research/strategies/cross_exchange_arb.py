"""Cross-exchange arbitrage strategy.

Monitors the same security on two exchanges and trades when prices
diverge beyond a threshold. Buys on the cheap exchange, sells on the
expensive one. Closes when the spread reverts. Manages leg risk by
unwinding imbalanced positions after a timeout.
"""
from __future__ import annotations

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema


class CrossExchangeArb(Strategy):
    def __init__(
        self,
        exchange_id_a: int,
        exchange_id_b: int,
        security_id: int,
        size: int = 1,
        max_position: int = 5,
        entry_threshold_bps: float = 2.0,
        close_threshold_bps: float = 0.5,
        max_imbalance_ticks: int = 100,
        max_staleness_ns: int = 200_000_000,
        warmup_ticks: int = 50,
        processing_time_ns: int = 0,
    ):
        self.eid_a = exchange_id_a
        self.eid_b = exchange_id_b
        self.security_id = security_id
        self.size = size
        self.max_position = max_position
        self.entry_threshold_bps = entry_threshold_bps
        self.close_threshold_bps = close_threshold_bps
        self.max_imbalance_ticks = max_imbalance_ticks
        self.max_staleness_ns = max_staleness_ns
        self.warmup_ticks = warmup_ticks
        self._processing_time_ns = processing_time_ns

        # Latest book state per exchange.
        self._best_bid: dict[int, int] = {}
        self._best_ask: dict[int, int] = {}
        self._mid: dict[int, int] = {}
        self._last_update_ts: dict[int, int] = {}

        self._tick_count = 0
        self._imbalance_ticks = 0

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_market_data(self, data: Schema) -> list[Intent]:
        if data.security_id != self.security_id:
            return []
        timestamp = data.event_timestamp
        eid = data.exchange_id
        if eid not in (self.eid_a, self.eid_b):
            return []

        # Update book state for this exchange.
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return []
        self._best_bid[eid] = bid
        self._best_ask[eid] = ask
        self._mid[eid] = (bid + ask) // 2
        self._last_update_ts[eid] = timestamp

        self._tick_count += 1
        if self._tick_count < self.warmup_ticks:
            return []

        # Need data from both exchanges.
        if self.eid_a not in self._mid or self.eid_b not in self._mid:
            return []

        # Staleness guard — skip if the other exchange's data is too old.
        other_eid = self.eid_b if eid == self.eid_a else self.eid_a
        other_ts = self._last_update_ts.get(other_eid, 0)
        if timestamp - other_ts > self.max_staleness_ns:
            return []

        mid_a = self._mid[self.eid_a]
        mid_b = self._mid[self.eid_b]
        avg_mid = (mid_a + mid_b) / 2
        if avg_mid <= 0:
            return []

        spread_bps = (mid_b - mid_a) / avg_mid * 10_000

        pos_a = self.oms.get_effective_quantity(self.eid_a, self.security_id)
        pos_b = self.oms.get_effective_quantity(self.eid_b, self.security_id)

        # --- Imbalance detection: one leg filled, other didn't ---
        if (pos_a != 0) != (pos_b != 0):
            self._imbalance_ticks += 1
        else:
            self._imbalance_ticks = 0

        if self._imbalance_ticks >= self.max_imbalance_ticks:
            # Unwind the exposed leg.
            self._imbalance_ticks = 0
            return self._unwind(pos_a, pos_b)

        # --- Close: spread reverted ---
        if pos_a != 0 and pos_b != 0:
            # We're in an arb: long one exchange, short the other.
            # Close if spread is within close_threshold of zero.
            if abs(spread_bps) < self.close_threshold_bps:
                return self._close_both(pos_a, pos_b)

        # --- Entry: spread exceeds threshold ---
        if abs(pos_a) >= self.max_position or abs(pos_b) >= self.max_position:
            return []

        if spread_bps > self.entry_threshold_bps:
            # B is expensive relative to A → buy A, sell B.
            return [
                self._take(self.eid_a, Side.ASK),  # buy on A (take the ask)
                self._take(self.eid_b, Side.BID),  # sell on B (take the bid)
            ]

        if spread_bps < -self.entry_threshold_bps:
            # A is expensive relative to B → buy B, sell A.
            return [
                self._take(self.eid_b, Side.ASK),  # buy on B
                self._take(self.eid_a, Side.BID),  # sell on A
            ]

        return []

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []

    def _take(self, exchange_id: int, side: Side) -> Intent:
        return Intent(
            exchange_id=exchange_id,
            security_id=self.security_id,
            take_side=side,
            take_size=self.size,
            take_order_type=OrderType.MARKET,
        )

    def _close_both(self, pos_a: int, pos_b: int) -> list[Intent]:
        """Close positions on both exchanges."""
        intents = []
        if pos_a > 0:
            intents.append(self._take(self.eid_a, Side.BID))  # sell A
        elif pos_a < 0:
            intents.append(self._take(self.eid_a, Side.ASK))  # buy A
        if pos_b > 0:
            intents.append(self._take(self.eid_b, Side.BID))  # sell B
        elif pos_b < 0:
            intents.append(self._take(self.eid_b, Side.ASK))  # buy B
        return intents

    def _unwind(self, pos_a: int, pos_b: int) -> list[Intent]:
        """Unwind any imbalanced position."""
        intents = []
        if pos_a > 0 and pos_b == 0:
            intents.append(self._take(self.eid_a, Side.BID))
        elif pos_a < 0 and pos_b == 0:
            intents.append(self._take(self.eid_a, Side.ASK))
        elif pos_b > 0 and pos_a == 0:
            intents.append(self._take(self.eid_b, Side.BID))
        elif pos_b < 0 and pos_a == 0:
            intents.append(self._take(self.eid_b, Side.ASK))
        return intents
