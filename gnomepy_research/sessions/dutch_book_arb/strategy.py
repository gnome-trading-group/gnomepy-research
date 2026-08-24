from __future__ import annotations

from enum import IntEnum

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema

PRICE_SCALE = 1_000_000_000
SIZE_SCALE = 1_000_000

Listing = tuple[int, int]  # (exchange_id, security_id)


class State(IntEnum):
    FLAT = 0
    ENTERING = 1
    PARTIAL = 2
    UNWINDING = 3


class DutchBookArb(Strategy):
    def __init__(
        self,
        size: int = 1_000_000,
        fee_rate: float = 0.07,
        min_edge_bps: float = 10.0,
        stale_threshold_ns: int = 2_000_000_000,
        imbalance_timeout_ns: int = 30_000_000_000,
        cooldown_ns: int = 500_000_000,
        processing_time_ns: int = 0,
    ):
        self.size = size
        self.fee_rate = fee_rate
        self.min_edge_bps = min_edge_bps
        self.stale_threshold_ns = stale_threshold_ns
        self.imbalance_timeout_ns = imbalance_timeout_ns
        self.cooldown_ns = cooldown_ns
        self._processing_time_ns = processing_time_ns

        self._listings: list[Listing] = []
        self._listing_set: set[Listing] = set()
        self._best_ask: dict[Listing, int] = {}
        self._best_ask_size: dict[Listing, int] = {}
        self._last_update_ts: dict[Listing, int] = {}

        self._state: State = State.FLAT
        self._entry_ts: int = 0
        self._last_attempt_ts: int = 0
        self._target_qty: int = 0

        self._buf = None
        self._col_ts = None
        self._col_sum_asks = None
        self._col_edge_bps = None
        self._col_state = None
        self._col_n_outcomes = None
        self._col_min_ask_size = None

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("diagnostics")
        self._col_ts = buf.addLongColumn("timestamp")
        self._col_sum_asks = buf.addDoubleColumn("sum_asks")
        self._col_edge_bps = buf.addDoubleColumn("edge_bps")
        self._col_state = buf.addLongColumn("state")
        self._col_n_outcomes = buf.addLongColumn("n_outcomes")
        self._col_min_ask_size = buf.addLongColumn("min_ask_size")
        buf.freeze()
        self._buf = buf

    def _log(self, ts: int, sum_asks: float, edge_bps: float) -> None:
        if self._buf is None:
            return
        row = self._buf.appendRow()
        self._buf.setLong(row, self._col_ts, ts)
        self._buf.setDouble(row, self._col_sum_asks, sum_asks)
        self._buf.setDouble(row, self._col_edge_bps, edge_bps)
        self._buf.setLong(row, self._col_state, int(self._state))
        self._buf.setLong(row, self._col_n_outcomes, len(self._listings))
        min_size = min(self._best_ask_size.values()) if self._best_ask_size else 0
        self._buf.setLong(row, self._col_min_ask_size, min_size)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def _net_qty(self, listing: Listing) -> int:
        pos = self.positions.get_position(*listing)
        return pos.net_quantity if pos is not None else 0

    def _compute_edge(self) -> tuple[float, float]:
        """Return (sum_asks, edge_bps) where sum_asks is in [0, 1] dollar terms."""
        sum_asks = 0.0
        total_fee = 0.0
        for lst in self._listings:
            p = self._best_ask[lst] / PRICE_SCALE
            sum_asks += p
            total_fee += self.fee_rate * p * (1.0 - p)
        edge_bps = (1.0 - sum_asks - total_fee) * 10_000
        return sum_asks, edge_bps

    def _all_books_valid(self, timestamp: int) -> bool:
        if len(self._listings) < 2:
            return False
        for lst in self._listings:
            if lst not in self._best_ask or self._best_ask[lst] <= 0:
                return False
            if timestamp - self._last_update_ts.get(lst, 0) > self.stale_threshold_ns:
                return False
        return True

    def on_market_data(self, data: Schema) -> list[Intent]:
        listing: Listing = (data.exchange_id, data.security_id)
        timestamp = data.event_timestamp

        if listing not in self._listing_set:
            self._listings.append(listing)
            self._listing_set.add(listing)

        ask = data.ask_price(0)
        ask_size = data.ask_size(0)
        if ask > 0:
            self._best_ask[listing] = ask
            self._best_ask_size[listing] = ask_size
        self._last_update_ts[listing] = timestamp

        if not self._all_books_valid(timestamp):
            self._log(timestamp, 0.0, 0.0)
            return []

        sum_asks, edge_bps = self._compute_edge()
        self._log(timestamp, sum_asks, edge_bps)

        if self._state == State.FLAT:
            return self._handle_flat(timestamp, edge_bps)
        if self._state == State.ENTERING:
            return self._handle_entering(timestamp)
        if self._state == State.PARTIAL:
            return self._handle_partial(timestamp)
        if self._state == State.UNWINDING:
            return self._handle_unwinding()
        return []

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []

    def _handle_flat(self, timestamp: int, edge_bps: float) -> list[Intent]:
        if timestamp - self._last_attempt_ts < self.cooldown_ns:
            return []
        if edge_bps < self.min_edge_bps:
            return []

        min_avail = min(self._best_ask_size.values())
        if min_avail <= 0:
            return []

        trade_size = min(self.size, min_avail)

        self._state = State.ENTERING
        self._entry_ts = timestamp
        self._last_attempt_ts = timestamp
        self._target_qty = trade_size

        return [
            Intent(
                exchange_id=lst[0],
                security_id=lst[1],
                take_side=Side.BID,
                take_size=self.positions.compliant_size(
                    lst[0], lst[1], trade_size, self._best_ask[lst]
                ),
                take_order_type=OrderType.MARKET,
            )
            for lst in self._listings
        ]

    def _handle_entering(self, timestamp: int) -> list[Intent]:
        # Give the taker delay + margin before checking fills
        if timestamp - self._entry_ts < 1_000_000_000:
            return []

        filled = [lst for lst in self._listings if self._net_qty(lst) >= self._target_qty]
        unfilled = [lst for lst in self._listings if self._net_qty(lst) < self._target_qty]

        if not unfilled:
            self._state = State.FLAT
            return []

        if filled:
            self._state = State.PARTIAL
            return []

        # Nothing filled — abandon and return to flat
        self._state = State.FLAT
        return []

    def _handle_partial(self, timestamp: int) -> list[Intent]:
        unfilled = [lst for lst in self._listings if self._net_qty(lst) < self._target_qty]
        if not unfilled:
            self._state = State.FLAT
            return []

        if timestamp - self._entry_ts > self.imbalance_timeout_ns:
            self._state = State.UNWINDING
            return self._unwind_intents()

        return []

    def _handle_unwinding(self) -> list[Intent]:
        open_lsts = [lst for lst in self._listings if self._net_qty(lst) > 0]
        if not open_lsts:
            self._state = State.FLAT
            return []
        return self._unwind_intents()

    def _unwind_intents(self) -> list[Intent]:
        intents = []
        for lst in self._listings:
            qty = self._net_qty(lst)
            if qty > 0:
                intents.append(
                    Intent(
                        exchange_id=lst[0],
                        security_id=lst[1],
                        take_side=Side.ASK,
                        take_size=qty,
                        take_order_type=OrderType.MARKET,
                    )
                )
        return intents
