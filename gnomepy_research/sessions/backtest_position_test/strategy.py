from __future__ import annotations

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.enums import ExecType
from gnomepy.java.schemas import Schema

Listing = tuple[int, int]  # (exchange_id, security_id)

ACTION_NONE = 0
ACTION_BUY_SENT = 1
ACTION_SELL_SENT = 2
ACTION_BUY_FILL = 3
ACTION_SELL_FILL = 4
ACTION_OBSERVE = 5

PHASE_IDLE = "idle"
PHASE_BUYING = "buying"
PHASE_HOLDING = "holding"
PHASE_SELLING = "selling"


class BacktestPositionTest(Strategy):
    def __init__(
        self,
        size: int = 1_000_000,
        buy_interval_ns: int = 5_000_000_000,
        hold_ticks: int = 30,
        processing_time_ns: int = 0,
    ):
        self.size = size
        self.buy_interval_ns = buy_interval_ns
        self.hold_ticks = hold_ticks
        self._processing_time_ns = processing_time_ns

        self._listings: list[Listing] = []
        self._listing_set: set[Listing] = set()
        self._listing_index: int = 0
        self._last_buy_ts: int = 0

        self._phase: dict[Listing, str] = {}
        self._expected_qty: dict[Listing, int] = {}
        self._ticks_holding: dict[Listing, int] = {}
        self._entry_price: dict[Listing, int] = {}

        self._buf = None
        self._col_ts = None
        self._col_eid = None
        self._col_sid = None
        self._col_action = None
        self._col_pv_net_qty = None
        self._col_expected_net_qty = None
        self._col_pv_realized_pnl = None
        self._col_pv_total_fees = None
        self._col_fill_price = None
        self._col_fill_qty = None
        self._col_diverged = None

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("position_tracking")
        self._col_ts = buf.addLongColumn("timestamp")
        self._col_eid = buf.addLongColumn("exchange_id")
        self._col_sid = buf.addLongColumn("security_id")
        self._col_action = buf.addLongColumn("action")
        self._col_pv_net_qty = buf.addLongColumn("pv_net_qty")
        self._col_expected_net_qty = buf.addLongColumn("expected_net_qty")
        self._col_pv_realized_pnl = buf.addDoubleColumn("pv_realized_pnl")
        self._col_pv_total_fees = buf.addDoubleColumn("pv_total_fees")
        self._col_fill_price = buf.addLongColumn("fill_price")
        self._col_fill_qty = buf.addLongColumn("fill_qty")
        self._col_diverged = buf.addLongColumn("diverged")
        buf.freeze()
        self._buf = buf

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def _init_listing(self, listing: Listing) -> None:
        self._listings.append(listing)
        self._listing_set.add(listing)
        self._phase[listing] = PHASE_IDLE
        self._expected_qty[listing] = 0
        self._ticks_holding[listing] = 0
        self._entry_price[listing] = 0

    def _log(
        self,
        ts: int,
        eid: int,
        sid: int,
        action: int,
        fill_price: int = 0,
        fill_qty: int = 0,
    ) -> None:
        if self._buf is None:
            return
        listing = (eid, sid)
        pos = self.positions.get_position(eid, sid)
        pv_net_qty = pos.net_quantity if pos is not None else 0
        pv_realized_pnl = pos.realized_pnl if pos is not None else 0.0
        pv_total_fees = pos.total_fees if pos is not None else 0.0
        expected_qty = self._expected_qty.get(listing, 0)

        row = self._buf.appendRow()
        self._buf.setLong(row, self._col_ts, ts)
        self._buf.setLong(row, self._col_eid, eid)
        self._buf.setLong(row, self._col_sid, sid)
        self._buf.setLong(row, self._col_action, action)
        self._buf.setLong(row, self._col_pv_net_qty, pv_net_qty)
        self._buf.setLong(row, self._col_expected_net_qty, expected_qty)
        self._buf.setDouble(row, self._col_pv_realized_pnl, pv_realized_pnl)
        self._buf.setDouble(row, self._col_pv_total_fees, pv_total_fees)
        self._buf.setLong(row, self._col_fill_price, fill_price)
        self._buf.setLong(row, self._col_fill_qty, fill_qty)
        self._buf.setLong(row, self._col_diverged, 1 if pv_net_qty != expected_qty else 0)

    def on_market_data(self, data: Schema) -> list[Intent]:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return []

        eid = data.exchange_id
        sid = data.security_id
        ts = data.event_timestamp
        listing: Listing = (eid, sid)

        if listing not in self._listing_set:
            self._init_listing(listing)

        self._log(ts, eid, sid, ACTION_OBSERVE)

        phase = self._phase[listing]

        if phase == PHASE_HOLDING:
            self._ticks_holding[listing] += 1
            if self._ticks_holding[listing] >= self.hold_ticks:
                qty = self._expected_qty[listing]
                if qty > 0:
                    self._phase[listing] = PHASE_SELLING
                    self._log(ts, eid, sid, ACTION_SELL_SENT)
                    return [Intent(
                        exchange_id=eid,
                        security_id=sid,
                        take_side=Side.ASK,
                        take_size=qty,
                        take_order_type=OrderType.MARKET,
                    )]
                self._phase[listing] = PHASE_IDLE
            return []

        if phase != PHASE_IDLE:
            return []

        n = len(self._listings)
        if n == 0:
            return []

        target = self._listings[self._listing_index % n]
        if target != listing:
            return []

        if ts - self._last_buy_ts < self.buy_interval_ns:
            return []

        size = self.positions.compliant_size(eid, sid, self.size, ask)
        if size <= 0:
            return []

        self._phase[listing] = PHASE_BUYING
        self._last_buy_ts = ts
        self._listing_index += 1
        self._log(ts, eid, sid, ACTION_BUY_SENT)
        return [Intent(
            exchange_id=eid,
            security_id=sid,
            take_side=Side.BID,
            take_size=size,
            take_order_type=OrderType.MARKET,
        )]

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        if report.exec_type not in (ExecType.FILL, ExecType.PARTIAL_FILL):
            return []
        if report.filled_qty <= 0:
            return []

        eid = report.exchange_id
        sid = report.security_id
        ts = report.timestamp_recv
        listing: Listing = (eid, sid)
        phase = self._phase.get(listing, PHASE_IDLE)

        if phase == PHASE_BUYING:
            self._expected_qty[listing] = self._expected_qty.get(listing, 0) + report.filled_qty
            self._entry_price[listing] = report.fill_price
            if report.leaves_qty == 0:
                self._phase[listing] = PHASE_HOLDING
                self._ticks_holding[listing] = 0
            self._log(ts, eid, sid, ACTION_BUY_FILL, report.fill_price, report.filled_qty)

        elif phase == PHASE_SELLING:
            self._expected_qty[listing] = self._expected_qty.get(listing, 0) - report.filled_qty
            if report.leaves_qty == 0:
                self._phase[listing] = PHASE_IDLE
            self._log(ts, eid, sid, ACTION_SELL_FILL, report.fill_price, report.filled_qty)

        return []
