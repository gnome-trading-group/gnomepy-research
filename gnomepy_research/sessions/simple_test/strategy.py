from __future__ import annotations

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema


class SimpleTest(Strategy):
    def __init__(self, size: int = 1000000, hold_ticks: int = 20, interval_ns: int = 5_000_000_000):
        self.size = size
        self.hold_ticks = hold_ticks
        self.interval_ns = interval_ns
        self._last_buy_ts: int = 0
        self._ticks_held: int = 0
        self._waiting: bool = False

    def on_market_data(self, data: Schema) -> list[Intent]:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return []

        eid = data.exchange_id
        sid = data.security_id
        ts = data.event_timestamp

        pos = self.positions.get_position(eid, sid)
        net_qty = pos.net_quantity if pos is not None else 0

        if net_qty != 0:
            self._ticks_held += 1
            if self._ticks_held >= self.hold_ticks:
                self._ticks_held = 0
                self._waiting = False
                close_side = Side.ASK if net_qty > 0 else Side.BID
                close_price = bid if net_qty > 0 else ask
                close_size = self.positions.compliant_size(eid, sid, abs(net_qty), close_price)
                return [Intent(
                    exchange_id=eid,
                    security_id=sid,
                    take_side=close_side,
                    take_size=close_size,
                    take_order_type=OrderType.MARKET,
                )]
            return []

        if self._waiting:
            return []

        if ts - self._last_buy_ts >= self.interval_ns:
            self._last_buy_ts = ts
            self._waiting = True
            self._ticks_held = 0
            size = self.positions.compliant_size(eid, sid, self.size, ask)
            return [Intent(
                exchange_id=eid,
                security_id=sid,
                take_side=Side.BID,
                take_size=size,
                take_order_type=OrderType.MARKET,
            )]

        return []

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []
