from __future__ import annotations

from gnomepy.java.schemas import JavaMBP10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class TradeArrivalTime(FlowSignal):
    """Time since the last trade event in nanoseconds.

    Updates every tick — value grows between trades and resets to 0
    on each trade. Uses data.timestamp_event (exchange time).
    """

    def __init__(self, warmup_trades: int = 2):
        self.warmup_trades = warmup_trades
        self._last_trade_ts = 0
        self._arrival_time_ns = 0
        self._trade_count = 0

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        exchange_ts = data.timestamp_event

        if is_trade_event(data):
            if self._last_trade_ts > 0:
                self._arrival_time_ns = exchange_ts - self._last_trade_ts
            self._last_trade_ts = exchange_ts
            self._trade_count += 1
        elif self._last_trade_ts > 0:
            self._arrival_time_ns = exchange_ts - self._last_trade_ts

    def value(self) -> float:
        return float(self._arrival_time_ns)

    def is_ready(self) -> bool:
        return self._trade_count >= self.warmup_trades

    def reset(self) -> None:
        self._last_trade_ts = 0
        self._arrival_time_ns = 0
        self._trade_count = 0
