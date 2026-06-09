from __future__ import annotations

from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class TradeIntensity(FlowSignal):
    """Trade arrival rate within a rolling window, in trades per second.

    Args:
        horizon_ns: Rolling window in nanoseconds.
        warmup_trades: Minimum trades before is_ready() returns True.
    """

    def __init__(self, horizon_ns: int = 1_000_000_000, warmup_trades: int = 10):
        self.horizon_ns = horizon_ns
        self.warmup_trades = warmup_trades
        self._timestamps: deque[int] = deque()
        self._total_trades = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if not is_trade_event(data):
            return

        self._total_trades += 1
        self._evict(timestamp)
        self._timestamps.append(timestamp)

    def _evict(self, now: int) -> None:
        cutoff = now - self.horizon_ns
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def value(self) -> float:
        horizon_sec = self.horizon_ns / 1e9
        return len(self._timestamps) / horizon_sec

    def is_ready(self) -> bool:
        return self._total_trades >= self.warmup_trades

    def reset(self) -> None:
        self._timestamps.clear()
        self._total_trades = 0
