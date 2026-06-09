from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.market_state.base import MarketStateSignal


class ExchangeLatency(MarketStateSignal):
    """EWMA of feed latency (timestamp_recv - timestamp_event), in nanoseconds.

    Sudden spikes indicate exchange congestion, connectivity issues, or
    processing delays. Useful for detecting when data may be stale.

    Args:
        alpha: EWMA smoothing factor (0.99 = slow/stable baseline).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, alpha: float = 0.99, warmup: int = 100):
        self.alpha = alpha
        self.warmup = warmup
        self._ewma = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        recv = data.timestamp_recv
        event = data.timestamp_event
        if recv <= 0 or event <= 0:
            return
        latency = recv - event
        if latency < 0:
            return

        if self._count == 0:
            self._ewma = float(latency)
        else:
            self._ewma = self.alpha * self._ewma + (1.0 - self.alpha) * latency
        self._count += 1

    def value(self) -> float:
        return self._ewma

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._ewma = 0.0
        self._count = 0
