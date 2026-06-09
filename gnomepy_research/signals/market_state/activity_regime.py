from __future__ import annotations

from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import is_trade_event
from gnomepy_research.signals.market_state.base import MarketStateSignal


class ActivityRegime(MarketStateSignal):
    """Ratio of fast trade intensity to slow trade intensity.

    Values > 1 = elevated activity relative to background;
    < 1 = below-average quiet period.

    Args:
        fast_horizon_ns: Short window for "current" activity level.
        slow_horizon_ns: Long window for the background rate.
        warmup_trades: Minimum trades before is_ready() returns True.
    """

    def __init__(
        self,
        fast_horizon_ns: int = 1_000_000_000,
        slow_horizon_ns: int = 60_000_000_000,
        warmup_trades: int = 50,
    ):
        self.fast_horizon_ns = fast_horizon_ns
        self.slow_horizon_ns = slow_horizon_ns
        self.warmup_trades = warmup_trades

        self._fast: deque[int] = deque()
        self._slow: deque[int] = deque()
        self._total_trades = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if not is_trade_event(data):
            return

        self._total_trades += 1

        fast_cutoff = timestamp - self.fast_horizon_ns
        while self._fast and self._fast[0] < fast_cutoff:
            self._fast.popleft()
        self._fast.append(timestamp)

        slow_cutoff = timestamp - self.slow_horizon_ns
        while self._slow and self._slow[0] < slow_cutoff:
            self._slow.popleft()
        self._slow.append(timestamp)

    def value(self) -> float:
        slow_rate = len(self._slow) / (self.slow_horizon_ns / 1e9)
        if slow_rate <= 0.0:
            return 1.0
        fast_rate = len(self._fast) / (self.fast_horizon_ns / 1e9)
        return fast_rate / slow_rate

    def is_ready(self) -> bool:
        return self._total_trades >= self.warmup_trades

    def reset(self) -> None:
        self._fast.clear()
        self._slow.clear()
        self._total_trades = 0
