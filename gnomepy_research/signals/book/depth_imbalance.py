from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class DepthImbalance(BookSignal):
    """Volume-weighted depth imbalance across top N book levels.

    (bid_vol - ask_vol) / (bid_vol + ask_vol), output in [-1, +1].
    Positive = more bid-side depth (bullish pressure).

    Args:
        num_levels: Number of price levels to include (max 10).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, warmup: int = 10):
        self.num_levels = num_levels
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid_vol = 0
        ask_vol = 0
        for i in range(self.num_levels):
            bid_vol += data.bid_size(i)
            ask_vol += data.ask_size(i)

        total = bid_vol + ask_vol
        if total <= 0:
            return

        self._value = (bid_vol - ask_vol) / total
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0
