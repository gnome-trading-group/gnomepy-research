from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class CountImbalance(BookSignal):
    """Order count imbalance across top N book levels.

    (bid_count - ask_count) / (bid_count + ask_count), output in [-1, +1].

    Count-based imbalance can differ from size-based in markets with many
    small orders on one side vs fewer large orders on the other.

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
        bid_cnt = 0
        ask_cnt = 0
        for i in range(self.num_levels):
            bid_cnt += data.bid_count(i)
            ask_cnt += data.ask_count(i)

        total = bid_cnt + ask_cnt
        if total <= 0:
            return

        self._value = (bid_cnt - ask_cnt) / total
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0
