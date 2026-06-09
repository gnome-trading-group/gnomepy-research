from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class TopHeaviness(BookSignal):
    """Fraction of total book depth concentrated at L1.

    (bid_size_0 + ask_size_0) / total_depth, output in [0, 1].
    High values mean the book is concentrated at the top — vulnerable to sweeps.

    Args:
        num_levels: Total levels to sum for the denominator (max 10).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, warmup: int = 10):
        self.num_levels = num_levels
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        l1 = data.bid_size(0) + data.ask_size(0)
        total = l1
        for i in range(1, self.num_levels):
            total += data.bid_size(i) + data.ask_size(i)

        if total <= 0:
            return

        self._value = l1 / total
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0
