from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class DepthRatio(BookSignal):
    """Ratio of L1 depth to deeper levels, averaged across both sides.

    (l1 + 1) / (rest + 1) per side to avoid division by zero.
    Values > 1 = L1 is thick relative to the rest; < 1 = uniform or back-weighted.

    Args:
        num_levels: Total levels to include (L1 + levels 1..N-1).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, warmup: int = 10):
        self.num_levels = num_levels
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid_l1 = data.bid_size(0)
        ask_l1 = data.ask_size(0)
        bid_rest = 0
        ask_rest = 0
        for i in range(1, self.num_levels):
            bid_rest += data.bid_size(i)
            ask_rest += data.ask_size(i)

        bid_ratio = (bid_l1 + 1) / (bid_rest + 1)
        ask_ratio = (ask_l1 + 1) / (ask_rest + 1)
        self._value = (bid_ratio + ask_ratio) / 2.0
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0
