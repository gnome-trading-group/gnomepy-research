from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class BookPressure(BookSignal):
    """Exponentially-weighted depth imbalance, prioritizing near levels.

    Each level i is weighted by decay^i, giving L1 the most influence.
    Output in [-1, +1].

    Args:
        num_levels: Number of price levels to include (max 10).
        decay: Weight decay per level. 0.5 means L1 weight = 1, L2 = 0.5, L3 = 0.25, ...
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, decay: float = 0.5, warmup: int = 10):
        self.num_levels = num_levels
        self.decay = decay
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid_vol = 0.0
        ask_vol = 0.0
        w = 1.0
        for i in range(self.num_levels):
            bid_vol += w * data.bid_size(i)
            ask_vol += w * data.ask_size(i)
            w *= self.decay

        total = bid_vol + ask_vol
        if total <= 0.0:
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
