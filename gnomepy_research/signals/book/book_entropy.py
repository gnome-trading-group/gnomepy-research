from __future__ import annotations

import math

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class BookEntropy(BookSignal):
    """Shannon entropy of the depth distribution across book levels.

    Computed per side, then averaged. Normalized to [0, 1] by dividing
    by log(num_levels) so 1.0 = perfectly uniform depth and 0.0 = all
    depth at one level.

    High entropy = liquidity spread across levels (resilient).
    Low entropy = concentrated liquidity (brittle, vulnerable to sweeps).

    Args:
        num_levels: Number of price levels to include (max 10).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, warmup: int = 10):
        self.num_levels = num_levels
        self.warmup = warmup
        self._log_n = math.log(num_levels)
        self._value = 0.0
        self._count = 0

    def _entropy(self, data: Mbp10Schema, use_bid: bool) -> float:
        total = 0.0
        for i in range(self.num_levels):
            total += data.bid_size(i) if use_bid else data.ask_size(i)
        if total <= 0.0:
            return 0.0
        h = 0.0
        for i in range(self.num_levels):
            sz = data.bid_size(i) if use_bid else data.ask_size(i)
            if sz > 0:
                p = sz / total
                h -= p * math.log(p)
        return h / self._log_n

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        h_bid = self._entropy(data, use_bid=True)
        h_ask = self._entropy(data, use_bid=False)
        if h_bid == 0.0 and h_ask == 0.0:
            return
        denom = (1 if h_bid > 0 else 0) + (1 if h_ask > 0 else 0)
        self._value = (h_bid + h_ask) / denom
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0
