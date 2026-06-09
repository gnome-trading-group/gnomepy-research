from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class LevelConcentration(BookSignal):
    """Herfindahl index of depth distribution across book levels.

    Computed separately for each side, then averaged.
    H = sum((size_i / total)^2) for i in 0..N-1

    1.0 = all depth at one level; 1/N = perfectly uniform.
    High concentration = liquidity clustered at specific prices.

    Args:
        num_levels: Number of price levels to include (max 10).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, warmup: int = 10):
        self.num_levels = num_levels
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def _herfindahl(self, data: Mbp10Schema, use_bid: bool) -> float:
        total = 0.0
        for i in range(self.num_levels):
            total += data.bid_size(i) if use_bid else data.ask_size(i)
        if total <= 0.0:
            return 0.0
        h = 0.0
        for i in range(self.num_levels):
            sz = data.bid_size(i) if use_bid else data.ask_size(i)
            share = sz / total
            h += share * share
        return h

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        h_bid = self._herfindahl(data, use_bid=True)
        h_ask = self._herfindahl(data, use_bid=False)
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
