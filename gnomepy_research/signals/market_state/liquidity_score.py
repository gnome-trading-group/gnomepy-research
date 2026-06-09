from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.market_state.base import MarketStateSignal


class LiquidityScore(MarketStateSignal):
    """Current book depth normalized by its EWMA.

    Values > 1 = above-average liquidity; < 1 = below-average.
    Drops below 1 signal thinning books and elevated adverse-selection risk.

    Args:
        num_levels: Number of price levels to sum for total depth.
        alpha: EWMA smoothing factor for the baseline (0.99 = slow).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, alpha: float = 0.99, warmup: int = 100):
        self.num_levels = num_levels
        self.alpha = alpha
        self.warmup = warmup
        self._ewma = 0.0
        self._last_depth = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        depth = 0.0
        for i in range(self.num_levels):
            depth += data.bid_size(i) + data.ask_size(i)

        if depth <= 0.0:
            return

        self._last_depth = depth
        if self._count == 0:
            self._ewma = depth
        else:
            self._ewma = self.alpha * self._ewma + (1.0 - self.alpha) * depth
        self._count += 1

    def value(self) -> float:
        if self._ewma <= 0.0:
            return 1.0
        return self._last_depth / self._ewma

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._ewma = 0.0
        self._last_depth = 0.0
        self._count = 0
