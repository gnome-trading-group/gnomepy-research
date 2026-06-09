from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class SyntheticDepth(BookSignal):
    """Cumulative depth available within a fixed bps radius of mid.

    Scans all 10 book levels and sums sizes within [mid - radius, mid + radius].
    Output is the average of bid and ask depth within the radius, in lots.

    Unlike level-based depth, this accounts for non-uniform tick sizes and
    varying level spacing.

    Args:
        bps_radius: Price radius from mid in basis points.
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, bps_radius: float = 10.0, warmup: int = 10):
        self.bps_radius = bps_radius
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid0 = data.bid_price(0)
        ask0 = data.ask_price(0)
        if bid0 <= 0 or ask0 <= 0:
            return
        mid = (bid0 + ask0) / 2.0
        threshold = mid * self.bps_radius / 10000.0

        bid_depth = 0.0
        for i in range(10):
            bp = data.bid_price(i)
            if bp <= 0 or mid - bp > threshold:
                break
            bid_depth += data.bid_size(i)

        ask_depth = 0.0
        for i in range(10):
            ap = data.ask_price(i)
            if ap <= 0 or ap - mid > threshold:
                break
            ask_depth += data.ask_size(i)

        self._value = (bid_depth + ask_depth) / 2.0
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0
