from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal


class LevelMagnetism(FlowSignal):
    """Distance from mid to the nearest abnormally thick resting price level.

    A "thick" level has size > size_threshold_mult * average_level_size.
    Prices tend to gravitate toward large resting orders.

    Positive = nearest thick level is above mid (pull toward ask).
    Negative = nearest thick level is below mid (pull toward bid).
    0.0 = no thick level found within num_levels.

    Output in basis points.

    Args:
        size_threshold_mult: Multiple of average level size to qualify as thick.
        num_levels: Number of book levels to scan (max 10).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(
        self,
        size_threshold_mult: float = 3.0,
        num_levels: int = 10,
        warmup: int = 10,
    ):
        self.size_threshold_mult = size_threshold_mult
        self.num_levels = num_levels
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid0 = data.bid_price(0)
        ask0 = data.ask_price(0)
        if bid0 <= 0 or ask0 <= 0:
            return
        mid = (bid0 + ask0) / 2.0

        total_size = 0.0
        for i in range(self.num_levels):
            total_size += data.bid_size(i) + data.ask_size(i)
        avg_size = total_size / (2 * self.num_levels)
        if avg_size <= 0.0:
            self._value = 0.0
            self._count += 1
            return

        threshold = self.size_threshold_mult * avg_size
        best_dist = -1.0
        best_price = 0.0

        for i in range(self.num_levels):
            bid_p = data.bid_price(i)
            if bid_p > 0 and data.bid_size(i) >= threshold:
                d = mid - bid_p
                if best_dist < 0 or d < best_dist:
                    best_dist = d
                    best_price = float(bid_p)

            ask_p = data.ask_price(i)
            if ask_p > 0 and data.ask_size(i) >= threshold:
                d = ask_p - mid
                if best_dist < 0 or d < best_dist:
                    best_dist = d
                    best_price = float(ask_p)

        if best_price == 0.0:
            self._value = 0.0
        else:
            self._value = (best_price - mid) / mid * 10000.0
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0
