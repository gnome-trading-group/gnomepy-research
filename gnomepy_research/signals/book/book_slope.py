from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class BookSlope(BookSignal):
    """OLS regression slope of cumulative depth vs price distance from mid.

    Fit: cumulative_size_i = slope * distance_bps_i for each side,
    then output the average of bid and ask slopes.

    Steep slope = more depth per bps further from mid (resilient book).
    Flat slope = thin book, large orders would move the price quickly.

    Output in lots per basis point.

    Args:
        num_levels: Number of price levels to include (max 10).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, warmup: int = 10):
        self.num_levels = num_levels
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def _side_slope(
        self,
        data: Mbp10Schema,
        mid: float,
        use_bid: bool,
    ) -> float:
        n = self.num_levels
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        cum_size = 0.0
        valid = 0

        for i in range(n):
            price = data.bid_price(i) if use_bid else data.ask_price(i)
            size = data.bid_size(i) if use_bid else data.ask_size(i)
            if price <= 0:
                break
            cum_size += size
            x = abs(mid - price) / mid * 10000.0  # bps from mid
            y = cum_size
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x
            valid += 1

        if valid < 2:
            return 0.0
        denom = valid * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            return 0.0
        return (valid * sum_xy - sum_x * sum_y) / denom

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid0 = data.bid_price(0)
        ask0 = data.ask_price(0)
        if bid0 <= 0 or ask0 <= 0:
            return

        mid = (bid0 + ask0) / 2.0
        bid_slope = self._side_slope(data, mid, use_bid=True)
        ask_slope = self._side_slope(data, mid, use_bid=False)
        self._value = (bid_slope + ask_slope) / 2.0
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0
