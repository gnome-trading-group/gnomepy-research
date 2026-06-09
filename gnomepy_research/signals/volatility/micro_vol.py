from __future__ import annotations

import math
from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.volatility.base import VolatilitySignal


class MicroVolatility(VolatilitySignal):
    """Rolling standard deviation of microprice changes, in basis points.

    Captures volatility including book-side pressure shifts that mid ignores,
    since microprice responds to depth imbalance changes without requiring
    a trade.

    microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)

    Args:
        horizon: Number of observations in the rolling window.
        warmup: Ticks before is_ready() returns True (defaults to horizon).
    """

    def __init__(self, horizon: int = 100, warmup: int | None = None):
        self.horizon = horizon
        self.warmup = horizon if warmup is None else warmup
        self._prev_micro = 0.0
        self._changes: deque[float] = deque(maxlen=horizon)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        bid_sz = data.bid_size(0)
        ask_sz = data.ask_size(0)
        total = bid_sz + ask_sz
        if bid <= 0 or ask <= 0 or total <= 0:
            return

        micro = (bid * ask_sz + ask * bid_sz) / total

        if self._prev_micro > 0.0:
            change = (micro - self._prev_micro) / self._prev_micro * 10000.0

            if self._count == self.horizon:
                old = self._changes[0]
                self._sum -= old
                self._sum_sq -= old * old
            else:
                self._count += 1

            self._changes.append(change)
            self._sum += change
            self._sum_sq += change * change

        self._prev_micro = micro

    def value(self) -> float:
        if self._count < 2:
            return 0.0
        mean = self._sum / self._count
        variance = self._sum_sq / self._count - mean * mean
        return math.sqrt(max(variance, 0.0))

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._prev_micro = 0.0
        self._changes.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0
