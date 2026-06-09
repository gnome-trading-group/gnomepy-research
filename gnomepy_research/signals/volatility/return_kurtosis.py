from __future__ import annotations

import math
from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.volatility.base import VolatilitySignal


class ReturnKurtosis(VolatilitySignal):
    """Rolling excess kurtosis of mid-price log returns.

    Excess kurtosis = m4/sigma^4 - 3.
    Positive values signal fat tails; negative = platykurtic (thin tails).
    High kurtosis often precedes regime breaks or spike moves.

    Args:
        horizon: Number of log-return observations in the rolling window.
        warmup: Ticks before is_ready() (defaults to horizon).
    """

    def __init__(self, horizon: int = 200, warmup: int | None = None):
        self.horizon = horizon
        self.warmup = horizon if warmup is None else warmup
        self._prev_mid = 0
        self._returns: deque[float] = deque(maxlen=horizon)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._sum_cb = 0.0
        self._sum_qt = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return
        mid = (bid + ask) // 2
        if mid <= 0:
            return

        if self._prev_mid > 0:
            r = math.log(mid / self._prev_mid)

            if self._count == self.horizon:
                old = self._returns[0]
                old2 = old * old
                self._sum -= old
                self._sum_sq -= old2
                self._sum_cb -= old2 * old
                self._sum_qt -= old2 * old2
            else:
                self._count += 1

            self._returns.append(r)
            r2 = r * r
            self._sum += r
            self._sum_sq += r2
            self._sum_cb += r2 * r
            self._sum_qt += r2 * r2

        self._prev_mid = mid

    def value(self) -> float:
        n = self._count
        if n < 4:
            return 0.0
        mean = self._sum / n
        m2 = self._sum_sq / n - mean * mean
        if m2 <= 0.0:
            return 0.0
        m4 = (
            self._sum_qt / n
            - 4.0 * mean * self._sum_cb / n
            + 6.0 * mean * mean * self._sum_sq / n
            - 3.0 * mean * mean * mean * mean
        )
        return m4 / (m2 * m2) - 3.0

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._prev_mid = 0
        self._returns.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._sum_cb = 0.0
        self._sum_qt = 0.0
        self._count = 0
