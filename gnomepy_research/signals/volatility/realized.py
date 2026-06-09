from __future__ import annotations

import math
from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.volatility.base import VolatilitySignal


class RealizedVolatility(VolatilitySignal):
    """Rolling standard deviation of mid-price log returns, in basis points.

    Args:
        horizon: Number of log-return observations in the rolling window.
        warmup: Ticks before is_ready() returns True (defaults to horizon).
    """

    def __init__(self, horizon: int = 100, warmup: int | None = None):
        self.horizon = horizon
        self.warmup = horizon if warmup is None else warmup
        self._prev_mid = 0
        self._returns: deque[float] = deque(maxlen=horizon)
        self._sum = 0.0
        self._sum_sq = 0.0
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
            log_ret = math.log(mid / self._prev_mid)

            if self._count == self.horizon:
                old = self._returns[0]
                self._sum -= old
                self._sum_sq -= old * old
            else:
                self._count += 1

            self._returns.append(log_ret)
            self._sum += log_ret
            self._sum_sq += log_ret * log_ret

        self._prev_mid = mid

    def value(self) -> float:
        if self._count < 2:
            return 0.0
        mean = self._sum / self._count
        variance = self._sum_sq / self._count - mean * mean
        return math.sqrt(max(variance, 0.0)) * 10000.0

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._prev_mid = 0
        self._returns.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0
