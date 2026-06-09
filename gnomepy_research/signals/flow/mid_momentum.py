from __future__ import annotations

import math
from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal


class MidMomentum(FlowSignal):
    """Signed cumulative return normalized by realized vol — a t-statistic of trend.

    output = sum(log_returns) / (std(log_returns) * sqrt(n))

    Positive = statistically significant upward trend over the horizon.
    Negative = statistically significant downward trend.
    Near 0 = no detectable trend.

    Args:
        horizon: Number of log-return observations in the rolling window.
        warmup: Ticks before is_ready() (defaults to horizon).
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
            r = math.log(mid / self._prev_mid)

            if self._count == self.horizon:
                old = self._returns[0]
                self._sum -= old
                self._sum_sq -= old * old
            else:
                self._count += 1

            self._returns.append(r)
            self._sum += r
            self._sum_sq += r * r

        self._prev_mid = mid

    def value(self) -> float:
        n = self._count
        if n < 2:
            return 0.0
        mean = self._sum / n
        variance = self._sum_sq / n - mean * mean
        std = math.sqrt(max(variance, 0.0))
        if std == 0.0:
            return 0.0
        return self._sum / (std * math.sqrt(n))

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._prev_mid = 0
        self._returns.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0
