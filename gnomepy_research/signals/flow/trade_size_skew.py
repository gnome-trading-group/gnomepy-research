from __future__ import annotations

import math
from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class TradeSizeSkew(FlowSignal):
    """Rolling skewness of trade size distribution.

    Positive skew = occasional large block trades (institutional flow).
    Near-zero skew = uniform, algorithmically uniform flow.

    Args:
        horizon: Number of trades in the rolling window.
        min_trades: Minimum trades for is_ready() (defaults to horizon).
    """

    def __init__(self, horizon: int = 100, min_trades: int | None = None):
        self.horizon = horizon
        self.min_trades = horizon if min_trades is None else min_trades
        self._sizes: deque[float] = deque(maxlen=horizon)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._sum_cb = 0.0
        self._count = 0
        self._trade_count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if not is_trade_event(data):
            return

        self._trade_count += 1
        size = float(data.size)

        if self._count == self.horizon:
            old = self._sizes[0]
            self._sum -= old
            self._sum_sq -= old * old
            self._sum_cb -= old * old * old
        else:
            self._count += 1

        self._sizes.append(size)
        self._sum += size
        self._sum_sq += size * size
        self._sum_cb += size * size * size

    def value(self) -> float:
        n = self._count
        if n < 3:
            return 0.0
        mean = self._sum / n
        variance = self._sum_sq / n - mean * mean
        if variance <= 0.0:
            return 0.0
        std = math.sqrt(variance)
        m3 = self._sum_cb / n - 3.0 * mean * self._sum_sq / n + 2.0 * mean * mean * mean
        return m3 / (std * std * std)

    def is_ready(self) -> bool:
        return self._trade_count >= self.min_trades

    def reset(self) -> None:
        self._sizes.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._sum_cb = 0.0
        self._count = 0
        self._trade_count = 0
