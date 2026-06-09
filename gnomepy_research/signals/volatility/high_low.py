from __future__ import annotations

import math
from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.volatility.base import VolatilitySignal


_PARKINSON_FACTOR = 1.0 / (4.0 * math.log(2.0))


class HighLowVolatility(VolatilitySignal):
    """Parkinson (1980) volatility estimator using rolling high-low range.

    sigma = sqrt(1/(4*ln2) * log(high/low)^2)

    More efficient than close-to-close volatility when drift is small.
    Output in basis points.

    Args:
        horizon: Number of mid-price ticks in the rolling window.
        warmup: Ticks before is_ready() returns True (defaults to horizon).
    """

    def __init__(self, horizon: int = 100, warmup: int | None = None):
        self.horizon = horizon
        self.warmup = horizon if warmup is None else warmup
        self._mids: deque[int] = deque(maxlen=horizon)
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) // 2
        if mid <= 0:
            return

        self._mids.append(mid)
        if self._count < self.horizon:
            self._count += 1

    def value(self) -> float:
        if self._count < 2:
            return 0.0
        h = max(self._mids)
        lo = min(self._mids)
        if lo <= 0:
            return 0.0
        log_hl = math.log(h / lo)
        return math.sqrt(_PARKINSON_FACTOR * log_hl * log_hl) * 10000.0

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._mids.clear()
        self._count = 0
