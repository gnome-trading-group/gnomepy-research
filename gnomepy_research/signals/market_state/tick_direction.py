from __future__ import annotations

from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.market_state.base import MarketStateSignal


class TickDirection(MarketStateSignal):
    """Normalized rolling count of up-ticks minus down-ticks.

    (upticks - downticks) / (upticks + downticks), output in [-1, +1].
    Positive = price action predominantly upward; negative = downward.
    Near 0 = oscillating (noise-dominated).

    Args:
        horizon: Number of mid-price changes in the rolling window.
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, horizon: int = 50, warmup: int = 50):
        self.horizon = horizon
        self.warmup = warmup
        self._prev_mid = 0
        self._ticks: deque[int] = deque(maxlen=horizon)  # +1, -1, or 0
        self._up = 0
        self._down = 0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return
        mid = (bid + ask) // 2

        if self._prev_mid > 0:
            if mid > self._prev_mid:
                direction = 1
            elif mid < self._prev_mid:
                direction = -1
            else:
                direction = 0

            if self._count == self.horizon:
                old = self._ticks[0]
                if old == 1:
                    self._up -= 1
                elif old == -1:
                    self._down -= 1
            else:
                self._count += 1

            self._ticks.append(direction)
            if direction == 1:
                self._up += 1
            elif direction == -1:
                self._down += 1

        self._prev_mid = mid

    def value(self) -> float:
        total = self._up + self._down
        if total == 0:
            return 0.0
        return (self._up - self._down) / total

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._prev_mid = 0
        self._ticks.clear()
        self._up = 0
        self._down = 0
        self._count = 0
