from __future__ import annotations

from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.volatility.base import VolatilitySignal


class BidAskBounce(VolatilitySignal):
    """Fraction of mid-price changes that immediately reverse direction.

    Related to Roll's (1984) implicit spread measure. A high bounce rate
    indicates microstructure noise (bid-ask bouncing); a low bounce rate
    indicates trending price action.

    Output in [0, 1]: 1.0 = every price change reverses.

    Args:
        horizon: Number of directional changes in the rolling window.
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, horizon: int = 50, warmup: int = 50):
        self.horizon = horizon
        self.warmup = warmup
        self._prev_mid = 0
        self._prev_direction = 0
        self._changes: deque[bool] = deque(maxlen=horizon)
        self._bounce_count = 0
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

            if direction != 0 and self._prev_direction != 0:
                is_bounce = direction != self._prev_direction

                if self._count == self.horizon:
                    if self._changes[0]:
                        self._bounce_count -= 1
                else:
                    self._count += 1

                self._changes.append(is_bounce)
                if is_bounce:
                    self._bounce_count += 1

            if direction != 0:
                self._prev_direction = direction

        self._prev_mid = mid

    def value(self) -> float:
        if self._count == 0:
            return 0.0
        return self._bounce_count / self._count

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._prev_mid = 0
        self._prev_direction = 0
        self._changes.clear()
        self._bounce_count = 0
        self._count = 0
