from __future__ import annotations

import math
from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.volatility.base import VolatilitySignal


class VolatilityAsymmetry(VolatilitySignal):
    """Ratio of upside volatility to downside volatility.

    up_vol / down_vol where each is the stdev of returns in that direction.

    > 1 = upward moves are more volatile than downward (unusual).
    < 1 = downward moves are more volatile (typical in equities, crypto).
    1.0 = symmetric volatility.

    Args:
        horizon: Number of log-return observations in the rolling window.
        warmup: Ticks before is_ready() (defaults to horizon).
    """

    def __init__(self, horizon: int = 100, warmup: int | None = None):
        self.horizon = horizon
        self.warmup = horizon if warmup is None else warmup
        self._prev_mid = 0
        self._all: deque[tuple[float, bool]] = deque(maxlen=horizon)
        self._up_sum = 0.0
        self._up_sq = 0.0
        self._up_count = 0
        self._down_sum = 0.0
        self._down_sq = 0.0
        self._down_count = 0
        self._total_count = 0

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
            is_up = r > 0

            if self._total_count == self.horizon:
                old_r, old_up = self._all[0]
                if old_up:
                    self._up_sum -= old_r
                    self._up_sq -= old_r * old_r
                    self._up_count -= 1
                else:
                    self._down_sum -= old_r
                    self._down_sq -= old_r * old_r
                    self._down_count -= 1
            else:
                self._total_count += 1

            self._all.append((r, is_up))
            if is_up:
                self._up_sum += r
                self._up_sq += r * r
                self._up_count += 1
            else:
                self._down_sum += r
                self._down_sq += r * r
                self._down_count += 1

        self._prev_mid = mid

    def value(self) -> float:
        if self._up_count < 2 or self._down_count < 2:
            return 1.0
        up_mean = self._up_sum / self._up_count
        up_var = self._up_sq / self._up_count - up_mean * up_mean
        up_std = math.sqrt(max(up_var, 0.0))
        down_mean = self._down_sum / self._down_count
        down_var = self._down_sq / self._down_count - down_mean * down_mean
        down_std = math.sqrt(max(down_var, 0.0))
        if down_std <= 0.0:
            return 1.0
        return up_std / down_std

    def is_ready(self) -> bool:
        return self._total_count >= self.warmup

    def reset(self) -> None:
        self._prev_mid = 0
        self._all.clear()
        self._up_sum = 0.0
        self._up_sq = 0.0
        self._up_count = 0
        self._down_sum = 0.0
        self._down_sq = 0.0
        self._down_count = 0
        self._total_count = 0
