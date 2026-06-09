from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal


class PriceAnchor(FlowSignal):
    """Distance of current mid from its EWMA, in basis points.

    Measures price displacement from a slowly-moving fair value anchor.
    Positive = above anchor (potential reversion down).
    Negative = below anchor (potential reversion up).

    Args:
        alpha: EWMA smoothing factor (0.999 = very slow anchor).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, alpha: float = 0.999, warmup: int = 100):
        self.alpha = alpha
        self.warmup = warmup
        self._ewma = 0.0
        self._last_mid = 0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return
        mid = (bid + ask) // 2
        self._last_mid = mid
        if self._count == 0:
            self._ewma = float(mid)
        else:
            self._ewma = self.alpha * self._ewma + (1.0 - self.alpha) * mid
        self._count += 1

    def value(self) -> float:
        if self._ewma <= 0.0:
            return 0.0
        return (self._last_mid - self._ewma) / self._ewma * 10000.0

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._ewma = 0.0
        self._last_mid = 0
        self._count = 0
