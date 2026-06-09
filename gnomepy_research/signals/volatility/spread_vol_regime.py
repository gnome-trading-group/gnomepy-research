from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.volatility.base import VolatilitySignal


class SpreadVolRegime(VolatilitySignal):
    """Current spread relative to its EWMA — a spread regime indicator.

    Values > 1 = spread widening (risk-off or low liquidity regime).
    Values < 1 = spread tightening relative to background.
    Values near 1 = stable spread regime.

    Args:
        alpha: EWMA smoothing factor for the background spread (0.999 = very slow).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, alpha: float = 0.999, warmup: int = 100):
        self.alpha = alpha
        self.warmup = warmup
        self._ewma_spread = 0.0
        self._last_spread = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 10000.0

        self._last_spread = spread_bps
        if self._count == 0:
            self._ewma_spread = spread_bps
        else:
            self._ewma_spread = self.alpha * self._ewma_spread + (1.0 - self.alpha) * spread_bps
        self._count += 1

    def value(self) -> float:
        if self._ewma_spread <= 0.0:
            return 1.0
        return self._last_spread / self._ewma_spread

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._ewma_spread = 0.0
        self._last_spread = 0.0
        self._count = 0
