from __future__ import annotations

from gnomepy.java.schemas import JavaMBP10Schema
from gnomepy_research.signals.volatility.base import VolatilitySignal


class SpreadVolatility(VolatilitySignal):
    """Volatility estimate derived from the bid-ask spread.

    The spread reflects the market's implied short-term volatility — wider spreads
    mean market makers expect more price uncertainty. Outputs the raw spread in bps
    each tick.

    Args:
        warmup_ticks: Minimum ticks before the signal is ready.
        scale: Multiplier on the spread to convert to vol estimate.
    """

    def __init__(self, warmup_ticks: int = 50, scale: float = 1.0):
        self.warmup_ticks = warmup_ticks
        self.scale = scale
        self._spread_bps = 0.0
        self._tick_count = 0

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2
        self._spread_bps = (ask - bid) / mid * 10000
        self._tick_count += 1

    def value(self) -> float:
        return self._spread_bps * self.scale

    def is_ready(self) -> bool:
        return self._tick_count >= self.warmup_ticks

    def reset(self) -> None:
        self._spread_bps = 0.0
        self._tick_count = 0
