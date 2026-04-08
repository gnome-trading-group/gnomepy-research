from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.fair_value.base import FairValueSignal


class MicropriceFairValue(FairValueSignal):
    """L1 microprice: volume-weighted mid using top-of-book.

    microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)

    When ask_size >> bid_size (selling pressure), microprice shifts toward bid.
    When bid_size >> ask_size (buying pressure), microprice shifts toward ask.
    """

    def __init__(self):
        self._fair_value = 0
        self._ready = False

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        bid_size = data.bid_size(0)
        ask_size = data.ask_size(0)

        if bid <= 0 or ask <= 0 or bid_size + ask_size <= 0:
            return

        self._fair_value = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
        self._ready = True

    def value(self) -> int:
        return int(self._fair_value)

    def is_ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        self._fair_value = 0
        self._ready = False


class WeightedMicropriceFairValue(FairValueSignal):
    """Multi-level weighted microprice using MBP book depth.

    Uses levels 0..num_levels from MBP10 data. Deeper levels are weighted
    less via exponential decay.

    weighted_microprice = Σ(w_i * (bid_i * ask_size_i + ask_i * bid_size_i)) /
                          Σ(w_i * (bid_size_i + ask_size_i))

    where w_i = decay ^ i

    Args:
        num_levels: Number of book levels to use (1-10).
        decay: Weight decay per level.
    """

    def __init__(self, num_levels: int = 5, decay: float = 0.5):
        self.num_levels = min(num_levels, 10)
        self.decay = decay
        self._fair_value = 0
        self._ready = False
        self._weights = [decay ** i for i in range(self.num_levels)]

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        numerator = 0.0
        denominator = 0.0

        for i in range(self.num_levels):
            bid = data.bid_price(i)
            ask = data.ask_price(i)
            bid_size = data.bid_size(i)
            ask_size = data.ask_size(i)

            if bid <= 0 or ask <= 0 or bid_size + ask_size <= 0:
                continue

            w = self._weights[i]
            numerator += w * (bid * ask_size + ask * bid_size)
            denominator += w * (bid_size + ask_size)

        if denominator <= 0:
            return

        self._fair_value = numerator / denominator
        self._ready = True

    def value(self) -> int:
        return int(self._fair_value)

    def is_ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        self._fair_value = 0
        self._ready = False
