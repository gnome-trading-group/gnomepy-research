from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.fair_value.base import FairValueSignal


class ImbalanceAdjustedMid(FairValueSignal):
    """Fair value: mid shifted by depth imbalance (Stoikov 2018).

    fair_value = mid + (spread / 2) * imbalance
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

    Outperforms simple mid by incorporating the directional information
    from asymmetric book depth.

    Args:
        num_levels: Price levels to include in depth imbalance (max 10).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 1, warmup: int = 10):
        self.num_levels = num_levels
        self.warmup = warmup
        self._value = 0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid0 = data.bid_price(0)
        ask0 = data.ask_price(0)
        if bid0 <= 0 or ask0 <= 0:
            return

        mid = (bid0 + ask0) // 2
        spread = ask0 - bid0

        bid_vol = 0
        ask_vol = 0
        for i in range(self.num_levels):
            bid_vol += data.bid_size(i)
            ask_vol += data.ask_size(i)

        total = bid_vol + ask_vol
        if total <= 0:
            self._value = mid
        else:
            imbalance = (bid_vol - ask_vol) / total
            self._value = int(mid + spread / 2.0 * imbalance)

        self._count += 1

    def value(self) -> int:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0
        self._count = 0
