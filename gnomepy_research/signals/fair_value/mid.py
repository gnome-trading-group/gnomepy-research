from __future__ import annotations

from gnomepy.java.schemas import JavaMBP10Schema
from gnomepy_research.signals.fair_value.base import FairValueSignal


class MidFairValue(FairValueSignal):
    """Fair value = simple mid price."""

    def __init__(self):
        self._mid = 0

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid > 0 and ask > 0:
            self._mid = (bid + ask) // 2

    def value(self) -> int:
        return self._mid

    def is_ready(self) -> bool:
        return self._mid > 0

    def reset(self) -> None:
        self._mid = 0
