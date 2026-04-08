from __future__ import annotations

from gnomepy.java.enums import Side
from gnomepy.java.schemas import JavaMBP10Schema
from gnomepy_research.signals.flow.base import FlowSignal


class LevelStaleness(FlowSignal):
    """Staleness of a single book level in nanoseconds.

    Tracks how long a specific price level (side + depth) has been
    unchanged. Resets to 0 when the level's price or size changes.

    Args:
        side: Side.BID or Side.ASK
        level: Book depth level (0 = top of book).
        warmup_ticks: Minimum updates before is_ready() returns True.
    """

    def __init__(self, side: Side, level: int = 0, warmup_ticks: int = 1):
        self.side = side
        self.level = level
        self.warmup_ticks = warmup_ticks

        self._last_price = 0
        self._last_size = 0
        self._last_change_ts = 0
        self._staleness_ns = 0
        self._tick_count = 0

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        exchange_ts = data.timestamp_event

        if self.side == Side.BID:
            price = data.bid_price(self.level)
            size = data.bid_size(self.level)
        else:
            price = data.ask_price(self.level)
            size = data.ask_size(self.level)

        if price != self._last_price or size != self._last_size:
            self._last_price = price
            self._last_size = size
            self._last_change_ts = exchange_ts

        if self._last_change_ts > 0:
            self._staleness_ns = exchange_ts - self._last_change_ts

        self._tick_count += 1

    def value(self) -> float:
        return float(self._staleness_ns)

    def is_ready(self) -> bool:
        return self._tick_count >= self.warmup_ticks

    def reset(self) -> None:
        self._last_price = 0
        self._last_size = 0
        self._last_change_ts = 0
        self._staleness_ns = 0
        self._tick_count = 0
