from __future__ import annotations

from collections import deque

from gnomepy.java.enums import Side
from gnomepy.java.schemas import JavaMBP10Schema
from gnomepy_research.signals.flow.base import FlowSignal


_ADD_ACTIONS = {"Add", "Modify"}
_CANCEL_ACTIONS = {"Cancel"}


class LevelLiquidityDelta(FlowSignal):
    """Net liquidity change at a specific book level over a rolling horizon.

    Tracks Add vs Cancel volume from the action field on MBP10 updates
    for a given side and depth level. Modify is treated as an add.

    Args:
        side: Side.BID or Side.ASK
        level: Book depth level (0 = top of book).
        horizon_ns: Rolling window in nanoseconds.
        warmup_events: Minimum add/cancel events before is_ready().
    """

    def __init__(
        self,
        side: Side,
        level: int = 0,
        horizon_ns: int = 1_000_000_000,
        warmup_events: int = 20,
    ):
        self.side = side
        self.level = level
        self.horizon_ns = horizon_ns
        self.warmup_events = warmup_events

        self._events: deque[tuple[int, int]] = deque()
        self._add_volume = 0
        self._cancel_volume = 0
        self._event_count = 0

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        exchange_ts = data.timestamp_event
        self._evict(exchange_ts)

        if data.side != self.side or data.depth != self.level:
            return

        action = data.action
        size = data.size
        if size <= 0:
            return

        if action in _ADD_ACTIONS:
            self._add_volume += size
            self._events.append((exchange_ts, size))
            self._event_count += 1
        elif action in _CANCEL_ACTIONS:
            self._cancel_volume += size
            self._events.append((exchange_ts, -size))
            self._event_count += 1

    def _evict(self, now: int) -> None:
        cutoff = now - self.horizon_ns
        while self._events and self._events[0][0] < cutoff:
            _, signed_size = self._events.popleft()
            if signed_size > 0:
                self._add_volume -= signed_size
            else:
                self._cancel_volume -= (-signed_size)

    def value(self) -> float:
        return float(self._add_volume - self._cancel_volume)

    def is_ready(self) -> bool:
        return self._event_count >= self.warmup_events

    def reset(self) -> None:
        self._events.clear()
        self._add_volume = 0
        self._cancel_volume = 0
        self._event_count = 0
