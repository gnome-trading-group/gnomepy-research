from __future__ import annotations

from collections import deque

from gnomepy.java.enums import Side
from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal

_ADD_ACTIONS = {"Add", "Modify"}


class AddImbalance(FlowSignal):
    """Net new-order volume imbalance over a rolling window.

    (bid_adds - ask_adds) / (bid_adds + ask_adds), output in [-1, +1].
    Positive = more new bid-side orders being posted; negative = ask-side.

    Modify is treated as an add since it can increase size.

    Args:
        horizon_ns: Rolling window in nanoseconds.
        warmup_events: Minimum add events before is_ready() returns True.
    """

    def __init__(self, horizon_ns: int = 5_000_000_000, warmup_events: int = 20):
        self.horizon_ns = horizon_ns
        self.warmup_events = warmup_events
        self._events: deque[tuple[int, int]] = deque()  # (ts, signed_size)
        self._bid_adds = 0
        self._ask_adds = 0
        self._event_count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if data.action not in _ADD_ACTIONS:
            return
        size = data.size
        if size <= 0:
            return
        side = data.side
        if side == Side.NONE:
            return

        self._event_count += 1
        self._evict(timestamp)

        if side == Side.BID:
            self._bid_adds += size
            self._events.append((timestamp, size))
        else:
            self._ask_adds += size
            self._events.append((timestamp, -size))

    def _evict(self, now: int) -> None:
        cutoff = now - self.horizon_ns
        while self._events and self._events[0][0] < cutoff:
            _, signed = self._events.popleft()
            if signed > 0:
                self._bid_adds -= signed
            else:
                self._ask_adds -= (-signed)

    def value(self) -> float:
        total = self._bid_adds + self._ask_adds
        if total <= 0:
            return 0.0
        return (self._bid_adds - self._ask_adds) / total

    def is_ready(self) -> bool:
        return self._event_count >= self.warmup_events

    def reset(self) -> None:
        self._events.clear()
        self._bid_adds = 0
        self._ask_adds = 0
        self._event_count = 0
