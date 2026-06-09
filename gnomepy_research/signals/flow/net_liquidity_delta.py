from __future__ import annotations

from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal

_ADD_ACTIONS = {"Add", "Modify"}
_CANCEL_ACTIONS = {"Cancel"}


class NetLiquidityDelta(FlowSignal):
    """Net aggregate liquidity change (add - cancel) across all book levels.

    Positive = book is growing (more adding than cancelling).
    Negative = book is shrinking (informed withdrawal).

    Args:
        horizon_ns: Rolling window in nanoseconds.
        warmup_events: Minimum add/cancel events before is_ready() returns True.
    """

    def __init__(self, horizon_ns: int = 5_000_000_000, warmup_events: int = 20):
        self.horizon_ns = horizon_ns
        self.warmup_events = warmup_events
        self._events: deque[tuple[int, int]] = deque()  # (ts, signed_size)
        self._net = 0
        self._event_count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        action = data.action
        size = data.size
        if size <= 0:
            return

        if action in _ADD_ACTIONS:
            signed = size
        elif action in _CANCEL_ACTIONS:
            signed = -size
        else:
            return

        self._event_count += 1
        self._evict(timestamp)
        self._net += signed
        self._events.append((timestamp, signed))

    def _evict(self, now: int) -> None:
        cutoff = now - self.horizon_ns
        while self._events and self._events[0][0] < cutoff:
            _, signed = self._events.popleft()
            self._net -= signed

    def value(self) -> float:
        return float(self._net)

    def is_ready(self) -> bool:
        return self._event_count >= self.warmup_events

    def reset(self) -> None:
        self._events.clear()
        self._net = 0
        self._event_count = 0
