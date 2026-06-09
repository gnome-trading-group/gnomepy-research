from __future__ import annotations

from collections import deque

from gnomepy.java.enums import Side
from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal

_ADD_ACTIONS = {"Add", "Modify"}
_CANCEL_ACTION = "Cancel"


class SpoofDetector(FlowSignal):
    """L1 cancel-to-add volume ratio over a short rolling window.

    Measures what fraction of posted L1 volume is quickly withdrawn.
    High values (near 1.0) indicate most L1 size is being added and
    then immediately cancelled — a proxy for layering/spoofing activity.

    Note: MBP-10 data does not expose individual order IDs, so this
    measures aggregate L1 cancel/add ratios rather than tracking
    specific orders.

    Output in [0, 1].

    Args:
        detection_window_ns: Rolling window to measure cancel/add ratio.
        warmup_events: Minimum L1 events before is_ready() returns True.
    """

    def __init__(
        self,
        detection_window_ns: int = 1_000_000_000,
        warmup_events: int = 20,
    ):
        self.detection_window_ns = detection_window_ns
        self.warmup_events = warmup_events
        self._events: deque[tuple[int, int]] = deque()  # (ts, signed_size)
        self._add_vol = 0
        self._cancel_vol = 0
        self._event_count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if data.depth != 0:
            return
        action = data.action
        size = data.size
        if size <= 0:
            return
        side = data.side
        if side == Side.NONE:
            return

        if action in _ADD_ACTIONS:
            signed = size
        elif action == _CANCEL_ACTION:
            signed = -size
        else:
            return

        self._event_count += 1
        self._evict(timestamp)
        if signed > 0:
            self._add_vol += signed
        else:
            self._cancel_vol += (-signed)
        self._events.append((timestamp, signed))

    def _evict(self, now: int) -> None:
        cutoff = now - self.detection_window_ns
        while self._events and self._events[0][0] < cutoff:
            _, signed = self._events.popleft()
            if signed > 0:
                self._add_vol -= signed
            else:
                self._cancel_vol -= (-signed)

    def value(self) -> float:
        if self._add_vol <= 0:
            return 0.0
        return min(1.0, self._cancel_vol / self._add_vol)

    def is_ready(self) -> bool:
        return self._event_count >= self.warmup_events

    def reset(self) -> None:
        self._events.clear()
        self._add_vol = 0
        self._cancel_vol = 0
        self._event_count = 0
