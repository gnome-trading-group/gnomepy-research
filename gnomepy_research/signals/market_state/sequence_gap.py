from __future__ import annotations

from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.market_state.base import MarketStateSignal


class SequenceGap(MarketStateSignal):
    """Rolling rate of detected sequence number gaps per tick.

    A gap is detected when the sequence number jumps by more than 1,
    indicating dropped or out-of-order messages from the feed.

    Output = gap_count / horizon (gaps per tick in the window).
    Non-zero values signal feed reliability issues.

    Args:
        horizon: Rolling window size in ticks.
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, horizon: int = 100, warmup: int = 100):
        self.horizon = horizon
        self.warmup = warmup
        self._prev_seq = -1
        self._ticks: deque[bool] = deque(maxlen=horizon)
        self._gap_count = 0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        seq = data.sequence
        is_gap = False
        if self._prev_seq >= 0 and seq > 0 and seq != self._prev_seq + 1:
            is_gap = True

        if seq > 0:
            self._prev_seq = seq

        if self._count == self.horizon:
            if self._ticks[0]:
                self._gap_count -= 1
        else:
            self._count += 1

        self._ticks.append(is_gap)
        if is_gap:
            self._gap_count += 1

    def value(self) -> float:
        if self._count == 0:
            return 0.0
        return self._gap_count / self._count

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._prev_seq = -1
        self._ticks.clear()
        self._gap_count = 0
        self._count = 0
