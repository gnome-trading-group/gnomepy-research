from __future__ import annotations

from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class QueueImbalanceDelta(BookSignal):
    """Rate of change of depth imbalance over a fixed tick window.

    output = DepthImbalance_now - DepthImbalance_{diff_ticks ago}

    Positive and rising = imbalance accelerating in the bid direction.
    A static imbalance has delta near 0; a tilting book has nonzero delta.

    Args:
        num_levels: Book levels for the imbalance computation.
        diff_ticks: Lag window for the difference.
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, diff_ticks: int = 10, warmup: int = 20):
        self.num_levels = num_levels
        self.diff_ticks = diff_ticks
        self.warmup = warmup
        self._buffer: deque[float] = deque(maxlen=diff_ticks + 1)
        self._count = 0

    def _imbalance(self, data: Mbp10Schema) -> float:
        bid_vol = 0
        ask_vol = 0
        for i in range(self.num_levels):
            bid_vol += data.bid_size(i)
            ask_vol += data.ask_size(i)
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        imb = self._imbalance(data)
        self._buffer.append(imb)
        self._count += 1

    def value(self) -> float:
        if len(self._buffer) < self.diff_ticks + 1:
            return 0.0
        return self._buffer[-1] - self._buffer[0]

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._buffer.clear()
        self._count = 0
