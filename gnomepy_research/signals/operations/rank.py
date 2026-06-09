from __future__ import annotations

from collections import deque

from gnomepy_research.signals.operations.base import Operation, apply


class RankOperation(Operation):
    """Percentile rank of the current value within a rolling window, output in [0, 1]."""

    preserves_scale = False

    def __init__(self, horizon: int = 100):
        self.horizon = horizon
        self._values: deque[float] = deque(maxlen=horizon)
        self._count = 0

    def update(self, value: float) -> None:
        self._values.append(value)
        if self._count < self.horizon:
            self._count += 1

    def value(self) -> float:
        if self._count < 2:
            return 0.5
        current = self._values[-1]
        below = 0
        for v in self._values:
            if v < current:
                below += 1
        return below / (self._count - 1)

    def is_ready(self) -> bool:
        return self._count >= self.horizon

    def reset(self) -> None:
        self._values.clear()
        self._count = 0


def Rank(signal, horizon: int = 100):
    """Apply rolling percentile rank to any signal, preserving its type."""
    return apply(RankOperation(horizon=horizon), signal)
