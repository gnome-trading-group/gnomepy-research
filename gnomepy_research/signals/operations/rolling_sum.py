from __future__ import annotations

from collections import deque

from gnomepy_research.signals.operations.base import Operation, apply


class RollingSumOperation(Operation):
    """Rolling sum over a fixed tick window."""

    preserves_scale = True

    def __init__(self, horizon: int = 100):
        self.horizon = horizon
        self._values: deque[float] = deque(maxlen=horizon)
        self._sum = 0.0
        self._count = 0

    def update(self, value: float) -> None:
        if self._count == self.horizon:
            self._sum -= self._values[0]
        else:
            self._count += 1
        self._values.append(value)
        self._sum += value

    def value(self) -> float:
        return self._sum

    def is_ready(self) -> bool:
        return self._count >= self.horizon

    def reset(self) -> None:
        self._values.clear()
        self._sum = 0.0
        self._count = 0


def RollingSum(signal, horizon: int = 100):
    """Apply rolling sum to any signal, preserving its type."""
    return apply(RollingSumOperation(horizon=horizon), signal)
