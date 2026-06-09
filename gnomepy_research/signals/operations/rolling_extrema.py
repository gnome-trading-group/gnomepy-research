from __future__ import annotations

from collections import deque

from gnomepy_research.signals.operations.base import Operation, apply


class RollingMaxOperation(Operation):
    """Rolling maximum over a fixed tick window."""

    preserves_scale = True

    def __init__(self, horizon: int = 100):
        self.horizon = horizon
        self._values: deque[float] = deque(maxlen=horizon)
        self._count = 0

    def update(self, value: float) -> None:
        self._values.append(value)
        if self._count < self.horizon:
            self._count += 1

    def value(self) -> float:
        return max(self._values) if self._values else 0.0

    def is_ready(self) -> bool:
        return self._count >= self.horizon

    def reset(self) -> None:
        self._values.clear()
        self._count = 0


class RollingMinOperation(Operation):
    """Rolling minimum over a fixed tick window."""

    preserves_scale = True

    def __init__(self, horizon: int = 100):
        self.horizon = horizon
        self._values: deque[float] = deque(maxlen=horizon)
        self._count = 0

    def update(self, value: float) -> None:
        self._values.append(value)
        if self._count < self.horizon:
            self._count += 1

    def value(self) -> float:
        return min(self._values) if self._values else 0.0

    def is_ready(self) -> bool:
        return self._count >= self.horizon

    def reset(self) -> None:
        self._values.clear()
        self._count = 0


def RollingMax(signal, horizon: int = 100):
    """Apply rolling max to any signal, preserving its type."""
    return apply(RollingMaxOperation(horizon=horizon), signal)


def RollingMin(signal, horizon: int = 100):
    """Apply rolling min to any signal, preserving its type."""
    return apply(RollingMinOperation(horizon=horizon), signal)
