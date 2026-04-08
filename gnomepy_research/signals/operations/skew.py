from __future__ import annotations

import math
from collections import deque

from gnomepy_research.signals.operations.base import Operation, apply


class SkewOperation(Operation):
    """Rolling population skewness over a fixed-size window."""

    preserves_scale = False

    def __init__(self, horizon: int = 100):
        self.horizon = horizon
        self._values: deque[float] = deque(maxlen=horizon)
        self._count = 0

    def update(self, value: float) -> None:
        if self._count < self.horizon:
            self._count += 1
        self._values.append(value)

    def value(self) -> float:
        if self._count < 3:
            return 0.0

        n = self._count
        mean = sum(self._values) / n
        m2 = 0.0
        m3 = 0.0
        for v in self._values:
            d = v - mean
            m2 += d * d
            m3 += d * d * d
        m2 /= n
        m3 /= n

        if m2 <= 0.0:
            return 0.0

        std = math.sqrt(m2)
        return m3 / (std * std * std)

    def is_ready(self) -> bool:
        return self._count >= self.horizon

    def reset(self) -> None:
        self._values.clear()
        self._count = 0


def Skew(signal, horizon: int = 100):
    """Apply rolling skewness to any signal, preserving its type."""
    return apply(SkewOperation(horizon=horizon), signal)
