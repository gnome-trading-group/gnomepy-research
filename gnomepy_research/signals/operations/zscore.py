from __future__ import annotations

import math
from collections import deque

from gnomepy_research.signals.operations.base import Operation, apply


class ZscoreOperation(Operation):
    """Z-score: (current_value - rolling_mean) / rolling_std."""

    preserves_scale = False

    def __init__(self, horizon: int = 100):
        self.horizon = horizon
        self._values: deque[float] = deque(maxlen=horizon)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def update(self, value: float) -> None:
        if self._count == self.horizon:
            old = self._values[0]
            self._sum -= old
            self._sum_sq -= old * old
        else:
            self._count += 1
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value

    def value(self) -> float:
        if self._count < 2:
            return 0.0
        mean = self._sum / self._count
        variance = self._sum_sq / self._count - mean * mean
        std = math.sqrt(max(variance, 0.0))
        if std == 0.0:
            return 0.0
        return (self._values[-1] - mean) / std

    def is_ready(self) -> bool:
        return self._count >= self.horizon

    def reset(self) -> None:
        self._values.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0


def Zscore(signal, horizon: int = 100):
    """Apply rolling z-score normalization to any signal, preserving its type."""
    return apply(ZscoreOperation(horizon=horizon), signal)
