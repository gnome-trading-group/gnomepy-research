from __future__ import annotations

from collections import deque

from gnomepy_research.signals.operations.base import Operation, apply


class AutocorrelationOperation(Operation):
    """Rolling lag-1 autocorrelation over a fixed-size window."""

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

        var = 0.0
        cov = 0.0
        prev = self._values[0]
        for i in range(1, n):
            curr = self._values[i]
            var += (curr - mean) * (curr - mean)
            cov += (curr - mean) * (prev - mean)
            prev = curr

        var += (self._values[0] - mean) * (self._values[0] - mean)

        if var <= 0.0:
            return 0.0

        return cov / var

    def is_ready(self) -> bool:
        return self._count >= self.horizon

    def reset(self) -> None:
        self._values.clear()
        self._count = 0


def Autocorrelation(signal, horizon: int = 100):
    """Apply rolling lag-1 autocorrelation to any signal, preserving its type."""
    return apply(AutocorrelationOperation(horizon=horizon), signal)
