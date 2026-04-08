from __future__ import annotations

from gnomepy_research.signals.operations.base import Operation, apply


class PctChangeOperation(Operation):
    """Percent change between consecutive values."""

    preserves_scale = False

    def __init__(self, warmup: int = 2):
        self.warmup = warmup
        self._prev = 0.0
        self._value = 0.0
        self._count = 0

    def update(self, value: float) -> None:
        if self._count > 0 and self._prev != 0.0:
            self._value = (value - self._prev) / self._prev
        self._prev = value
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._prev = 0.0
        self._value = 0.0
        self._count = 0


def PctChange(signal, warmup: int = 2):
    """Apply percent change to any signal, preserving its type."""
    return apply(PctChangeOperation(warmup=warmup), signal)
