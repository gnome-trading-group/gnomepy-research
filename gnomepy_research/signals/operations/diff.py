from __future__ import annotations

from collections import deque

from gnomepy_research.signals.operations.base import Operation, apply


class DiffOperation(Operation):
    """First difference: value(t) - value(t-n)."""

    preserves_scale = False

    def __init__(self, n: int = 1):
        self.n = n
        self._buffer: deque[float] = deque(maxlen=n + 1)

    def update(self, value: float) -> None:
        self._buffer.append(value)

    def value(self) -> float:
        return self._buffer[-1] - self._buffer[0]

    def is_ready(self) -> bool:
        return len(self._buffer) >= self.n + 1

    def reset(self) -> None:
        self._buffer.clear()


def Diff(signal, n: int = 1):
    """Apply first difference to any signal, preserving its type."""
    return apply(DiffOperation(n=n), signal)
