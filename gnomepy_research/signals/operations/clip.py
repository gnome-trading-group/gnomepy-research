from __future__ import annotations

from gnomepy_research.signals.operations.base import Operation, apply


class ClipOperation(Operation):
    """Clamps the signal value to [lo, hi]."""

    preserves_scale = True

    def __init__(self, lo: float, hi: float):
        self.lo = lo
        self.hi = hi
        self._value = 0.0
        self._ready = False

    def update(self, value: float) -> None:
        if value < self.lo:
            self._value = self.lo
        elif value > self.hi:
            self._value = self.hi
        else:
            self._value = value
        self._ready = True

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        self._value = 0.0
        self._ready = False


def Clip(signal, lo: float, hi: float):
    """Clamp any signal's output to [lo, hi], preserving its type."""
    return apply(ClipOperation(lo=lo, hi=hi), signal)
