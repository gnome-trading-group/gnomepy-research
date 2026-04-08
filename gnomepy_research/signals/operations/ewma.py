from __future__ import annotations

from gnomepy_research.signals.operations.base import Operation, apply


class EWMAOperation(Operation):
    """Exponential weighted moving average.

    value = alpha * previous + (1 - alpha) * new

    Args:
        alpha: Smoothing factor. 0.99 = slow/smooth, 0.9 = fast/responsive.
        warmup: Minimum updates before is_ready() returns True.
    """

    def __init__(self, alpha: float = 0.95, warmup: int = 1):
        self.alpha = alpha
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def update(self, value: float) -> None:
        if self._count == 0:
            self._value = value
        else:
            self._value = self.alpha * self._value + (1 - self.alpha) * value
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0


def EWMA(signal, alpha: float = 0.95, warmup: int = 1):
    """Apply EWMA smoothing to any signal, preserving its type."""
    return apply(EWMAOperation(alpha=alpha, warmup=warmup), signal)
