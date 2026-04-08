from __future__ import annotations

from typing import Callable

from gnomepy.java.schemas import JavaMBP10Schema
from gnomepy_research.signals.base import Signal


def _is_signal(obj) -> bool:
    return isinstance(obj, Signal)


class CompositeSignal(Signal[float]):
    """Combines two signals with a binary operator on their values."""

    def __init__(
        self,
        left: Signal,
        right: Signal,
        op: Callable[[float, float], float],
        op_name: str,
    ):
        self.left = left
        self.right = right
        self.op = op
        self.op_name = op_name

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        self.left.update(timestamp, data)
        self.right.update(timestamp, data)

    def value(self) -> float:
        lv = float(self.left.value())
        rv = float(self.right.value())
        if rv == 0 and self.op_name == "/":
            return 0.0
        return self.op(lv, rv)

    def is_ready(self) -> bool:
        return self.left.is_ready() and self.right.is_ready()

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()


class ScalarOpSignal(Signal[float]):
    """Combines a signal with a scalar constant."""

    def __init__(
        self,
        signal: Signal,
        scalar: float,
        op: Callable[[float, float], float],
        op_name: str,
        scalar_left: bool = False,
    ):
        self.signal = signal
        self.scalar = scalar
        self.op = op
        self.op_name = op_name
        self.scalar_left = scalar_left

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        self.signal.update(timestamp, data)

    def value(self) -> float:
        sv = float(self.signal.value())
        if self.scalar_left:
            if sv == 0 and self.op_name == "/":
                return 0.0
            return self.op(self.scalar, sv)
        if self.scalar == 0 and self.op_name == "/":
            return 0.0
        return self.op(sv, self.scalar)

    def is_ready(self) -> bool:
        return self.signal.is_ready()

    def reset(self) -> None:
        self.signal.reset()


def _make_op(left, right, op, op_name):
    if _is_signal(left) and _is_signal(right):
        return CompositeSignal(left, right, op, op_name)
    if _is_signal(left) and isinstance(right, (int, float)):
        return ScalarOpSignal(left, float(right), op, op_name, scalar_left=False)
    if isinstance(left, (int, float)) and _is_signal(right):
        return ScalarOpSignal(right, float(left), op, op_name, scalar_left=True)
    return NotImplemented
