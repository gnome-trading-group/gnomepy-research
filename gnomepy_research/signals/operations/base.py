from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from gnomepy.java.schemas import JavaMBP10Schema
from gnomepy_research.signals.base import Signal
from gnomepy_research.signals.fair_value.base import FairValueSignal
from gnomepy_research.signals.flow.base import FlowSignal
from gnomepy_research.signals.volatility.base import VolatilitySignal

T = TypeVar("T", int, float)


class Operation(ABC):
    """Stateful value processor wrapped around a Signal.

    Receives a scalar via update() and exposes a filtered/smoothed/combined
    output via value(). Subclasses that change the semantic type of the
    signal (e.g. Log turns a price into a return) should set
    preserves_scale = False.
    """

    preserves_scale: bool = True

    @abstractmethod
    def update(self, value: float) -> None: ...

    @abstractmethod
    def value(self) -> float: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

    def reset(self) -> None:
        return None


class _OperationAdapter(Signal[T], Generic[T]):
    """Wraps a Signal + Operation, exposing the Signal's interface.

    Concrete subclasses pin T and inherit from the matching Signal subclass
    so isinstance() checks against the original signal type still hold.
    """

    def __init__(self, signal: Signal[T], op: Operation):
        self.signal = signal
        self.op = op

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        self.signal.update(timestamp, data)
        if self.signal.is_ready():
            self.op.update(float(self.signal.value()))

    def is_ready(self) -> bool:
        return self.signal.is_ready() and self.op.is_ready()

    def reset(self) -> None:
        self.signal.reset()
        self.op.reset()


class FairValueOperationAdapter(_OperationAdapter[int], FairValueSignal):
    def value(self) -> int:
        return int(self.op.value())


class VolatilityOperationAdapter(_OperationAdapter[float], VolatilitySignal):
    def value(self) -> float:
        return self.op.value()


class FlowOperationAdapter(_OperationAdapter[float], FlowSignal):
    def value(self) -> float:
        return self.op.value()


def apply(op: Operation, signal: Signal) -> Signal:
    """Wrap a signal with an operation, preserving the signal's type."""
    if isinstance(signal, FairValueSignal):
        return FairValueOperationAdapter(signal, op)
    if isinstance(signal, FlowSignal):
        return FlowOperationAdapter(signal, op)
    if isinstance(signal, VolatilitySignal):
        return VolatilityOperationAdapter(signal, op)
    raise TypeError(
        f"Cannot apply operation to {type(signal).__name__}. "
        f"Expected FairValueSignal, VolatilitySignal, or FlowSignal."
    )
