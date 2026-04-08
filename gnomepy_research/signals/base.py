from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from gnomepy.java.schemas import JavaMBP10Schema

T = TypeVar("T", int, float)


class Signal(ABC, Generic[T]):
    """Stateful signal: ingests market data, exposes a typed scalar value.

    Subclasses pin T to either int (e.g. fair-value in raw price units)
    or float (volatility, flow). Arithmetic operators on the base produce
    float-valued composite signals; subclasses may override the dunders if
    they need to constrain output type.
    """

    @abstractmethod
    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        """Feed market data for the current tick."""
        ...

    @abstractmethod
    def value(self) -> T:
        """Current scalar value of the signal."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """False during warmup."""
        ...

    def reset(self) -> None:
        """Clear internal state. Default no-op; override if stateful."""
        return None

    # ---- Arithmetic ------------------------------------------------------
    # Implementations live in operations.composite to avoid circular imports.

    def __add__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(self, other, operator.add, "+")

    def __radd__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(other, self, operator.add, "+")

    def __sub__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(self, other, operator.sub, "-")

    def __rsub__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(other, self, operator.sub, "-")

    def __mul__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(self, other, operator.mul, "*")

    def __rmul__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(other, self, operator.mul, "*")

    def __truediv__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(self, other, operator.truediv, "/")

    def __rtruediv__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(other, self, operator.truediv, "/")

    def __pow__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(self, other, operator.pow, "**")

    def __rpow__(self, other):
        from gnomepy_research.signals.operations.composite import _make_op
        return _make_op(other, self, operator.pow, "**")

    def __neg__(self):
        from gnomepy_research.signals.operations.unary import Negate
        return Negate(self)

    def __abs__(self):
        from gnomepy_research.signals.operations.unary import Abs
        return Abs(self)
