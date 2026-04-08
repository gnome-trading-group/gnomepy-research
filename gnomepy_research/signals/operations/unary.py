from __future__ import annotations

import math as _math
from typing import Callable

from gnomepy_research.signals.operations.base import Operation, apply


class _UnaryOperation(Operation):
    """Stateless-ish operation that applies a scalar function to each value.

    `fn` returns the new output, or None to skip the update (e.g. invalid
    input). `is_ready()` becomes True after the first accepted value.
    """

    def __init__(self, fn: Callable[[float], float | None], preserves_scale: bool = True):
        self._fn = fn
        self.preserves_scale = preserves_scale
        self._value = 0.0
        self._ready = False

    def update(self, value: float) -> None:
        out = self._fn(value)
        if out is None:
            return
        self._value = out
        self._ready = True

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        self._value = 0.0
        self._ready = False


def _unary(name: str, fn: Callable[[float], float | None], preserves_scale: bool = True):
    """Build an Operation subclass + signal-wrapper function for a unary scalar fn."""
    cls = type(
        name,
        (_UnaryOperation,),
        {"__init__": lambda self, _fn=fn, _ps=preserves_scale: _UnaryOperation.__init__(self, _fn, _ps)},
    )

    def wrap(signal, _cls=cls):
        return apply(_cls(), signal)

    return cls, wrap


NegateOperation, Negate = _unary("NegateOperation", lambda v: -v)
AbsOperation, Abs = _unary("AbsOperation", abs)
LogOperation, Log = _unary(
    "LogOperation",
    lambda v: _math.log(v) if v > 0.0 else None,
    preserves_scale=False,
)
