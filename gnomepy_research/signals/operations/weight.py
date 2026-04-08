from __future__ import annotations

from typing import Generic, TypeVar

from gnomepy.java.schemas import JavaMBP10Schema
from gnomepy_research.signals.base import Signal
from gnomepy_research.signals.fair_value.base import FairValueSignal
from gnomepy_research.signals.flow.base import FlowSignal
from gnomepy_research.signals.volatility.base import VolatilitySignal

T = TypeVar("T", int, float)


class _WeightedSignal(Signal[T], Generic[T]):
    """Weighted combination of multiple signals of the same type.

    value = Σ(w_i * signal_i.value())  (weights auto-normalized to sum to 1)
    """

    def __init__(self, signals: list[Signal[T]], weights: list[float]):
        if len(signals) != len(weights):
            raise ValueError(f"Got {len(signals)} signals but {len(weights)} weights")
        self.signals = signals
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        for s in self.signals:
            s.update(timestamp, data)

    def is_ready(self) -> bool:
        return all(s.is_ready() for s in self.signals)

    def reset(self) -> None:
        for s in self.signals:
            s.reset()


class WeightedFairValue(_WeightedSignal[int], FairValueSignal):
    def value(self) -> int:
        return int(sum(w * s.value() for w, s in zip(self.weights, self.signals)))


class WeightedVolatility(_WeightedSignal[float], VolatilitySignal):
    def value(self) -> float:
        return sum(w * s.value() for w, s in zip(self.weights, self.signals))


class WeightedFlow(_WeightedSignal[float], FlowSignal):
    def value(self) -> float:
        return sum(w * s.value() for w, s in zip(self.weights, self.signals))


def Weight(signals: list, weights: list[float]):
    """Weighted combination of signals. All signals must be the same type.

    Weights are auto-normalized to sum to 1.
    """
    if all(isinstance(s, FairValueSignal) for s in signals):
        return WeightedFairValue(signals, weights)
    if all(isinstance(s, VolatilitySignal) for s in signals):
        return WeightedVolatility(signals, weights)
    if all(isinstance(s, FlowSignal) for s in signals):
        return WeightedFlow(signals, weights)
    raise TypeError(
        "All signals must be the same type "
        "(FairValueSignal, VolatilitySignal, or FlowSignal)"
    )
