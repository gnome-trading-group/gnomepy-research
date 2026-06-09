from __future__ import annotations

from abc import abstractmethod

from gnomepy_research.signals.base import Signal


class BookSignal(Signal[float]):
    """Order book microstructure signal.

    Computes structural features from the order book shape: depth imbalances,
    queue composition, spread metrics. Output is a float.
    """

    @abstractmethod
    def value(self) -> float: ...
