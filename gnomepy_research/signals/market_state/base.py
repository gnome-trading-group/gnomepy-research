from __future__ import annotations

from abc import abstractmethod

from gnomepy_research.signals.base import Signal


class MarketStateSignal(Signal[float]):
    """Market state / regime indicator signal.

    Encodes the current market condition as a continuous scalar — liquidity,
    activity, and spread regime. Output is a float.
    """

    @abstractmethod
    def value(self) -> float: ...
