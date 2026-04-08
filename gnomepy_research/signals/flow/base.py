from __future__ import annotations

from abc import abstractmethod

from gnomepy.java.enums import Action
from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.base import Signal


def is_trade_event(data: Mbp10Schema) -> bool:
    """Check if an MBP10 update represents a trade event."""
    return data.action == Action.TRADE.value and data.price > 0 and data.size > 0


class FlowSignal(Signal[float]):
    """Order/trade flow scalar.

    Outputs a scalar value derived from trade or order flow data.
    Typically fed trade-schema updates (TRADES, MBO).
    """

    @abstractmethod
    def value(self) -> float:
        """Current flow signal value."""
        ...
