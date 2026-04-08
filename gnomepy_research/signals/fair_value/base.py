from __future__ import annotations

from abc import abstractmethod

from gnomepy_research.signals.base import Signal


class FairValueSignal(Signal[int]):
    """Fair value estimate in raw price units (same scale as bid/ask).

    The market maker quotes around this value instead of raw mid.
    """

    @abstractmethod
    def value(self) -> int:
        """Current fair value in raw price units."""
        ...
