from __future__ import annotations

from abc import abstractmethod

from gnomepy_research.signals.base import Signal


class VolatilitySignal(Signal[float]):
    """Volatility estimate, typically in basis points.

    A 50 bps volatility means "expected price variation of ~50 bps
    over the model's horizon." Used to size spreads, set risk limits, etc.
    """

    @abstractmethod
    def value(self) -> float:
        """Current volatility estimate in basis points."""
        ...
