from __future__ import annotations

from gnomepy_research.signals.operations.std import Std
from gnomepy_research.signals.volatility.realized import RealizedVolatility


def VolOfVol(inner_horizon: int = 50, outer_horizon: int = 100):
    """Rolling standard deviation of RealizedVolatility.

    Composition: Std(RealizedVolatility(inner_horizon), outer_horizon).
    High vol-of-vol means the volatility regime is unstable.
    """
    return Std(RealizedVolatility(horizon=inner_horizon), horizon=outer_horizon)
