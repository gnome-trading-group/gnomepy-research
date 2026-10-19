"""Each listing's fee rates for code that prices against the ``rate × p(1−p)`` curve Kalshi and Polymarket charge."""
from __future__ import annotations

from typing import Callable

from gnomepy_research.listing import Listing

# A listing's (taker, maker) curve rates; a negative maker rate is a rebate.
FeeRates = Callable[[Listing], tuple[float, float]]


def registry_fee_rates(strategy) -> FeeRates:
    """The rates of each listing's fee schedule in the registry, as in force at the strategy's time (the backtest's
    simulated time in a backtest), which is what the simulator charges its fills. Looked up on each call, so a
    schedule change mid-backtest takes effect from then on; the strategy's positions only exist once it starts."""

    def rates(listing: Listing) -> tuple[float, float]:
        schedule = strategy.positions.fee_schedule(listing[0], listing[1])
        if schedule is None:
            raise ValueError(f"The registry has no fee schedule for listing {listing} at this time")
        return schedule.price_curve_rates()

    return rates
