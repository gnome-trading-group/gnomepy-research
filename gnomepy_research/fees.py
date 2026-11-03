"""Each listing's fee rates for code that prices against the ``rate × p(1−p)`` curve Kalshi and Polymarket charge."""
from __future__ import annotations

from typing import Callable

# A listing's (taker, maker) curve rates, by listing id; a negative maker rate is a rebate.
FeeRates = Callable[[int], tuple[float, float]]


def registry_fee_rates(strategy) -> FeeRates:
    """The rates of each listing's fee schedule in the registry, as in force at the strategy's time (the backtest's
    simulated time in a backtest), which is what the simulator charges its fills. Looked up on each call, so a
    schedule change mid-backtest takes effect from then on; the strategy's positions only exist once it starts."""

    def rates(listing_id: int) -> tuple[float, float]:
        listing = strategy.streams.listing(listing_id)
        schedule = strategy.positions.fee_schedule(
            int(listing.exchange().exchangeId()), int(listing.security().securityId()))
        if schedule is None:
            raise ValueError(f"The registry has no fee schedule for listing {listing_id} at this time")
        return schedule.price_curve_rates()

    return rates
