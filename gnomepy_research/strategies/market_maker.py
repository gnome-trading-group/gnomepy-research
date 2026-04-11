"""Avellaneda-Stoikov inspired market maker with pluggable models.

Quotes both sides around a reservation price derived from a fair value
signal, with spread sized by a volatility model and skewed by a quadratic
inventory penalty.

Usage::

    from gnomepy_research.signals import MicropriceFairValue, SpreadVolatility, EWMA
    from gnomepy_research.strategies.market_maker import MarketMaker

    mm = MarketMaker(
        exchange_id=1, security_id=1,
        fair_value=MicropriceFairValue(),
        volatility=EWMA(SpreadVolatility(), alpha=0.99),
        gamma=0.01, delta=1.0,
    )
"""
from __future__ import annotations

from gnomepy import ExecutionReport, Intent, Strategy
from gnomepy.java.schemas import Schema

from gnomepy_research.signals import (
    FairValueSignal,
    MicropriceFairValue,
    SpreadVolatility,
    VolatilitySignal,
)


class MarketMaker(Strategy):
    def __init__(
        self,
        exchange_id: int,
        security_id: int,
        fair_value: FairValueSignal | None = None,
        volatility: VolatilitySignal | None = None,
        size: int = 1,
        max_position: int = 10,
        gamma: float = 0.01,
        delta: float = 1.0,
        min_spread_bps: float = 1.0,
        warmup_ticks: int = 100,
        processing_time_ns: int = 0,
    ):
        self.exchange_id = exchange_id
        self.security_id = security_id
        self.size = size
        self.max_position = max_position
        self.gamma = gamma
        self.delta = delta
        self.min_spread_bps = min_spread_bps
        self.warmup_ticks = warmup_ticks
        self._processing_time_ns = processing_time_ns

        self._fv = fair_value or MicropriceFairValue()
        self._vol = volatility or SpreadVolatility()
        self._tick_count = 0

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_market_data(self, timestamp: int, data: Schema) -> list[Intent]:
        if data.security_id != self.security_id or data.exchange_id != self.exchange_id:
            return []

        self._fv.update(timestamp, data)
        self._vol.update(timestamp, data)
        self._tick_count += 1

        if self._tick_count < self.warmup_ticks:
            return []
        if not (self._fv.is_ready() and self._vol.is_ready()):
            return []

        fv = self._fv.value()
        if fv <= 0:
            return []

        vol_bps = self._vol.value()
        position = self.oms.get_effective_quantity(self.exchange_id, self.security_id)

        # Convert volatility from bps to raw price units.
        vol_raw = fv * (vol_bps / 10_000)

        # Quadratic inventory skew: shift reservation price away from
        # the side we're heavy on.
        if vol_raw > 0:
            skew = self.gamma * position * vol_raw * vol_raw / fv
        else:
            skew = 0.0
        reservation = fv - skew

        # Spread: vol-scaled with a minimum floor.
        min_half = fv * (self.min_spread_bps / 10_000) / 2
        half_spread = max(vol_raw * self.delta / 2, min_half)

        bid_price = int(reservation - half_spread)
        ask_price = int(reservation + half_spread)

        # Position limits — cancel the side that would increase exposure.
        bid_size = self.size
        ask_size = self.size
        if position >= self.max_position:
            bid_size = 0
            bid_price = 0
        if position <= -self.max_position:
            ask_size = 0
            ask_price = 0

        return [Intent(
            exchange_id=self.exchange_id,
            security_id=self.security_id,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
        )]

    def on_execution_report(self, timestamp: int, report: ExecutionReport) -> None:
        return None
