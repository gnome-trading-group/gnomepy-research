from __future__ import annotations

from gnomepy import ExecutionReport, Intent, Strategy
from gnomepy.java.schemas import Schema

from gnomepy_research.signals import (
    EWMA,
    MicropriceFairValue,
    PerAsset,
    SpreadVolatility,
)


class HyperliquidBTCUSDE_MM(Strategy):
    def __init__(
        self,
        exchange_id: int,
        usdc_security_id: int,
        usde_security_id: int,
        size: int = 100000,
        max_position: int = 5,
        gamma: float = 0.1,
        delta: float = 1.0,
        min_spread_bps: float = 3.0,
        warmup_ticks: int = 100,
        vol_alpha: float = 0.99,
        processing_time_ns: int = 0,
    ):
        self.exchange_id = exchange_id
        self.usdc_security_id = usdc_security_id
        self.usde_security_id = usde_security_id
        self.size = size
        self.max_position = max_position
        self.gamma = gamma
        self.delta = delta
        self.min_spread_bps = min_spread_bps
        self.warmup_ticks = warmup_ticks
        self._processing_time_ns = processing_time_ns

        self._ref_fv = PerAsset(MicropriceFairValue(), security_id=usdc_security_id)
        self._vol = PerAsset(EWMA(SpreadVolatility(), alpha=vol_alpha), security_id=usde_security_id)

        self._usde_tick_count = 0

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_market_data(self, data: Schema) -> list[Intent]:
        timestamp = data.event_timestamp

        self._ref_fv.update(timestamp, data)
        self._vol.update(timestamp, data)

        if data.security_id != self.usde_security_id or data.exchange_id != self.exchange_id:
            return []

        self._usde_tick_count += 1
        if self._usde_tick_count < self.warmup_ticks:
            return []
        if not (self._ref_fv.is_ready() and self._vol.is_ready()):
            return []

        fv = self._ref_fv.value()
        if fv <= 0:
            return []

        vol_bps = self._vol.value()
        position = self.oms.get_effective_quantity(self.exchange_id, self.usde_security_id)

        vol_raw = fv * (vol_bps / 10_000)

        if vol_raw > 0:
            skew = self.gamma * position * vol_raw * vol_raw / fv
        else:
            skew = 0.0
        reservation = fv - skew

        min_half = fv * (self.min_spread_bps / 10_000) / 2
        half_spread = max(vol_raw * self.delta / 2, min_half)

        bid_price = int(reservation - half_spread)
        ask_price = int(reservation + half_spread)

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
            security_id=self.usde_security_id,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
        )]

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []
