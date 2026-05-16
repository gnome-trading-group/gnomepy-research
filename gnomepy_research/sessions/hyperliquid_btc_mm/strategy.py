from __future__ import annotations

from gnomepy import ExecutionReport, Intent, Strategy
from gnomepy.java.schemas import Schema

from gnomepy_research.signals import (
    EWMA,
    MicropriceFairValue,
    PerAsset,
    SpreadVolatility,
)
from gnomepy_research.signals.operations.ewma import EWMAOperation


class HyperliquidBTCUSDE_MM(Strategy):
    """Market maker on BTC/USDE with basis-deviation reservation skew.

    fv = USDE microprice (tracks market exactly, no lag)
    reservation = fv - inventory_skew - basis_skew
    basis_skew = basis_skew_weight * (current_basis - basis_ewma)
      > 0 when USDE expensive vs history → reservation falls → aggressive selling
      < 0 when USDE cheap vs history → reservation rises → aggressive buying
    """

    def __init__(
        self,
        exchange_id: int,
        usdc_security_id: int,
        usde_security_id: int,
        size: int = 100000,
        max_position: float = 0.5,
        skew_per_btc_bps: float = 2.0,
        basis_skew_weight: float = 0.1,
        delta: float = 1.0,
        min_spread_bps: float = 3.0,
        warmup_ticks: int = 200,
        basis_alpha: float = 0.99,
        vol_alpha: float = 0.99,
        processing_time_ns: int = 0,
    ):
        self.exchange_id = exchange_id
        self.usdc_security_id = usdc_security_id
        self.usde_security_id = usde_security_id
        self.size = size
        self.max_position = max_position
        self.skew_per_btc_bps = skew_per_btc_bps
        self.basis_skew_weight = basis_skew_weight
        self.delta = delta
        self.min_spread_bps = min_spread_bps
        self.warmup_ticks = warmup_ticks
        self._processing_time_ns = processing_time_ns

        self._usdc_fv = PerAsset(MicropriceFairValue(), security_id=usdc_security_id)
        self._usde_fv = PerAsset(MicropriceFairValue(), security_id=usde_security_id)
        self._vol = PerAsset(EWMA(SpreadVolatility(), alpha=vol_alpha), security_id=usde_security_id)

        self._basis_ewma = EWMAOperation(alpha=basis_alpha)

        self._usde_tick_count = 0

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_market_data(self, data: Schema) -> list[Intent]:
        timestamp = data.event_timestamp

        self._usdc_fv.update(timestamp, data)
        self._usde_fv.update(timestamp, data)
        self._vol.update(timestamp, data)

        if self._usdc_fv.is_ready() and self._usde_fv.is_ready():
            current_basis = float(self._usde_fv.value() - self._usdc_fv.value())
            self._basis_ewma.update(current_basis)

        if data.security_id != self.usde_security_id or data.exchange_id != self.exchange_id:
            return []

        self._usde_tick_count += 1
        if self._usde_tick_count < self.warmup_ticks:
            return []
        if not (self._usdc_fv.is_ready() and self._usde_fv.is_ready() and
                self._vol.is_ready() and self._basis_ewma.is_ready()):
            return []

        fv = float(self._usde_fv.value())
        if fv <= 0:
            return []

        current_basis = float(self._usde_fv.value() - self._usdc_fv.value())
        basis_dev = current_basis - self._basis_ewma.value()
        basis_skew = self.basis_skew_weight * basis_dev

        vol_bps = self._vol.value()
        position_raw = self.oms.get_effective_quantity(self.exchange_id, self.usde_security_id)
        position_btc = position_raw / 1_000_000

        skew_pos = max(-self.max_position, min(self.max_position, position_btc))
        inventory_skew = self.skew_per_btc_bps * skew_pos * fv / 10_000

        reservation = fv - inventory_skew - basis_skew

        vol_raw = fv * (vol_bps / 10_000)
        min_half = fv * (self.min_spread_bps / 10_000) / 2
        half_spread = max(vol_raw * self.delta / 2, min_half)

        bid_price = reservation - half_spread
        ask_price = reservation + half_spread

        if bid_price <= 0 or ask_price <= 0 or ask_price <= bid_price:
            return []

        bid_size = self.size
        ask_size = self.size
        if position_btc >= self.max_position:
            bid_price = fv * 0.5
        if position_btc <= -self.max_position:
            ask_price = fv * 2.0

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
