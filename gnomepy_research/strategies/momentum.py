"""Basic momentum-taking strategy.

Compares the live microprice against an EWMA-smoothed reference. When
momentum exceeds a threshold, fires an aggressive IOC market take in the
direction of the move, subject to a position cap.
"""
from __future__ import annotations

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema

from gnomepy_research.signals import EWMA, MicropriceFairValue


class MomentumTaker(Strategy):
    def __init__(
        self,
        exchange_id: int,
        security_id: int,
        size: int = 1,
        max_position: int = 5,
        threshold_bps: float = 2.0,
        ewma_alpha: float = 0.95,
        warmup_ticks: int = 100,
    ):
        self.exchange_id = exchange_id
        self.security_id = security_id
        self.size = size
        self.max_position = max_position
        self.threshold_bps = threshold_bps
        self.warmup_ticks = warmup_ticks

        self._micro = MicropriceFairValue()
        self._smoothed = EWMA(MicropriceFairValue(), alpha=ewma_alpha)
        self._tick_count = 0

    def on_market_data(self, timestamp: int, data: Schema) -> list[Intent]:
        if data.security_id != self.security_id or data.exchange_id != self.exchange_id:
            return []

        self._micro.update(timestamp, data)
        self._smoothed.update(timestamp, data)
        self._tick_count += 1

        if self._tick_count < self.warmup_ticks:
            return []
        if not (self._micro.is_ready() and self._smoothed.is_ready()):
            return []

        micro = self._micro.value()
        smoothed = self._smoothed.value()
        if smoothed <= 0:
            return []

        delta_bps = (micro - smoothed) / smoothed * 10_000
        position = self.oms.get_effective_quantity(self.exchange_id, self.security_id)

        if delta_bps > self.threshold_bps and position < self.max_position:
            return [self._take(Side.ASK)]
        if delta_bps < -self.threshold_bps and position > -self.max_position:
            return [self._take(Side.BID)]
        return []

    def on_execution_report(self, timestamp: int, report: ExecutionReport) -> None:
        # OMS already tracks position; nothing to do here for now.
        return None

    def _take(self, side: Side) -> Intent:
        return Intent(
            exchange_id=self.exchange_id,
            security_id=self.security_id,
            take_side=side,
            take_size=self.size,
            take_order_type=OrderType.MARKET,
        )
