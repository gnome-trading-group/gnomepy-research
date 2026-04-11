"""Momentum-taking strategy with stop-loss and take-profit.

Compares the live microprice against an EWMA-smoothed reference. When
momentum exceeds a threshold, fires an aggressive IOC market take in the
direction of the move, subject to a position cap. Exits via stop-loss or
take-profit when the position moves against or in favor by a configurable
number of basis points. A cooldown period prevents immediate re-entry
after an exit.
"""
from __future__ import annotations

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema

from gnomepy_research.signals import EWMA, MicropriceFairValue


class MomentumTaker(Strategy):
    """Momentum-taking strategy with EWMA smoothing and dynamic exits."""
    
    # Basis point conversion constant (1 bp = 0.0001)
    BPS_CONVERSION = 10_000
    
    def __init__(
        self,
        exchange_id: int,
        security_id: int,
        size: int = 1,
        max_position: int = 5,
        threshold_bps: float = 2.0,
        ewma_alpha: float = 0.95,
        warmup_ticks: int = 100,
        stop_loss_bps: float = 10.0,
        take_profit_bps: float = 15.0,
        cooldown_ticks: int = 50,
        processing_time_ns: int = 0,
    ):
        self.exchange_id = exchange_id
        self.security_id = security_id
        self.size = size
        self.max_position = max_position
        self.threshold_bps = threshold_bps
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps
        self.cooldown_ticks = cooldown_ticks
        self.warmup_ticks = warmup_ticks
        self._processing_time_ns = processing_time_ns

        self._micro = MicropriceFairValue()
        self._smoothed = EWMA(MicropriceFairValue(), alpha=ewma_alpha)
        self._tick_count = 0
        self._ticks_since_exit = self.cooldown_ticks  # start ready to trade

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_market_data(self, timestamp: int, data: Schema) -> list[Intent]:
        if data.security_id != self.security_id or data.exchange_id != self.exchange_id:
            return []

        self._micro.update(timestamp, data)
        self._smoothed.update(timestamp, data)
        self._tick_count += 1
        self._ticks_since_exit += 1

        if self._tick_count < self.warmup_ticks:
            return []
        if not (self._micro.is_ready() and self._smoothed.is_ready()):
            return []

        micro = self._micro.value()
        position = self.oms.get_effective_quantity(self.exchange_id, self.security_id)

        # --- Exit logic: stop-loss / take-profit ---
        if position != 0:
            pos_info = self.oms.get_position(self.exchange_id, self.security_id)
            if pos_info is not None:
                entry_price = pos_info.avg_entry_price
                if entry_price > 0:
                    if position > 0:
                        pnl_bps = (micro - entry_price) / entry_price * 10_000
                    else:
                        pnl_bps = (entry_price - micro) / entry_price * 10_000

                    if pnl_bps <= -self.stop_loss_bps:
                        self._ticks_since_exit = 0
                        return [self._close(position)]
                    if pnl_bps >= self.take_profit_bps:
                        self._ticks_since_exit = 0
                        return [self._close(position)]

        # --- Cooldown after exit ---
        if self._ticks_since_exit < self.cooldown_ticks:
            return []

        # --- Entry logic: momentum signal ---
        smoothed = self._smoothed.value()
        if smoothed <= 0:
            return []

        # Calculate momentum as deviation from EWMA in basis points
        delta_bps = (micro - smoothed) / smoothed * 10_000

        # Buy on positive momentum (microprice > EWMA), sell on negative momentum
        if delta_bps > self.threshold_bps and position < self.max_position:
            return [self._take(Side.ASK)]
        if delta_bps < -self.threshold_bps and position > -self.max_position:
            return [self._take(Side.BID)]
        return []

    def on_execution_report(self, timestamp: int, report: ExecutionReport) -> None:
        return None

    def _take(self, side: Side) -> Intent:
        return Intent(
            exchange_id=self.exchange_id,
            security_id=self.security_id,
            take_side=side,
            take_size=self.size,
            take_order_type=OrderType.MARKET,
        )

    def _close(self, position: int) -> Intent:
        """Close the entire position with a market IOC."""
        if position > 0:
            side = Side.ASK
        else:
            side = Side.BID
        return Intent(
            exchange_id=self.exchange_id,
            security_id=self.security_id,
            take_side=side,
            take_size=abs(position),
            take_order_type=OrderType.MARKET,
        )
