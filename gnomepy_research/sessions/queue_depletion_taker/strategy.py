from __future__ import annotations

import datetime

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema

from gnomepy_research.signals import (
    EWMA,
    Aggression,
    LevelLiquidityDelta,
    LevelStaleness,
    MicropriceFairValue,
    TradeImbalance,
)


class QueueDepletionTaker(Strategy):
    def __init__(
        self,
        exchange_id: int,
        security_id: int,
        size: int = 1,
        max_position: int = 10,
        entry_threshold: float = 0.3,
        exit_threshold: float = 0.1,
        stop_loss_bps: float = 15.0,
        take_profit_bps: float = 20.0,
        cooldown_ticks: int = 50,
        warmup_ticks: int = 50,
        flow_horizon_ns: int = 500_000_000,
        liq_horizon_ns: int = 1_000_000_000,
        w_imbalance: float = 0.5,
        w_staleness: float = 0.25,
        w_liq_delta: float = 0.25,
        toxicity_threshold_bps: float = 8.0,
        session_start_hour: int = 9,
        processing_time_ns: int = 0,
    ):
        self.exchange_id = exchange_id
        self.security_id = security_id
        self.size = size
        self.max_position = max_position
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps
        self.cooldown_ticks = cooldown_ticks
        self.warmup_ticks = warmup_ticks
        self.w_imbalance = w_imbalance
        self.w_staleness = w_staleness
        self.w_liq_delta = w_liq_delta
        self.toxicity_threshold_bps = toxicity_threshold_bps
        self.session_start_hour = session_start_hour
        self._processing_time_ns = processing_time_ns

        self._fv = MicropriceFairValue()
        self._imbalance = TradeImbalance(horizon_ns=flow_horizon_ns, warmup_trades=20)
        self._bid_staleness = LevelStaleness(Side.BID, level=0, warmup_ticks=10)
        self._ask_staleness = LevelStaleness(Side.ASK, level=0, warmup_ticks=10)
        self._bid_liq = LevelLiquidityDelta(Side.BID, level=0, horizon_ns=liq_horizon_ns, warmup_events=10)
        self._ask_liq = LevelLiquidityDelta(Side.ASK, level=0, horizon_ns=liq_horizon_ns, warmup_events=10)
        self._aggression = EWMA(Aggression(warmup_trades=20), alpha=0.9)
        self._score_smooth = 0.0
        self._score_alpha = 0.7  # EWMA alpha for depletion score (fast response)

        self._tick_count = 0
        self._ticks_since_exit = cooldown_ticks
        self._pending_order = False  # True while any order is in-flight

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_market_data(self, data: Schema) -> list[Intent]:
        if data.security_id != self.security_id or data.exchange_id != self.exchange_id:
            return []

        ts = data.event_timestamp
        self._fv.update(ts, data)
        self._imbalance.update(ts, data)
        self._bid_staleness.update(ts, data)
        self._ask_staleness.update(ts, data)
        self._bid_liq.update(ts, data)
        self._ask_liq.update(ts, data)
        self._aggression.update(ts, data)
        self._tick_count += 1
        self._ticks_since_exit += 1

        if self._tick_count < self.warmup_ticks:
            return []
        if not self._fv.is_ready():
            return []

        fv = self._fv.value()
        if fv <= 0:
            return []

        position = self.oms.get_effective_quantity(self.exchange_id, self.security_id)

        # --- Depletion score in [-1, +1] ------------------------------------
        score_imbalance = self._imbalance.value() if self._imbalance.is_ready() else 0.0

        if self._bid_staleness.is_ready() and self._ask_staleness.is_ready():
            bid_st = self._bid_staleness.value()
            ask_st = self._ask_staleness.value()
            score_staleness = (ask_st - bid_st) / (bid_st + ask_st + 1.0)
        else:
            score_staleness = 0.0

        if self._bid_liq.is_ready() and self._ask_liq.is_ready():
            bid_ld = self._bid_liq.value()
            ask_ld = self._ask_liq.value()
            score_liq = (bid_ld - ask_ld) / (abs(bid_ld) + abs(ask_ld) + 1.0)
        else:
            score_liq = 0.0

        raw_score = max(-1.0, min(1.0,
            self.w_imbalance * score_imbalance
            + self.w_staleness * score_staleness
            + self.w_liq_delta * score_liq
        ))
        self._score_smooth = self._score_alpha * raw_score + (1.0 - self._score_alpha) * self._score_smooth
        score = self._score_smooth

        if self._pending_order:
            return []

        # --- Exit: SL/TP only -----------------------------------------------
        if position != 0:
            pos_info = self.oms.get_position(self.exchange_id, self.security_id)
            entry_price = pos_info.avg_entry_price
            if entry_price > 0:
                if position > 0:
                    pnl_bps = (fv - entry_price) / entry_price * 10_000
                else:
                    pnl_bps = (entry_price - fv) / entry_price * 10_000
                if pnl_bps <= -self.stop_loss_bps or pnl_bps >= self.take_profit_bps:
                    self._ticks_since_exit = 0
                    self._pending_order = True
                    return [self._close(position)]
            return []

        # --- Cooldown --------------------------------------------------------
        if self._ticks_since_exit < self.cooldown_ticks:
            return []

        # --- Toxicity gate ---------------------------------------------------
        aggression_bps = abs(self._aggression.value()) if self._aggression.is_ready() else 0.0
        effective_threshold = self.entry_threshold * (2.0 if aggression_bps > self.toxicity_threshold_bps else 1.0)

        # --- Entry (contrarian: strong buy pressure → sell, sell pressure → buy) ---
        if score > effective_threshold and position > -self.max_position:
            self._pending_order = True
            return [self._take(Side.ASK)]
        if score < -effective_threshold and position < self.max_position:
            self._pending_order = True
            return [self._take(Side.BID)]

        return []

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        self._pending_order = False
        return []

    def _take(self, side: Side) -> Intent:
        return Intent(
            exchange_id=self.exchange_id,
            security_id=self.security_id,
            take_side=side,
            take_size=self.size,
            take_order_type=OrderType.MARKET,
        )

    def _close(self, position: int) -> Intent:
        side = Side.ASK if position > 0 else Side.BID
        return Intent(
            exchange_id=self.exchange_id,
            security_id=self.security_id,
            take_side=side,
            take_size=abs(position),
            take_order_type=OrderType.MARKET,
        )
