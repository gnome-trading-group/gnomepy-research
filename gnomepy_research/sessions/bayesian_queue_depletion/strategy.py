from __future__ import annotations

from gnomepy import ExecutionReport, Intent, Strategy
from gnomepy.java.enums import Side
from gnomepy.java.schemas import Schema

from gnomepy_research.signals import (
    EWMA,
    Aggression,
    LevelLiquidityDelta,
    LevelStaleness,
    MicropriceFairValue,
    SpreadVolatility,
    TradeImbalance,
)


class BayesianQueueDepletionMM(Strategy):
    """Market maker skewed by a queue depletion score.

    Combines trade imbalance, level staleness asymmetry, and liquidity-delta
    asymmetry into a single depletion score in [-1, +1]:
      positive → ask side more vulnerable → price likely moves up
      negative → bid side more vulnerable → price likely moves down

    Skews reservation price and quote sizes in the direction of the predicted
    move. EWMA-smoothed aggression gates out toxic/informed flow.
    """

    def __init__(
        self,
        exchange_id: int,
        security_id: int,
        size: int = 1,
        max_position: int = 20,
        gamma: float = 0.01,
        delta: float = 1.0,
        min_spread_bps: float = 1.5,
        warmup_ticks: int = 200,
        flow_horizon_ns: int = 500_000_000,
        liq_horizon_ns: int = 1_000_000_000,
        w_imbalance: float = 0.5,
        w_staleness: float = 0.25,
        w_liq_delta: float = 0.25,
        depletion_skew_bps: float = 3.0,
        size_skew_factor: float = 0.5,
        toxicity_threshold_bps: float = 8.0,
        toxicity_spread_mult: float = 3.0,
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
        self.w_imbalance = w_imbalance
        self.w_staleness = w_staleness
        self.w_liq_delta = w_liq_delta
        self.depletion_skew_bps = depletion_skew_bps
        self.size_skew_factor = size_skew_factor
        self.toxicity_threshold_bps = toxicity_threshold_bps
        self.toxicity_spread_mult = toxicity_spread_mult
        self._processing_time_ns = processing_time_ns

        self._fv = MicropriceFairValue()
        self._vol = EWMA(SpreadVolatility(), alpha=0.99)
        self._imbalance = TradeImbalance(horizon_ns=flow_horizon_ns, warmup_trades=20)
        self._bid_staleness = LevelStaleness(Side.BID, level=0, warmup_ticks=10)
        self._ask_staleness = LevelStaleness(Side.ASK, level=0, warmup_ticks=10)
        self._bid_liq = LevelLiquidityDelta(Side.BID, level=0, horizon_ns=liq_horizon_ns, warmup_events=10)
        self._ask_liq = LevelLiquidityDelta(Side.ASK, level=0, horizon_ns=liq_horizon_ns, warmup_events=10)
        self._aggression = EWMA(Aggression(warmup_trades=20), alpha=0.9)

        self._tick_count = 0

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_market_data(self, data: Schema) -> list[Intent]:
        if data.security_id != self.security_id or data.exchange_id != self.exchange_id:
            return []

        timestamp = data.event_timestamp
        self._fv.update(timestamp, data)
        self._vol.update(timestamp, data)
        self._imbalance.update(timestamp, data)
        self._bid_staleness.update(timestamp, data)
        self._ask_staleness.update(timestamp, data)
        self._bid_liq.update(timestamp, data)
        self._ask_liq.update(timestamp, data)
        self._aggression.update(timestamp, data)
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

        # --- Depletion score in [-1, +1] ----------------------------------
        # Component 1: trade imbalance (+1 = all buys → ask vulnerable)
        score_imbalance = self._imbalance.value() if self._imbalance.is_ready() else 0.0

        # Component 2: staleness asymmetry — stale ask = vulnerable ask
        if self._bid_staleness.is_ready() and self._ask_staleness.is_ready():
            bid_st = self._bid_staleness.value()
            ask_st = self._ask_staleness.value()
            denom_st = bid_st + ask_st + 1.0
            score_staleness = (ask_st - bid_st) / denom_st
        else:
            score_staleness = 0.0

        # Component 3: liquidity delta asymmetry
        # bid_liq > 0 = bid side adding liquidity; ask losing = ask vulnerable
        if self._bid_liq.is_ready() and self._ask_liq.is_ready():
            bid_ld = self._bid_liq.value()
            ask_ld = self._ask_liq.value()
            denom_ld = abs(bid_ld) + abs(ask_ld) + 1.0
            score_liq = (bid_ld - ask_ld) / denom_ld
        else:
            score_liq = 0.0

        depletion_score = (
            self.w_imbalance * score_imbalance
            + self.w_staleness * score_staleness
            + self.w_liq_delta * score_liq
        )
        depletion_score = max(-1.0, min(1.0, depletion_score))

        # --- Toxicity gate ---------------------------------------------------
        aggression_bps = abs(self._aggression.value()) if self._aggression.is_ready() else 0.0
        toxic = aggression_bps > self.toxicity_threshold_bps
        spread_mult = self.toxicity_spread_mult if toxic else 1.0

        # --- Reservation price (inventory + depletion skew) ------------------
        vol_raw = fv * (vol_bps / 10_000)
        min_half = fv * (self.min_spread_bps / 10_000) / 2
        half_spread_base = max(vol_raw * self.delta / 2, min_half)

        depletion_skew_raw = fv * (self.depletion_skew_bps / 10_000) * depletion_score
        reservation = fv + depletion_skew_raw

        # --- Asymmetric spread: widen the side at risk of adverse selection ---
        # When depletion_score < 0 (market dropping): widen bid, tighten ask
        # When depletion_score > 0 (market rising): widen ask, tighten bid
        asym = abs(depletion_score) * self.gamma
        if depletion_score < 0:
            bid_half = half_spread_base * (1.0 + asym) * spread_mult
            ask_half = half_spread_base * max(0.5, 1.0 - asym * 0.5) * spread_mult
        else:
            bid_half = half_spread_base * max(0.5, 1.0 - asym * 0.5) * spread_mult
            ask_half = half_spread_base * (1.0 + asym) * spread_mult

        bid_price = int(reservation - bid_half)
        ask_price = int(reservation + ask_half)

        # --- Size skew: lean on the expected-fill side -----------------------
        raw_bid = self.size * (1.0 + self.size_skew_factor * depletion_score)
        raw_ask = self.size * (1.0 - self.size_skew_factor * depletion_score)
        bid_size = max(0, int(round(raw_bid)))
        ask_size = max(0, int(round(raw_ask)))

        # --- Position limits -------------------------------------------------
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

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []
