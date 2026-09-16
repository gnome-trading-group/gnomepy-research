from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone

from gnomepy import ExecutionReport, Intent, Strategy
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

from gnomepy_research.signals.fair_value.microprice import MicropriceFairValue
from gnomepy_research.signals.operations.kalman import KalmanOperation

PRICE_SCALE = 1_000_000_000


class KalshiSportsMM(Strategy):
    def __init__(
        self,
        ref_listing_id: int,
        team_a_listing_id: int,
        team_b_listing_id: int,
        size: int = 3_000_000,
        max_exposure: int = 30,
        base_spread: float = 0.02,
        inventory_skew: float = 0.001,
        divergence_gate: float = 0.025,
        overround_gate: float = 0.05,
        vol_gate: float = 0.005,
        vol_horizon: int = 20,
        kalman_Q: float = 1e-4,
        kalman_R: float = 1e-2,
        warmup_ticks: int = 50,
        max_ref_staleness_ns: int = 5_000_000_000,
        min_quote_interval_ns: int = 250_000_000,
        tau_pull_threshold: float = 0.05,
        tau_widen_threshold: float = 0.15,
        resolution_time_override_ns: int = 0,
        processing_time_ns: int = 5_000_000,
    ):
        self.size = size
        self.max_exposure = max_exposure
        self.base_spread = base_spread
        self._inventory_skew = inventory_skew
        self._divergence_gate = divergence_gate
        self._overround_gate = overround_gate
        self._vol_gate = vol_gate
        self.warmup_ticks = warmup_ticks
        self._max_ref_staleness_ns = max_ref_staleness_ns
        self._min_quote_interval_ns = min_quote_interval_ns
        self._tau_pull_threshold = tau_pull_threshold
        self._tau_widen_threshold = tau_widen_threshold
        self._processing_time_ns = processing_time_ns

        registry = RegistryClient()

        ref_listings = registry.get_listing(listing_id=ref_listing_id)
        if not ref_listings:
            raise ValueError(f"No listing for ref_listing_id={ref_listing_id}")
        self._ref_eid: int = ref_listings[0].exchange_id
        self._ref_sid: int = ref_listings[0].security_id

        a_listings = registry.get_listing(listing_id=team_a_listing_id)
        if not a_listings:
            raise ValueError(f"No listing for team_a_listing_id={team_a_listing_id}")
        self._kalshi_eid: int = a_listings[0].exchange_id
        self._sid_a: int = a_listings[0].security_id

        b_listings = registry.get_listing(listing_id=team_b_listing_id)
        if not b_listings:
            raise ValueError(f"No listing for team_b_listing_id={team_b_listing_id}")
        if b_listings[0].exchange_id != self._kalshi_eid:
            raise ValueError("team_a and team_b must be on the same exchange")
        self._sid_b: int = b_listings[0].security_id

        specs = registry.get_listing_spec(listing_id=team_a_listing_id)
        self._tick_size: int = int(specs[0].tick_size) if specs else 1
        self._lot_size: int = int(specs[0].lot_size) if specs else 1

        if resolution_time_override_ns:
            self.resolution_time_ns: int = resolution_time_override_ns
        else:
            contracts = registry.get_event_contracts(security_id=self._sid_a)
            if not contracts:
                raise ValueError(f"No event contracts for security_id={self._sid_a}")
            events = registry.get_event(event_id=contracts[0].event_id)
            if not events or events[0].expiry is None:
                raise ValueError(f"Event {contracts[0].event_id} has no expiry set")
            expiry_dt = datetime.fromisoformat(events[0].expiry.replace('Z', '+00:00'))
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
            self.resolution_time_ns = int(expiry_dt.timestamp() * 1e9)

        self._ref_fv = MicropriceFairValue()
        self._kalman = KalmanOperation(Q=kalman_Q, R=kalman_R)

        self._ref_fair_value: float = 0.5
        self._kalshi_mid_a: float = 0.5
        self._kalshi_mid_b: float = 0.5
        self._ref_last_ts: int = 0
        self._prev_ref_value: float = 0.0
        self._ref_returns: deque = deque(maxlen=vol_horizon)

        self._start_time_ns: int | None = None
        self._total_time_ns: int | None = None
        self._tick_count: int = 0
        self._last_quote_ts: int = 0
        self._quote_seq: int = 0
        self._metrics_buf = None

        self._last_a_bid: int = -1
        self._last_a_ask: int = -1
        self._last_a_bid_sz: int = -1
        self._last_a_ask_sz: int = -1
        self._last_b_bid: int = -1
        self._last_b_ask: int = -1
        self._last_b_bid_sz: int = -1
        self._last_b_ask_sz: int = -1
        self._quotes_active: bool = False

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("ksm_signals")
        self._m_ts = buf.addLongColumn("timestamp")
        self._m_p = buf.addDoubleColumn("p")
        self._m_mid_a = buf.addDoubleColumn("mid_a")
        self._m_mid_b = buf.addDoubleColumn("mid_b")
        self._m_overround = buf.addDoubleColumn("overround")
        self._m_vol = buf.addDoubleColumn("vol")
        self._m_D = buf.addDoubleColumn("D")
        self._m_qa = buf.addDoubleColumn("q_a")
        self._m_qb = buf.addDoubleColumn("q_b")
        self._m_ab = buf.addDoubleColumn("a_bid")
        self._m_aa = buf.addDoubleColumn("a_ask")
        self._m_bb = buf.addDoubleColumn("b_bid")
        self._m_ba = buf.addDoubleColumn("b_ask")
        buf.freeze()
        self._metrics_buf = buf

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []

    def _poly_vol(self) -> float:
        n = len(self._ref_returns)
        if n < 2:
            return 0.0
        mean = sum(self._ref_returns) / n
        variance = sum((r - mean) ** 2 for r in self._ref_returns) / n
        return math.sqrt(max(variance, 0.0))

    def on_market_data(self, data: Schema) -> list[Intent]:
        ts = data.event_timestamp

        if data.exchange_id == self._ref_eid and data.security_id == self._ref_sid:
            self._ref_fv.update(ts, data)
            if self._ref_fv.is_ready():
                raw = self._ref_fv.value() / PRICE_SCALE
                self._kalman.update(raw)
                if self._kalman.is_ready():
                    new_fv = float(self._kalman.value())
                    if self._prev_ref_value > 0.0 and new_fv > 0.0:
                        self._ref_returns.append(math.log(new_fv / self._prev_ref_value))
                    self._prev_ref_value = new_fv
                    self._ref_fair_value = new_fv
                self._ref_last_ts = ts
            return []

        if data.exchange_id != self._kalshi_eid:
            return []
        if data.security_id == self._sid_a:
            if data.bid_price(0) > 0 and data.ask_price(0) > 0:
                self._kalshi_mid_a = (data.bid_price(0) + data.ask_price(0)) / (2.0 * PRICE_SCALE)
        elif data.security_id == self._sid_b:
            if data.bid_price(0) > 0 and data.ask_price(0) > 0:
                self._kalshi_mid_b = (data.bid_price(0) + data.ask_price(0)) / (2.0 * PRICE_SCALE)
        else:
            return []

        if self._start_time_ns is None:
            self._start_time_ns = ts
            self._total_time_ns = max(self.resolution_time_ns - ts, 1)

        self._tick_count += 1
        if self._tick_count < self.warmup_ticks:
            return []

        if self._ref_last_ts == 0 or (ts - self._ref_last_ts) > self._max_ref_staleness_ns:
            return self._cancel_all()

        if ts - self._last_quote_ts < self._min_quote_interval_ns:
            return []

        current_vol = self._poly_vol()
        if current_vol > self._vol_gate:
            return self._cancel_all()

        remaining_ns = max(self.resolution_time_ns - ts, 0)
        tau_norm = max(min(float(remaining_ns / self._total_time_ns), 1.0), 0.0)

        if tau_norm < self._tau_pull_threshold:
            return self._cancel_all()

        divergence = abs(self._ref_fair_value - self._kalshi_mid_a)
        if divergence > self._divergence_gate:
            return self._cancel_all()

        overround = self._kalshi_mid_a + self._kalshi_mid_b
        if abs(overround - 1.0) > self._overround_gate:
            return self._cancel_all()

        self._last_quote_ts = ts
        self._quote_seq += 1

        p = max(min(self._ref_fair_value, 0.99), 0.01)
        tau_multiplier = 2.0 if tau_norm < self._tau_widen_threshold else 1.0
        spread = self.base_spread * tau_multiplier
        tick = self._tick_size

        q_a = self.positions.get_effective_quantity(self._kalshi_eid, self._sid_a) // self._lot_size
        q_b = self.positions.get_effective_quantity(self._kalshi_eid, self._sid_b) // self._lot_size
        D = q_a - q_b

        r_a = max(min(p - q_a * self._inventory_skew, 0.99), 0.01)
        r_b = max(min((1.0 - p) - q_b * self._inventory_skew, 0.99), 0.01)

        a_bid = (int((r_a - spread) * PRICE_SCALE) // tick) * tick
        a_bid = max(min(a_bid, PRICE_SCALE - tick), tick)
        a_ask = (int((r_a + spread) * PRICE_SCALE) + tick - 1) // tick * tick
        a_ask = max(min(a_ask, PRICE_SCALE - tick), tick)

        b_bid = (int((r_b - spread) * PRICE_SCALE) // tick) * tick
        b_bid = max(min(b_bid, PRICE_SCALE - tick), tick)
        b_ask = (int((r_b + spread) * PRICE_SCALE) + tick - 1) // tick * tick
        b_ask = max(min(b_ask, PRICE_SCALE - tick), tick)

        if a_bid >= a_ask:
            a_bid, a_ask = 0, 0
        if b_bid >= b_ask:
            b_bid, b_ask = 0, 0

        a_bid_scale = max(0.0, 1.0 - max(q_a, 0) / self.max_exposure)
        a_ask_scale = max(0.0, 1.0 - max(-q_a, 0) / self.max_exposure)
        b_bid_scale = max(0.0, 1.0 - max(q_b, 0) / self.max_exposure)
        b_ask_scale = max(0.0, 1.0 - max(-q_b, 0) / self.max_exposure)

        if D > 0:
            d_increase_scale = max(0.0, 1.0 - D / self.max_exposure)
            d_decrease_scale = 1.0
        elif D < 0:
            d_increase_scale = 1.0
            d_decrease_scale = max(0.0, 1.0 - abs(D) / self.max_exposure)
        else:
            d_increase_scale = 1.0
            d_decrease_scale = 1.0

        a_bid_sz = int(self.size * a_bid_scale * d_increase_scale)
        a_ask_sz = int(self.size * a_ask_scale * d_decrease_scale)
        b_bid_sz = int(self.size * b_bid_scale * d_decrease_scale)
        b_ask_sz = int(self.size * b_ask_scale * d_increase_scale)

        if self._quote_seq % 5 == 0:
            a_bid_sz = 0
            a_ask_sz = 0
            b_bid_sz = 0
            b_ask_sz = 0

        a_changed = (
            a_bid != self._last_a_bid or a_ask != self._last_a_ask
            or a_bid_sz != self._last_a_bid_sz or a_ask_sz != self._last_a_ask_sz
        )
        b_changed = (
            b_bid != self._last_b_bid or b_ask != self._last_b_ask
            or b_bid_sz != self._last_b_bid_sz or b_ask_sz != self._last_b_ask_sz
        )
        if not a_changed and not b_changed:
            return []

        self._quotes_active = True
        self._last_a_bid = a_bid
        self._last_a_ask = a_ask
        self._last_a_bid_sz = a_bid_sz
        self._last_a_ask_sz = a_ask_sz
        self._last_b_bid = b_bid
        self._last_b_ask = b_ask
        self._last_b_bid_sz = b_bid_sz
        self._last_b_ask_sz = b_ask_sz

        if self._metrics_buf is not None:
            row = self._metrics_buf.appendRow()
            self._metrics_buf.setLong(row, self._m_ts, ts)
            self._metrics_buf.setDouble(row, self._m_p, p)
            self._metrics_buf.setDouble(row, self._m_mid_a, self._kalshi_mid_a)
            self._metrics_buf.setDouble(row, self._m_mid_b, self._kalshi_mid_b)
            self._metrics_buf.setDouble(row, self._m_overround, overround)
            self._metrics_buf.setDouble(row, self._m_vol, current_vol)
            self._metrics_buf.setDouble(row, self._m_D, float(D))
            self._metrics_buf.setDouble(row, self._m_qa, float(q_a))
            self._metrics_buf.setDouble(row, self._m_qb, float(q_b))
            self._metrics_buf.setDouble(row, self._m_ab, a_bid / PRICE_SCALE)
            self._metrics_buf.setDouble(row, self._m_aa, a_ask / PRICE_SCALE)
            self._metrics_buf.setDouble(row, self._m_bb, b_bid / PRICE_SCALE)
            self._metrics_buf.setDouble(row, self._m_ba, b_ask / PRICE_SCALE)

        intents = []
        if a_changed:
            intents.append(Intent(
                exchange_id=self._kalshi_eid,
                security_id=self._sid_a,
                bid_price=a_bid if a_bid_sz > 0 else 0,
                bid_size=a_bid_sz,
                ask_price=a_ask if a_ask_sz > 0 else 0,
                ask_size=a_ask_sz,
            ))
        if b_changed:
            intents.append(Intent(
                exchange_id=self._kalshi_eid,
                security_id=self._sid_b,
                bid_price=b_bid if b_bid_sz > 0 else 0,
                bid_size=b_bid_sz,
                ask_price=b_ask if b_ask_sz > 0 else 0,
                ask_size=b_ask_sz,
            ))
        return intents

    def _cancel_all(self) -> list[Intent]:
        if not self._quotes_active:
            return []
        self._quotes_active = False
        self._last_a_bid = -1
        self._last_a_ask = -1
        self._last_a_bid_sz = -1
        self._last_a_ask_sz = -1
        self._last_b_bid = -1
        self._last_b_ask = -1
        self._last_b_bid_sz = -1
        self._last_b_ask_sz = -1
        return [
            Intent(exchange_id=self._kalshi_eid, security_id=self._sid_a),
            Intent(exchange_id=self._kalshi_eid, security_id=self._sid_b),
        ]
