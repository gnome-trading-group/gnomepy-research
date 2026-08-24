"""
Multi-venue informed market maker for binary prediction market contracts.

Uses a reference venue's book (e.g. Polymarket, deeper liquidity) as the fair
value signal and quotes HJB-optimal spreads on a quoting venue (e.g. Kalshi).
By anchoring to the cross-venue microprice rather than the local book mid,
the strategy reduces adverse selection from informed flow on the thinner venue.

The HJB value function V(t,p,q) is identical to PredictionMarketMaker — the
only difference is that p comes from the reference venue's Kalman-smoothed
microprice rather than the quoting venue's local mid.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gnomepy import ExecutionReport, Intent, Strategy
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

from gnomepy_research.signals.fair_value.microprice import MicropriceFairValue
from gnomepy_research.signals.operations.kalman import KalmanOperation
from gnomepy_research.solvers.prediction_market_hjb import (
    HJBParams,
    load_solution,
    optimal_ask,
    optimal_bid,
)

PRICE_SCALE = 1_000_000_000


class InformedPredictionMarketMaker(Strategy):
    def __init__(
        self,
        ref_listing_id: int,
        quote_listing_id: int,
        value_function_path: str,
        size: int = 1_000_000,
        max_position: int = 10,
        inventory_fade: float = 0.5,
        warmup_ticks: int = 50,
        max_ref_staleness_ns: int = 5_000_000_000,
        kalman_Q: float = 1e-4,
        kalman_R: float = 1e-2,
        maker_fee_rate: float = 0.0175,
        processing_time_ns: int = 0,
    ):
        self.size = size
        self.max_position = max_position
        self.inventory_fade = inventory_fade
        self.warmup_ticks = warmup_ticks
        self._max_ref_staleness_ns = max_ref_staleness_ns
        self._maker_fee_rate = maker_fee_rate
        self._processing_time_ns = processing_time_ns

        self._V, self._tau_grid, self._p_grid, self._q_levels, self._params = (
            load_solution(value_function_path)
        )
        self._interp = RegularGridInterpolator(
            (self._tau_grid, self._p_grid, self._q_levels), self._V,
            method='linear', bounds_error=False, fill_value=None,
        )

        registry = RegistryClient()

        ref_listings = registry.get_listing(listing_id=ref_listing_id)
        if not ref_listings:
            raise ValueError(f"No listing found for ref_listing_id={ref_listing_id}")
        self._ref_eid: int = ref_listings[0].exchange_id
        self._ref_sid: int = ref_listings[0].security_id

        quote_listings = registry.get_listing(listing_id=quote_listing_id)
        if not quote_listings:
            raise ValueError(f"No listing found for quote_listing_id={quote_listing_id}")
        self._quote_eid: int = quote_listings[0].exchange_id
        self._quote_sid: int = quote_listings[0].security_id

        specs = registry.get_listing_spec(listing_id=quote_listing_id)
        self._tick_size: int = int(specs[0].tick_size) if specs else 1
        self._lot_size: int = int(specs[0].lot_size) if specs else 1

        contracts = registry.get_event_contracts(security_id=self._quote_sid)
        if not contracts:
            raise ValueError(f"No event contract found for security_id={self._quote_sid}")
        events = registry.get_event(event_id=contracts[0].event_id)
        if not events or events[0].expiry is None:
            raise ValueError(f"Event {contracts[0].event_id} has no expiry set")
        expiry_dt = datetime.fromisoformat(events[0].expiry.replace('Z', '+00:00'))
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        self.resolution_time_ns: int = int(expiry_dt.timestamp() * 1e9)

        self._ref_fv = MicropriceFairValue()
        self._kalman = KalmanOperation(Q=kalman_Q, R=kalman_R)
        self._ref_fair_value: float = 0.5
        self._ref_last_ts: int = 0

        self._start_time_ns: int | None = None
        self._total_time_ns: int | None = None
        self._tick_count = 0
        self._metrics_buf = None

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("ipmm_signals")
        self._m_ts = buf.addLongColumn("timestamp")
        self._m_ref_fv = buf.addDoubleColumn("ref_fair_value")
        self._m_tau = buf.addDoubleColumn("tau_norm")
        self._m_q = buf.addDoubleColumn("position")
        self._m_zb = buf.addDoubleColumn("z_b")
        self._m_za = buf.addDoubleColumn("z_a")
        self._m_bid = buf.addDoubleColumn("optimal_bid")
        self._m_ask = buf.addDoubleColumn("optimal_ask")
        self._m_fee = buf.addDoubleColumn("fee_shift")
        self._m_stale = buf.addLongColumn("ref_staleness_ns")
        buf.freeze()
        self._metrics_buf = buf

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def _nearest_q_idx(self, position: int) -> int:
        q = float(np.clip(position, -self._params.Q, self._params.Q))
        return int(np.argmin(np.abs(self._q_levels - q)))

    def on_market_data(self, data: Schema) -> list[Intent]:
        try:
            return self._on_market_data_impl(data)
        except Exception:
            traceback.print_exc()
            raise

    def _on_market_data_impl(self, data: Schema) -> list[Intent]:
        if data.exchange_id == self._ref_eid and data.security_id == self._ref_sid:
            self._ref_fv.update(data.event_timestamp, data)
            if self._ref_fv.is_ready():
                smoothed = self._ref_fv.value() / PRICE_SCALE
                self._kalman.update(smoothed)
                if self._kalman.is_ready():
                    self._ref_fair_value = float(self._kalman.value())
                self._ref_last_ts = data.event_timestamp
            return []

        if data.exchange_id != self._quote_eid or data.security_id != self._quote_sid:
            return []

        ts = data.event_timestamp
        if self._start_time_ns is None:
            self._start_time_ns = ts
            self._total_time_ns = max(self.resolution_time_ns - ts, 1)

        self._tick_count += 1
        if self._tick_count < self.warmup_ticks:
            return []

        if self._ref_last_ts == 0 or (ts - self._ref_last_ts) > self._max_ref_staleness_ns:
            return []

        p = float(np.clip(self._ref_fair_value, self._params.p_min, self._params.p_max))

        remaining_ns = max(self.resolution_time_ns - ts, 0)
        tau_norm = float(np.clip(remaining_ns / self._total_time_ns, 0.0, 1.0))

        raw_position = self.positions.get_effective_quantity(self._quote_eid, self._quote_sid) // self._lot_size
        position = int(np.clip(raw_position, -self.max_position, self.max_position))

        qi = self._nearest_q_idx(position)
        params = self._params
        t_real = params.T * (1.0 - tau_norm)
        q_val = float(self._q_levels[qi])
        can_bid = qi < len(self._q_levels) - 1 and raw_position < self.max_position
        can_ask = qi > 0 and raw_position > -self.max_position

        bid_price_int = 0
        ask_price_int = 0
        pi_b = 0.0
        pi_a = 0.0
        z_b_val = 0.0
        z_a_val = 0.0

        pts = [[tau_norm, p, q_val]]
        if can_bid:
            pts.append([tau_norm, p, float(self._q_levels[qi + 1])])
        if can_ask:
            pts.append([tau_norm, p, float(self._q_levels[qi - 1])])
        vals = self._interp(pts)
        v_here = vals[0]

        fee_shift = self._maker_fee_rate * p * (1.0 - p) * PRICE_SCALE
        tick = self._tick_size

        if can_bid:
            z_b_val = (v_here - vals[1]) / params.delta_q
            pi_b = optimal_bid(z_b_val, p, t_real, params)
            raw = int(pi_b * PRICE_SCALE - fee_shift)
            bid_price_int = int(np.clip((raw // tick) * tick, tick, PRICE_SCALE - tick))

        if can_ask:
            z_a_val = (v_here - vals[-1]) / params.delta_q
            pi_a = optimal_ask(z_a_val, p, t_real, params)
            raw = int(pi_a * PRICE_SCALE + fee_shift)
            ask_price_int = int(np.clip(-(-raw // tick) * tick, tick, PRICE_SCALE - tick))

        if bid_price_int > 0 and ask_price_int > 0 and bid_price_int >= ask_price_int:
            return []

        if self._metrics_buf is not None:
            row = self._metrics_buf.appendRow()
            self._metrics_buf.setLong(row, self._m_ts, ts)
            self._metrics_buf.setDouble(row, self._m_ref_fv, p)
            self._metrics_buf.setDouble(row, self._m_tau, tau_norm)
            self._metrics_buf.setDouble(row, self._m_q, float(raw_position))
            self._metrics_buf.setDouble(row, self._m_zb, z_b_val)
            self._metrics_buf.setDouble(row, self._m_za, z_a_val)
            self._metrics_buf.setDouble(row, self._m_bid, pi_b)
            self._metrics_buf.setDouble(row, self._m_ask, pi_a)
            self._metrics_buf.setDouble(row, self._m_fee, fee_shift / PRICE_SCALE)
            self._metrics_buf.setLong(row, self._m_stale, ts - self._ref_last_ts)

        abs_pos = abs(raw_position)
        fade_start = self.max_position * self.inventory_fade
        if abs_pos <= fade_start:
            scale = 1.0
        else:
            scale = max((self.max_position - abs_pos) / (self.max_position - fade_start), 0.0)
        bid_size_scaled = max(int(self.size * scale), 0) if raw_position > 0 else self.size
        ask_size_scaled = max(int(self.size * scale), 0) if raw_position < 0 else self.size

        return [Intent(
            exchange_id=self._quote_eid,
            security_id=self._quote_sid,
            bid_price=bid_price_int,
            bid_size=bid_size_scaled if bid_price_int > 0 else 0,
            ask_price=ask_price_int,
            ask_size=ask_size_scaled if ask_price_int > 0 else 0,
        )]

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []
