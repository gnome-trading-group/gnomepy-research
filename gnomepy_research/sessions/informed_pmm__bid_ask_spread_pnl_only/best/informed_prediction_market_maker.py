from __future__ import annotations

import traceback
from datetime import datetime, timezone

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gnomepy import ExecutionReport, Intent, Side, Strategy
from gnomepy.java.enums import ExecType
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

from gnomepy_research.signals.fair_value.microprice import MicropriceFairValue
from gnomepy_research.signals.operations.kalman import KalmanOperation
from gnomepy_research.solvers.prediction_market_hjb import (
    load_solution,
    optimal_ask,
    optimal_bid,
)

PRICE_SCALE = 1_000_000_000


class SpreadCapturePMM(Strategy):
    def __init__(
        self,
        ref_yes_listing_id: int,
        ref_no_listing_id: int,
        quote_yes_listing_id: int,
        quote_no_listing_id: int,
        value_function_path: str,
        size: int = 5_000_000,
        max_position: int = 20,
        inventory_fade: float = 0.5,
        warmup_ticks: int = 50,
        max_ref_staleness_ns: int = 5_000_000_000,
        kalman_Q: float = 1e-4,
        kalman_R: float = 1e-2,
        maker_fee_rate: float = 0.0,
        processing_time_ns: int = 5_000_000,
        min_quote_interval_ns: int = 1_000_000_000,
        resolution_time_override_ns: int = 0,
        close_spread_ticks: int = 1,
        yes_bid_tau_threshold: float = 0.0,
        no_bid_tau_threshold: float = 0.0,
    ):
        self.size = size
        self.max_position = max_position
        self._yes_bid_tau_threshold = yes_bid_tau_threshold
        self._no_bid_tau_threshold = no_bid_tau_threshold
        self.inventory_fade = inventory_fade
        self.warmup_ticks = warmup_ticks
        self._max_ref_staleness_ns = max_ref_staleness_ns
        self._maker_fee_rate = maker_fee_rate
        self._processing_time_ns = processing_time_ns
        self._min_quote_interval_ns = min_quote_interval_ns
        self._last_quote_ts: int = 0

        self._V, self._tau_grid, self._p_grid, self._q_levels, self._params = (
            load_solution(value_function_path)
        )
        self._interp = RegularGridInterpolator(
            (self._tau_grid, self._p_grid, self._q_levels), self._V,
            method='linear', bounds_error=False, fill_value=None,
        )

        registry = RegistryClient()

        ref_yes = registry.get_listing(listing_id=ref_yes_listing_id)
        if not ref_yes:
            raise ValueError(f"No listing for ref_yes_listing_id={ref_yes_listing_id}")
        self._ref_eid: int = ref_yes[0].exchange_id
        self._ref_yes_sid: int = ref_yes[0].security_id

        ref_no = registry.get_listing(listing_id=ref_no_listing_id)
        if not ref_no:
            raise ValueError(f"No listing for ref_no_listing_id={ref_no_listing_id}")
        self._ref_no_sid: int = ref_no[0].security_id

        quote_yes = registry.get_listing(listing_id=quote_yes_listing_id)
        if not quote_yes:
            raise ValueError(f"No listing for quote_yes_listing_id={quote_yes_listing_id}")
        self._quote_eid: int = quote_yes[0].exchange_id
        self._yes_sid: int = quote_yes[0].security_id

        quote_no = registry.get_listing(listing_id=quote_no_listing_id)
        if not quote_no:
            raise ValueError(f"No listing for quote_no_listing_id={quote_no_listing_id}")
        self._no_twin_sid: int = quote_no[0].security_id

        specs = registry.get_listing_spec(listing_id=quote_yes_listing_id)
        self._tick_size: int = int(specs[0].tick_size) if specs else 1
        self._lot_size: int = int(specs[0].lot_size) if specs else 1
        self._close_spread_int: int = close_spread_ticks * self._tick_size

        contracts = registry.get_event_contracts(security_id=self._yes_sid)
        if not contracts:
            raise ValueError(f"No event contract for security_id={self._yes_sid}")
        events = registry.get_event(event_id=contracts[0].event_id)
        if not events or events[0].expiry is None:
            raise ValueError(f"Event has no expiry")
        expiry_dt = datetime.fromisoformat(events[0].expiry.replace('Z', '+00:00'))
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        registry_resolution_ns = int(expiry_dt.timestamp() * 1e9)
        self.resolution_time_ns: int = (
            resolution_time_override_ns if resolution_time_override_ns > 0
            else registry_resolution_ns
        )

        self._ref_yes_fv = MicropriceFairValue()
        self._ref_no_fv = MicropriceFairValue()
        self._kalman_yes = KalmanOperation(Q=kalman_Q, R=kalman_R)
        self._kalman_no = KalmanOperation(Q=kalman_Q, R=kalman_R)
        self._ref_yes_value: float = 0.5
        self._ref_no_value: float = 0.5
        self._ref_yes_ready: bool = False
        self._ref_no_ready: bool = False
        self._ref_last_ts: int = 0

        self._yes_book_bid: int = 0
        self._yes_book_ask: int = 0
        self._no_book_bid: int = 0
        self._no_book_ask: int = 0

        self._last_posted_bid: dict[int, int] = {}
        self._last_posted_ask: dict[int, int] = {}
        self._filled_qty: dict[int, float] = {}

        self._start_time_ns: int | None = None
        self._total_time_ns: int | None = None
        self._tick_count: int = 0
        self._metrics_buf = None

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("sc_pmm")
        self._m_ts = buf.addLongColumn("timestamp")
        self._m_ref_fv = buf.addDoubleColumn("ref_fair_value")
        self._m_tau = buf.addDoubleColumn("tau_norm")
        self._m_yes_q = buf.addDoubleColumn("yes_position")
        self._m_no_q = buf.addDoubleColumn("no_position")
        self._m_yes_bid = buf.addDoubleColumn("yes_bid")
        self._m_yes_ask = buf.addDoubleColumn("yes_ask")
        self._m_no_bid = buf.addDoubleColumn("no_bid")
        self._m_no_ask = buf.addDoubleColumn("no_ask")
        buf.freeze()
        self._metrics_buf = buf

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def _nearest_q_idx(self, position: int) -> int:
        q = float(np.clip(position, -self._params.Q, self._params.Q))
        return int(np.argmin(np.abs(self._q_levels - q)))

    def _compute_hjb_quotes(
        self,
        tau_norm: float,
        p: float,
        position: int,
        book_bid: int,
        book_ask: int,
    ) -> tuple[int, int]:
        can_bid = position < self.max_position
        can_ask = position > 0

        qi = self._nearest_q_idx(position)
        params = self._params
        t_real = params.T * (1.0 - tau_norm)
        q_val = float(self._q_levels[qi])

        pts = [[tau_norm, p, q_val]]
        idx_bid = -1
        idx_ask = -1
        if can_bid and qi + 1 < len(self._q_levels):
            idx_bid = len(pts)
            pts.append([tau_norm, p, float(self._q_levels[qi + 1])])
        if can_ask and qi > 0:
            idx_ask = len(pts)
            pts.append([tau_norm, p, float(self._q_levels[qi - 1])])

        vals = self._interp(pts)
        v_here = vals[0]

        fee_shift = self._maker_fee_rate * p * (1.0 - p) * PRICE_SCALE
        tick = self._tick_size

        bid_price_int = 0
        ask_price_int = 0

        if can_bid and idx_bid >= 0:
            z_b = (v_here - vals[idx_bid]) / params.delta_q
            pi_b = optimal_bid(z_b, p, t_real, params)
            raw = int(pi_b * PRICE_SCALE - fee_shift)
            bid_price_int = int(np.clip((raw // tick) * tick, tick, PRICE_SCALE - tick))

        if can_ask and idx_ask >= 0:
            z_a = (v_here - vals[idx_ask]) / params.delta_q
            pi_a = optimal_ask(z_a, p, t_real, params)
            raw = int(pi_a * PRICE_SCALE + fee_shift)
            ask_price_int = int(np.clip(-(-raw // tick) * tick, tick, PRICE_SCALE - tick))

        if bid_price_int > 0 and ask_price_int > 0 and bid_price_int >= ask_price_int:
            return 0, 0

        if bid_price_int > 0 and book_ask > 0 and bid_price_int >= book_ask:
            bid_price_int = max(book_ask - tick, tick)
        if ask_price_int > 0 and book_bid > 0 and ask_price_int <= book_bid:
            ask_price_int = min(book_bid + tick, PRICE_SCALE - tick)

        if bid_price_int > 0 and ask_price_int > 0 and bid_price_int >= ask_price_int:
            return 0, 0

        return bid_price_int, ask_price_int

    def _compute_side_sizes(self, position: int) -> tuple[int, int]:
        fade_start = int(self.max_position * self.inventory_fade)
        if position >= self.max_position:
            bid_sz = 0
        elif position > fade_start:
            scale = (self.max_position - position) / max(self.max_position - fade_start, 1)
            bid_sz = max(int(self.size * scale), 0)
        else:
            bid_sz = self.size
        ask_sz = self.size if position > 0 else 0
        return bid_sz, ask_sz

    def on_market_data(self, data: Schema) -> list[Intent]:
        try:
            return self._on_market_data_impl(data)
        except Exception:
            traceback.print_exc()
            raise

    def _on_market_data_impl(self, data: Schema) -> list[Intent]:
        eid = data.exchange_id
        sid = data.security_id
        ts = data.event_timestamp

        if eid == self._ref_eid and sid == self._ref_yes_sid:
            self._ref_yes_fv.update(ts, data)
            if self._ref_yes_fv.is_ready():
                smoothed = self._ref_yes_fv.value() / PRICE_SCALE
                self._kalman_yes.update(smoothed)
                if self._kalman_yes.is_ready():
                    self._ref_yes_value = float(self._kalman_yes.value())
                    self._ref_yes_ready = True
            self._ref_last_ts = ts
            return []

        if eid == self._ref_eid and sid == self._ref_no_sid:
            self._ref_no_fv.update(ts, data)
            if self._ref_no_fv.is_ready():
                smoothed = self._ref_no_fv.value() / PRICE_SCALE
                self._kalman_no.update(smoothed)
                if self._kalman_no.is_ready():
                    self._ref_no_value = float(self._kalman_no.value())
                    self._ref_no_ready = True
            self._ref_last_ts = ts
            return []

        if eid == self._quote_eid and sid == self._yes_sid:
            self._yes_book_bid = max(int(data.bid_price(0)), 0)
            self._yes_book_ask = int(data.ask_price(0))
        elif eid == self._quote_eid and sid == self._no_twin_sid:
            self._no_book_bid = max(int(data.bid_price(0)), 0)
            self._no_book_ask = int(data.ask_price(0))
        else:
            return []

        if self._start_time_ns is None:
            self._start_time_ns = ts
            self._total_time_ns = max(self.resolution_time_ns - ts, 1)

        self._tick_count += 1
        if self._tick_count < self.warmup_ticks:
            return []

        if not self._ref_yes_ready or (ts - self._ref_last_ts) > self._max_ref_staleness_ns:
            return []

        if ts - self._last_quote_ts < self._min_quote_interval_ns:
            return []
        self._last_quote_ts = ts

        if self._ref_yes_ready and self._ref_no_ready:
            p_raw = (self._ref_yes_value + self._ref_no_value) / 2.0
        else:
            p_raw = self._ref_yes_value
        p = float(np.clip(p_raw, self._params.p_min, self._params.p_max))
        p_no = float(np.clip(1.0 - p_raw, self._params.p_min, self._params.p_max))

        remaining_ns = max(self.resolution_time_ns - ts, 0)
        tau_norm = float(np.clip(remaining_ns / self._total_time_ns, 0.0, 1.0))

        yes_qty_eff = int(np.clip(
            max(self.positions.get_effective_quantity(self._quote_eid, self._yes_sid), 0) // self._lot_size,
            0, self.max_position,
        ))
        no_qty_eff = int(np.clip(
            max(self.positions.get_effective_quantity(self._quote_eid, self._no_twin_sid), 0) // self._lot_size,
            0, self.max_position,
        ))
        yes_qty_actual = int(np.clip(
            max(self._filled_qty.get(self._yes_sid, 0.0), 0.0) // self._lot_size,
            0, self.max_position,
        ))
        no_qty_actual = int(np.clip(
            max(self._filled_qty.get(self._no_twin_sid, 0.0), 0.0) // self._lot_size,
            0, self.max_position,
        ))

        yes_bid, yes_ask = self._compute_hjb_quotes(
            tau_norm, p, yes_qty_actual, self._yes_book_bid, self._yes_book_ask,
        )
        no_bid, no_ask = self._compute_hjb_quotes(
            tau_norm, p_no, no_qty_actual, self._no_book_bid, self._no_book_ask,
        )

        yes_bid_sz, yes_ask_sz = self._compute_side_sizes(yes_qty_eff)
        no_bid_sz, no_ask_sz = self._compute_side_sizes(no_qty_eff)
        yes_ask_sz = min(yes_ask_sz, int(max(self._filled_qty.get(self._yes_sid, 0.0), 0.0)))
        no_ask_sz = min(no_ask_sz, int(max(self._filled_qty.get(self._no_twin_sid, 0.0), 0.0)))

        if self._yes_bid_tau_threshold > 0.0 and tau_norm < self._yes_bid_tau_threshold:
            yes_bid = 0
            yes_bid_sz = 0

        if self._no_bid_tau_threshold > 0.0 and tau_norm < self._no_bid_tau_threshold:
            no_bid = 0
            no_bid_sz = 0

        if self._metrics_buf is not None:
            row = self._metrics_buf.appendRow()
            self._metrics_buf.setLong(row, self._m_ts, ts)
            self._metrics_buf.setDouble(row, self._m_ref_fv, p)
            self._metrics_buf.setDouble(row, self._m_tau, tau_norm)
            self._metrics_buf.setDouble(row, self._m_yes_q, float(yes_qty_actual))
            self._metrics_buf.setDouble(row, self._m_no_q, float(no_qty_actual))
            self._metrics_buf.setDouble(row, self._m_yes_bid, yes_bid / PRICE_SCALE if yes_bid else 0.0)
            self._metrics_buf.setDouble(row, self._m_yes_ask, yes_ask / PRICE_SCALE if yes_ask else 0.0)
            self._metrics_buf.setDouble(row, self._m_no_bid, no_bid / PRICE_SCALE if no_bid else 0.0)
            self._metrics_buf.setDouble(row, self._m_no_ask, no_ask / PRICE_SCALE if no_ask else 0.0)

        self._last_posted_bid[self._yes_sid] = yes_bid
        self._last_posted_ask[self._yes_sid] = yes_ask
        self._last_posted_bid[self._no_twin_sid] = no_bid
        self._last_posted_ask[self._no_twin_sid] = no_ask

        return [
            Intent(
                exchange_id=self._quote_eid,
                security_id=self._yes_sid,
                bid_price=yes_bid,
                bid_size=yes_bid_sz if yes_bid > 0 else 0,
                ask_price=yes_ask,
                ask_size=yes_ask_sz if yes_ask > 0 else 0,
            ),
            Intent(
                exchange_id=self._quote_eid,
                security_id=self._no_twin_sid,
                bid_price=no_bid,
                bid_size=no_bid_sz if no_bid > 0 else 0,
                ask_price=no_ask,
                ask_size=no_ask_sz if no_ask > 0 else 0,
            ),
        ]

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        if report.exec_type not in (ExecType.FILL, ExecType.PARTIAL_FILL):
            return []

        eid = report.exchange_id
        sid = report.security_id
        if eid != self._quote_eid or sid not in (self._yes_sid, self._no_twin_sid):
            return []

        fill_price = report.fill_price
        fill_qty = report.filled_qty
        last_bid = self._last_posted_bid.get(sid, 0)
        last_ask = self._last_posted_ask.get(sid, 0)

        if last_bid > 0 and last_ask > 0:
            is_bid = abs(fill_price - last_bid) <= abs(fill_price - last_ask)
        elif last_bid > 0:
            is_bid = True
        elif last_ask > 0:
            is_bid = False
        else:
            return []

        if is_bid:
            self._filled_qty[sid] = self._filled_qty.get(sid, 0.0) + fill_qty
        else:
            self._filled_qty[sid] = self._filled_qty.get(sid, 0.0) - fill_qty

        if not is_bid:
            return []

        self._last_quote_ts = report.timestamp_recv
        tick = self._tick_size
        book_bid = self._yes_book_bid if sid == self._yes_sid else self._no_book_bid
        close_ask = int(np.clip(
            fill_price + self._close_spread_int,
            (book_bid + tick) if book_bid > 0 else fill_price + tick,
            PRICE_SCALE - tick,
        ))
        other_sid = self._no_twin_sid if sid == self._yes_sid else self._yes_sid

        return [
            Intent(exchange_id=eid, security_id=sid,
                   bid_price=0, bid_size=0, ask_price=close_ask,
                   ask_size=fill_qty),
            Intent(exchange_id=eid, security_id=other_sid,
                   bid_price=0, bid_size=0, ask_price=0, ask_size=0),
        ]
