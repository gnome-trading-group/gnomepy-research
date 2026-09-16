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


class OracleSpreadMaker(Strategy):
    def __init__(
        self,
        ref_listing_id: int,
        quote_listing_id: int,
        size: int = 3_000_000,
        max_position: int = 30,
        base_spread: float = 0.01,
        vol_gate: float = 0.005,
        vol_horizon: int = 20,
        kalman_Q: float = 1e-4,
        kalman_R: float = 1e-2,
        divergence_gate: float = 0.025,
        warmup_ticks: int = 50,
        max_ref_staleness_ns: int = 5_000_000_000,
        min_quote_interval_ns: int = 250_000_000,
        tau_pull_threshold: float = 0.05,
        tau_widen_threshold: float = 0.15,
        resolution_time_override_ns: int = 0,
        processing_time_ns: int = 5_000_000,
    ):
        self.size = size
        self.max_position = max_position
        self.base_spread = base_spread
        self._vol_gate = vol_gate
        self._divergence_gate = divergence_gate
        self.warmup_ticks = warmup_ticks
        self._max_ref_staleness_ns = max_ref_staleness_ns
        self._min_quote_interval_ns = min_quote_interval_ns
        self._tau_pull_threshold = tau_pull_threshold
        self._tau_widen_threshold = tau_widen_threshold
        self._processing_time_ns = processing_time_ns

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
        self._yes_sid: int = quote_listings[0].security_id

        specs = registry.get_listing_spec(listing_id=quote_listing_id)
        self._tick_size: int = int(specs[0].tick_size) if specs else 1
        self._lot_size: int = int(specs[0].lot_size) if specs else 1

        contracts = registry.get_event_contracts(security_id=self._yes_sid)
        if not contracts:
            raise ValueError(f"No event contract found for security_id={self._yes_sid}")

        if resolution_time_override_ns:
            self.resolution_time_ns: int = resolution_time_override_ns
        else:
            events = registry.get_event(event_id=contracts[0].event_id)
            if not events or events[0].expiry is None:
                raise ValueError(f"Event {contracts[0].event_id} has no expiry set")
            expiry_dt = datetime.fromisoformat(events[0].expiry.replace('Z', '+00:00'))
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
            self.resolution_time_ns = int(expiry_dt.timestamp() * 1e9)

        all_event_contracts = registry.get_event_contracts(event_id=contracts[0].event_id)
        no_contracts = [c for c in all_event_contracts if c.security_id != self._yes_sid]
        if not no_contracts:
            raise ValueError(f"No complementary contract for event_id={contracts[0].event_id}")
        no_sid = no_contracts[0].security_id
        no_listings = registry.get_listing(exchange_id=self._quote_eid, security_id=no_sid)
        if not no_listings:
            raise ValueError(
                f"No listing for NO contract sid={no_sid} on eid={self._quote_eid}"
            )
        self._no_sid: int = no_sid

        self._ref_fv = MicropriceFairValue()
        self._kalman = KalmanOperation(Q=kalman_Q, R=kalman_R)

        self._ref_fair_value: float = 0.5
        self._poly_mid: float = 0.5
        self._ref_last_ts: int = 0
        self._prev_ref_value: float = 0.0
        self._ref_returns: deque = deque(maxlen=100)

        self._prev_poly_mid: float = 0.0
        self._poly_returns: deque = deque(maxlen=vol_horizon)

        self._start_time_ns: int | None = None
        self._total_time_ns: int | None = None
        self._tick_count = 0
        self._last_quote_ts: int = 0
        self._metrics_buf = None

        self._last_yes_bid_price: int = -1
        self._last_no_bid_price: int = -1
        self._last_yes_bid_size: int = -1
        self._last_no_bid_size: int = -1
        self._quotes_active: bool = False
        self._bid_refresh_seq: int = 0

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("osm_signals")
        self._m_ts = buf.addLongColumn("timestamp")
        self._m_poly_mid = buf.addDoubleColumn("poly_mid")
        self._m_vol = buf.addDoubleColumn("poly_vol")
        self._m_yq = buf.addDoubleColumn("yes_qty")
        self._m_nq = buf.addDoubleColumn("no_qty")
        self._m_yb = buf.addDoubleColumn("yes_bid")
        self._m_nb = buf.addDoubleColumn("no_bid")
        self._m_matched = buf.addDoubleColumn("matched_pairs")
        buf.freeze()
        self._metrics_buf = buf

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []

    def _poly_vol(self) -> float:
        n = len(self._poly_returns)
        if n < 2:
            return 0.0
        mean = sum(self._poly_returns) / n
        variance = sum((r - mean) ** 2 for r in self._poly_returns) / n
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

        if data.exchange_id != self._quote_eid or data.security_id != self._yes_sid:
            return []

        if data.bid_price(0) > 0 and data.ask_price(0) > 0:
            self._poly_mid = (data.bid_price(0) + data.ask_price(0)) / (2.0 * PRICE_SCALE)

        if self._start_time_ns is None:
            self._start_time_ns = ts
            self._total_time_ns = max(self.resolution_time_ns - ts, 1)

        self._tick_count += 1
        if self._tick_count < self.warmup_ticks:
            return []

        if self._ref_last_ts == 0 or (ts - self._ref_last_ts) > self._max_ref_staleness_ns:
            return []

        if ts - self._last_quote_ts < self._min_quote_interval_ns:
            return []

        # Update vol at quoting cadence (250ms), not per market-event
        p_now = max(min(self._poly_mid, 0.99), 0.01)
        if self._prev_poly_mid > 0.0:
            self._poly_returns.append(math.log(p_now / self._prev_poly_mid))
        self._prev_poly_mid = p_now

        current_vol = self._poly_vol()
        if current_vol > self._vol_gate:
            if self._quotes_active:
                self._quotes_active = False
                self._last_yes_bid_price = -1
                self._last_no_bid_price = -1
                self._last_yes_bid_size = -1
                self._last_no_bid_size = -1
                return [
                    Intent(exchange_id=self._quote_eid, security_id=self._yes_sid),
                    Intent(exchange_id=self._quote_eid, security_id=self._no_sid),
                ]
            return []

        self._last_quote_ts = ts
        self._bid_refresh_seq += 1

        remaining_ns = max(self.resolution_time_ns - ts, 0)
        tau_norm = max(min(float(remaining_ns / self._total_time_ns), 1.0), 0.0)

        if tau_norm < self._tau_pull_threshold:
            if self._quotes_active:
                self._quotes_active = False
                self._last_yes_bid_price = -1
                self._last_no_bid_price = -1
                self._last_yes_bid_size = -1
                self._last_no_bid_size = -1
                return [
                    Intent(exchange_id=self._quote_eid, security_id=self._yes_sid),
                    Intent(exchange_id=self._quote_eid, security_id=self._no_sid),
                ]
            return []

        inst_divergence = abs(self._ref_fair_value - self._poly_mid)
        if inst_divergence > self._divergence_gate:
            return []

        yes_qty = max(self.positions.get_effective_quantity(self._quote_eid, self._yes_sid), 0) // self._lot_size
        no_qty = max(self.positions.get_effective_quantity(self._quote_eid, self._no_sid), 0) // self._lot_size

        p = max(min(self._poly_mid, 0.99), 0.01)

        tau_multiplier = 2.0 if tau_norm < self._tau_widen_threshold else 1.0
        spread = self.base_spread * tau_multiplier
        tick = self._tick_size

        yes_bid_price = (int((p - spread) * PRICE_SCALE) // tick) * tick
        yes_bid_price = max(min(yes_bid_price, PRICE_SCALE - tick), tick)

        no_bid_price = (int(((1.0 - p) - spread) * PRICE_SCALE) // tick) * tick
        no_bid_price = max(min(no_bid_price, PRICE_SCALE - tick), tick)

        yes_scale = max(0.0, 1.0 - yes_qty / self.max_position)
        no_scale = max(0.0, 1.0 - no_qty / self.max_position)

        yes_bid_size = max(int(self.size * yes_scale), 0)
        no_bid_size = max(int(self.size * no_scale), 0)

        if self._bid_refresh_seq % 5 == 0:
            yes_bid_size = 0
            no_bid_size = 0

        yes_changed = (
            yes_bid_price != self._last_yes_bid_price
            or yes_bid_size != self._last_yes_bid_size
        )
        no_changed = (
            no_bid_price != self._last_no_bid_price
            or no_bid_size != self._last_no_bid_size
        )
        if not yes_changed and not no_changed:
            return []

        self._quotes_active = True
        intents = []

        if yes_changed:
            self._last_yes_bid_price = yes_bid_price
            self._last_yes_bid_size = yes_bid_size
            intents.append(Intent(
                exchange_id=self._quote_eid,
                security_id=self._yes_sid,
                bid_price=yes_bid_price if yes_bid_size > 0 else 0,
                bid_size=yes_bid_size,
            ))

        if no_changed:
            self._last_no_bid_price = no_bid_price
            self._last_no_bid_size = no_bid_size
            intents.append(Intent(
                exchange_id=self._quote_eid,
                security_id=self._no_sid,
                bid_price=no_bid_price if no_bid_size > 0 else 0,
                bid_size=no_bid_size,
            ))

        if self._metrics_buf is not None:
            matched_pairs = float(min(yes_qty, no_qty))
            row = self._metrics_buf.appendRow()
            self._metrics_buf.setLong(row, self._m_ts, ts)
            self._metrics_buf.setDouble(row, self._m_poly_mid, self._poly_mid)
            self._metrics_buf.setDouble(row, self._m_vol, current_vol)
            self._metrics_buf.setDouble(row, self._m_yq, float(yes_qty))
            self._metrics_buf.setDouble(row, self._m_nq, float(no_qty))
            self._metrics_buf.setDouble(row, self._m_yb, yes_bid_price / PRICE_SCALE)
            self._metrics_buf.setDouble(row, self._m_nb, no_bid_price / PRICE_SCALE)
            self._metrics_buf.setDouble(row, self._m_matched, matched_pairs)

        return intents
