"""Cross-venue prediction-market arbitrage specialised for N>=3 outcome events.

Direct fork of sessions/cross_prediction_arb at the point its chase regression and
correctness backlog were fixed. Identical logic; diverges from here.

Why fork: measured cross-venue dislocation lifetimes differ by an order of magnitude
between 2-outcome and 3-outcome markets. Across 15 EQUIVALENT pairs the median episode
lasted ~120ms, but the 3-way soccer market (ManCity/Draw/Sunderland) ran 400-440ms --
long enough to survive a ~37ms eu-west-1 <-> us-east-1 hop, which the 2-outcome markets
are not.

Known limitation inherited from the parent, and the first thing to address here:
_listing_is_busy serialises any pairings that share a listing. At N=3 the cartesian
product yields 6 pairings and each listing appears in 4 of them, so in practice only
one pairing is ever active. See the note on that method.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import product
from typing import NamedTuple

from gnomepy import ExecutionReport, Intent, OrderType, RejectReason, Scales, Side, Strategy
from gnomepy.java.enums import Action, ExecType
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

PRICE_SCALE = Scales.PRICE   # 1_000_000_000
SIZE_SCALE = Scales.SIZE      # 1_000_000

# PM taker orders take ~250ms (taker_delay) + 50ms (latency) to settle.
# Don't make cancel decisions until this window has passed.
_PM_TAKER_SETTLE_NS = 500_000_000

# After sending a K maker cancel, wait this long before re-entering.
# Without this: cancel + new K maker arrive simultaneously; the engine cancels
# the new order instead of the old one. 200ms >> 2×latency ensures settlement.
_CANCEL_SETTLE_NS = 200_000_000

# Prediction-market venues quote in whole cents. The registry currently reports a
# $0.001 tick for Kalshi, which the recorded books contradict — see _effective_tick.
_MIN_TICK = PRICE_SCALE // 100

# Close retries before a pairing stops trying and returns to SCANNING. Without a cap
# an un-closeable residual (sub-lot, or below the min-notional floor) looped forever.
_MAX_CLOSE_ATTEMPTS = 5


# ---------------------------------------------------------------------------
# State machine phases
# ---------------------------------------------------------------------------

class Phase(IntEnum):
    SCANNING = 0
    ENTERING = 1
    PARTIAL_FILL = 2
    UNWINDING = 3


# ---------------------------------------------------------------------------
# Book storage
# ---------------------------------------------------------------------------

class BookLevel(NamedTuple):
    price: int
    size: int


@dataclass
class Book:
    bids: list[BookLevel] = field(default_factory=list)
    asks: list[BookLevel] = field(default_factory=list)
    last_update_ts: int = 0

    def best_bid(self) -> int:
        return self.bids[0].price if self.bids else 0

    def best_ask(self) -> int:
        return self.asks[0].price if self.asks else 0

    def mid(self) -> int:
        b, a = self.best_bid(), self.best_ask()
        if b > 0 and a > 0:
            return (b + a) // 2
        return b or a


# ---------------------------------------------------------------------------
# Leg state
# ---------------------------------------------------------------------------

@dataclass
class LegState:
    listing: tuple[int, int]
    target_qty: int = 0
    filled_qty: int = 0
    fill_cost: int = 0  # sum of price * qty (scaled)
    entry_fees: float = 0.0  # cumulative dollar fees paid on entry fills
    entry_bid: int = 0
    last_chase_bid: int = 0
    posted: bool = True

    @property
    def is_filled(self) -> bool:
        return self.target_qty > 0 and self.filled_qty >= self.target_qty

    def record_fill(self, price: int, qty: int, fee: float = 0.0) -> None:
        self.filled_qty += qty
        self.fill_cost += price * qty
        self.entry_fees += fee


# ---------------------------------------------------------------------------
# Pairings
# ---------------------------------------------------------------------------

@dataclass
class Pairing:
    index: int
    label: str
    legs: list[tuple[int, int]]


@dataclass
class PairingState:
    pairing: Pairing
    phase: Phase = Phase.SCANNING
    legs: list[LegState] = field(default_factory=list)
    partial_fill_since: int | None = None
    last_close_ts: int = 0
    entry_ts: int = 0
    last_cancel_ts: int = 0
    base_qty: int = 0
    close_attempts: int = 0
    completion_ts: int = 0


# ---------------------------------------------------------------------------
# Cost model (pluggable)
# ---------------------------------------------------------------------------

class CostComponent:
    def cost(self, listing: tuple[int, int], price_scaled: int, qty: int, **kwargs) -> float:
        return 0.0

    def on_book_update(self, listing: tuple[int, int], book: Book) -> None:
        pass


class FeeCost(CostComponent):
    def __init__(
        self,
        maker_rates: dict[str, float],
        taker_rates: dict[str, float],
        exchange_id_to_label: dict[int, str],
    ):
        self._maker: dict[str, float] = {str(k): float(v) for k, v in maker_rates.items()}
        self._taker: dict[str, float] = {str(k): float(v) for k, v in taker_rates.items()}
        self._id_to_label = exchange_id_to_label

    def cost(self, listing, price_scaled, qty, maker=True, **kwargs) -> float:
        label = self._id_to_label.get(listing[0], "")
        rate = self._maker.get(label, 0.0) if maker else self._taker.get(label, 0.0)
        p = price_scaled / PRICE_SCALE
        return rate * p * (1.0 - p)


class FillRiskCost(CostComponent):
    """Penalizes entries when fill probability on any leg is low.

    Uses either a simple exponential model or CST birth-death approximation.
    Stuck cost = (1 - P_fill) * spread * unwind_mult, attributed per leg.
    """
    def __init__(
        self,
        fill_risk_lambda: float,
        unwind_spread_mult: float = 2.0,
        use_cst_model: bool = False,
    ):
        self._lambda = fill_risk_lambda
        self._unwind_mult = unwind_spread_mult
        self._use_cst = use_cst_model
        self._spreads: dict[tuple[int, int], float] = {}
        self._queue_depths: dict[tuple[int, int], int] = {}

    def on_book_update(self, listing, book: Book) -> None:
        bid, ask = book.best_bid(), book.best_ask()
        if bid > 0 and ask > 0:
            self._spreads[listing] = (ask - bid) / PRICE_SCALE
        if book.bids:
            self._queue_depths[listing] = book.bids[0].size

    def fill_prob(self, listing: tuple[int, int]) -> float:
        spread = self._spreads.get(listing, 0.0)
        if self._use_cst:
            queue = self._queue_depths.get(listing, 0) / SIZE_SCALE
            cancel_rate = 0.7
            cross_rate = max(0.01, 1.0 - cancel_rate)
            ratio = cancel_rate / (cancel_rate + cross_rate)
            return 1.0 - ratio ** max(queue, 1)
        return math.exp(-self._lambda * spread) if spread > 0 else 1.0

    def cost(self, listing, price_scaled, qty, other_listing=None, **kwargs) -> float:
        if other_listing is None:
            return 0.0
        p_fill = self.fill_prob(other_listing)
        spread = self._spreads.get(other_listing, 0.0)
        return (1.0 - p_fill) * spread * self._unwind_mult


class DepthCoverageCost(CostComponent):
    """Penalizes entries when the available ask depth is thin relative to our size."""
    def __init__(self, depth_mult: float, price_window_cents: int = 2):
        self._depth_mult = depth_mult
        self._window_scaled = price_window_cents * (PRICE_SCALE // 100)
        self._cheap_depth: dict[tuple[int, int], float] = {}

    def on_book_update(self, listing, book: Book) -> None:
        depth = 0
        if book.asks:
            ref = book.asks[0].price
            for level in book.asks:
                if level.price - ref > self._window_scaled:
                    break
                depth += level.size
        self._cheap_depth[listing] = depth / SIZE_SCALE if depth > 0 else 0.0

    def cost(self, listing, price_scaled, qty, **kwargs) -> float:
        depth = self._cheap_depth.get(listing, 0.0)
        if depth <= 0.0:
            return 999.0
        coverage = (qty / SIZE_SCALE) / depth
        return self._depth_mult * max(0.0, coverage - 0.5)


class CompositeCost:
    def __init__(self, components: list[CostComponent]):
        self._components = components

    def total_cost(self, listing, price_scaled, qty, **kwargs) -> float:
        return sum(c.cost(listing, price_scaled, qty, **kwargs) for c in self._components)

    def on_book_update(self, listing: tuple[int, int], book: Book) -> None:
        for c in self._components:
            c.on_book_update(listing, book)


# ---------------------------------------------------------------------------
# Price model (pluggable)
# ---------------------------------------------------------------------------

class PriceModel:
    def compute_bid(self, listing, best_bid: int, best_ask: int) -> int:
        raise NotImplementedError

    def on_book_update(self, listing: tuple[int, int], book: Book) -> None:
        pass


class JoinBestBidModel(PriceModel):
    def compute_bid(self, listing, best_bid, best_ask):
        return best_bid


class AggressiveModel(PriceModel):
    def __init__(self, improve_bps: float = 50.0):
        self._improve_bps = improve_bps

    def compute_bid(self, listing, best_bid, best_ask):
        mid = (best_bid + best_ask) // 2
        price = best_bid + int(mid * self._improve_bps / 10000)
        # best_ask - 1 is one raw scale unit below the ask and lands off-tick; the
        # caller quantizes, which floors it back onto the grid.
        return min(price, best_ask - 1)


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

class CrossPredictionArbMulti(Strategy):
    def __init__(
        self,
        outcomes: list[dict[str, int]],  # [{"pm": listing_id, "k": listing_id}, ...]
        max_position: int = 100,
        min_edge_cents: float = 2.0,
        min_contract_price: float = 0.25,
        max_price_divergence_cents: float = 0.0,
        imbalance_timeout_ns: int = 30_000_000_000,
        max_staleness_ns: int = 60_000_000_000,
        processing_time_ns: int = 5_000_000,
        # Fee rates keyed by exchange label (e.g. "pm", "k")
        maker_fee_rates: dict[str, float] | None = None,
        taker_fee_rates: dict[str, float] | None = None,
        # Cost model
        fill_risk_lambda: float = 0.0,
        unwind_spread_mult: float = 2.0,
        depth_coverage_mult: float = 0.0,
        use_cst_fill_model: bool = False,
        price_window_cents: int = 2,
        # Price model
        price_model: str = "join_best_bid",
        price_improve_bps: float = 50.0,
        # Taker labels — legs on these exchanges cross the ask for immediate fills
        taker_labels: list[str] | None = None,
        # Dutch book labels — exchanges where single-exchange pairings are allowed
        dutch_book_labels: list[str] | None = None,
        # Settlement risk (disabled by default — sports resolve before official expiry)
        # Chase: how long to stay in arb-positive zone before entering damage-control
        maker_patience_ns: int = 30_000_000_000,
        enable_chase: bool = False,
        passive_unwind_attempts: int = 0,
        fill_risk_horizon_ns: int = 0,
        flow_window_ns: int = 600_000_000_000,
        taker_complete_edge_cents: float | None = None,
        lead_with_illiquid: bool = False,
        # Diagnostics
        debug: bool = False,
    ):
        registry = RegistryClient()

        if maker_fee_rates is None:
            maker_fee_rates = {"pm": 0.0, "k": 0.0175}
        if taker_fee_rates is None:
            taker_fee_rates = {"pm": 0.07, "k": 0.07}
        self._maker_fee_rates: dict[str, float] = {str(k): float(v) for k, v in maker_fee_rates.items()}
        self._taker_fee_rates: dict[str, float] = {str(k): float(v) for k, v in taker_fee_rates.items()}
        self._taker_labels: set[str] = set(str(x) for x in taker_labels) if taker_labels else set()
        self._dutch_book_labels: set[str] = set(str(x) for x in dutch_book_labels) if dutch_book_labels else set()

        self._max_position = max_position * SIZE_SCALE
        self._min_edge = min_edge_cents / 100.0
        self._min_contract_price_scaled = int(min_contract_price * PRICE_SCALE)
        self._max_price_divergence = max_price_divergence_cents / 100.0
        self._imbalance_timeout_ns = imbalance_timeout_ns
        self._maker_patience_ns = maker_patience_ns
        # The chase walks a resting maker bid upward until it fills. Measured on the
        # Seahawks event it costs money at every setting tried: no chase $106.52,
        # chase as shipped $66.55, chase with a correct ceiling $47.38, chase capped
        # at the ceiling $58.88. Off by default until a variant beats the baseline.
        self._enable_chase = enable_chase
        # Close attempts made passively (resting offers) before crossing the spread.
        # DEFAULT 0 (off). Measured on the soccer event: a passive exit posts an offer
        # while the entry logic posts a bid, and the two trade against each other --
        # 1,108 fills, $73k bought against $71.9k sold, $1,567 of spread bled plus
        # $2,118 of fees, for a net position of zero. Aggressive exit lost $758 on the
        # same config; passive lost $3,685. Do not enable without an entry lockout.
        self._passive_unwind_attempts = passive_unwind_attempts
        # Horizon over which a resting maker bid is assumed to have to fill. 0 disables
        # the expected-value discount entirely (old behaviour: every maker leg assumed
        # to fill with certainty).
        self._fill_risk_horizon_ns = fill_risk_horizon_ns
        self._flow_window_ns = flow_window_ns
        self._lead_with_illiquid = lead_with_illiquid
        self._taker_complete_edge = (
            None if taker_complete_edge_cents is None else taker_complete_edge_cents / 100.0
        )
        self._unwind_spread_mult = unwind_spread_mult
        self._max_staleness_ns = max_staleness_ns
        self._processing_time_ns = processing_time_ns
        self._debug = debug

        def resolve(listing_id: int) -> tuple[tuple[int, int], tuple[int, int, int], int]:
            results = registry.get_listing(listing_id=listing_id)
            if not results:
                raise ValueError(f"No listing for listing_id={listing_id}")
            specs = registry.get_listing_spec(listing_id=listing_id)
            if not specs:
                raise ValueError(f"No listing spec for listing_id={listing_id}")
            listing = (results[0].exchange_id, results[0].security_id)
            spec = (int(specs[0].min_notional or 0), int(specs[0].lot_size or 0), int(specs[0].min_size or 0))
            tick_size = int(specs[0].tick_size) if specs[0].tick_size else PRICE_SCALE // 100
            return listing, spec, tick_size

        # Build exchange_id → label mapping and resolve per-outcome listings
        self._exchange_id_to_label: dict[int, str] = {}
        self._tick_sizes: dict[tuple[int, int], int] = {}
        outcome_listings: list[list[tuple[int, int]]] = []
        listing_specs: dict[tuple[int, int], tuple[int, int, int]] = {}
        for outcome in outcomes:
            resolved = []
            for label, lid in outcome.items():
                r, spec, tick_size = resolve(lid)
                self._exchange_id_to_label[r[0]] = label
                self._tick_sizes[r] = tick_size
                resolved.append(r)
                listing_specs[r] = spec
            outcome_listings.append(resolved)

        # Generate all exchange assignments via cartesian product
        n_outcomes = len(outcomes)
        choices = [[(oi, lst) for lst in outcome_listings[oi]] for oi in range(n_outcomes)]
        raw_pairings: list[tuple[int, str, list[tuple[int, int]]]] = []
        idx = 0
        for combo in product(*choices):
            legs = [lst for _, lst in combo]
            exchange_ids = {lst[0] for lst in legs}
            if len(exchange_ids) < 2:
                eid = next(iter(exchange_ids))
                lbl = self._exchange_id_to_label.get(eid, "")
                if lbl in self._dutch_book_labels:
                    raw_pairings.append((idx, f"DUTCH_{lbl.upper()}", legs))
                    idx += 1
                continue
            label_parts = [
                f"{self._exchange_id_to_label[lst[0]].upper()}_O{oi}"
                for oi, lst in combo
            ]
            raw_pairings.append((idx, "+".join(label_parts), legs))
            idx += 1

        if len(raw_pairings) > 20:
            raise ValueError(f"Too many pairings ({len(raw_pairings)}) for {n_outcomes} outcomes")

        self._pairings: list[Pairing] = []
        self._tracked_listings: set[tuple[int, int]] = set()
        for p_idx, p_label, p_legs in raw_pairings:
            sorted_legs = sorted(
                p_legs,
                key=lambda lg: 0 if self._exchange_id_to_label.get(lg[0], "") in self._taker_labels else 1,
            )
            self._pairings.append(Pairing(index=p_idx, label=p_label, legs=sorted_legs))
            for leg in p_legs:
                self._tracked_listings.add(leg)

        self._listing_specs: dict[tuple[int, int], tuple[int, int, int]] = listing_specs

        # Time-decayed volume of trades that hit the bid, per listing: (volume, last trade ts).
        self._sell_flow: dict[tuple[int, int], tuple[float, int]] = {}
        self._now_ns = 0

        # Book storage
        self._books: dict[tuple[int, int], Book] = {lst: Book() for lst in self._tracked_listings}

        # Cost model
        cost_components: list[CostComponent] = [
            FeeCost(maker_fee_rates, taker_fee_rates, self._exchange_id_to_label)
        ]
        if fill_risk_lambda > 0.0:
            cost_components.append(FillRiskCost(fill_risk_lambda, unwind_spread_mult, use_cst_fill_model))
        if depth_coverage_mult > 0.0:
            cost_components.append(DepthCoverageCost(depth_coverage_mult, price_window_cents))
        self._cost_model = CompositeCost(cost_components)

        # Price model
        if price_model == "aggressive":
            self._price_model: PriceModel = AggressiveModel(price_improve_bps)
        else:
            self._price_model = JoinBestBidModel()

        # Per-pairing state machines. Pairings that share no listing run concurrently;
        # pairings sharing a listing are serialized by _listing_is_busy.
        self._pairing_states: list[PairingState] = [
            PairingState(pairing=p) for p in self._pairings
        ]
        # Lookup: listing → pairing states that include it (for fill routing)
        self._listing_to_ps: dict[tuple[int, int], list[PairingState]] = {}
        for ps in self._pairing_states:
            for leg in ps.pairing.legs:
                self._listing_to_ps.setdefault(leg, []).append(ps)

        # Metrics buffer (populated in register_metrics)
        self._metrics_buf = None

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("cpa_signals")
        self._m_ts = buf.addLongColumn("timestamp")
        self._m_phase = buf.addIntColumn("phase")
        self._m_pairing = buf.addIntColumn("pairing")
        self._m_edge = buf.addDoubleColumn("edge")
        self._m_qty = buf.addLongColumn("qty")
        buf.freeze()
        self._metrics_buf = buf

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _log_intents(self, context: str, intents: list[Intent], ts: int = 0) -> None:
        if not self._debug or not intents:
            return
        ts_s = f"{ts / 1e9:.3f}" if ts else "?"
        print(f"[DEBUG {ts_s}] {context}: {len(intents)} intent(s)")
        CENT = PRICE_SCALE // 100
        for intent in intents:
            eid = intent.exchange_id
            sid = intent.security_id
            label = self._exchange_id_to_label.get(eid, f"eid={eid}")
            parts = []
            if intent.bid_price is not None and intent.bid_price > 0:
                price_cents = intent.bid_price / CENT
                qty_contracts = intent.bid_size / SIZE_SCALE if intent.bid_size else 0
                notional = price_cents / 100 * qty_contracts
                tick_ok = intent.bid_price % CENT == 0
                parts.append(
                    f"  MAKER BID {label} sid={sid} "
                    f"price={price_cents:.2f}¢ {'OK' if tick_ok else 'SUBTICK!'} "
                    f"qty={qty_contracts:.2f}c notional=${notional:.2f}"
                )
            if intent.take_size is not None and intent.take_size > 0:
                side = "BID" if intent.take_side == Side.BID else "ASK"
                price_cents = (intent.take_limit_price or 0) / CENT
                qty_contracts = intent.take_size / SIZE_SCALE
                notional = price_cents / 100 * qty_contracts
                tick_ok = (intent.take_limit_price or 0) % CENT == 0
                parts.append(
                    f"  TAKER {side} {label} sid={sid} "
                    f"price={price_cents:.2f}¢ {'OK' if tick_ok else 'SUBTICK!'} "
                    f"qty={qty_contracts:.2f}c notional=${notional:.2f}"
                )
            if not parts:
                parts.append(f"  CANCEL {label} sid={sid}")
            for p in parts:
                print(p)

    # ------------------------------------------------------------------
    # Market data entry point
    # ------------------------------------------------------------------

    def on_market_data(self, data: Schema) -> list[Intent]:
        listing = (data.exchange_id, data.security_id)
        if listing not in self._tracked_listings:
            return []

        ts = data.event_timestamp
        self._update_book(listing, data, ts)

        intents = []
        for ps in self._pairing_states:
            if ps.phase == Phase.SCANNING:
                intents.extend(self._on_scanning(ps, ts))
            elif ps.phase == Phase.ENTERING:
                intents.extend(self._on_entering(ps, ts))
            elif ps.phase == Phase.PARTIAL_FILL:
                intents.extend(self._on_partial_fill(ps, ts))
            elif ps.phase == Phase.UNWINDING:
                intents.extend(self._on_closing(ps, ts))
        return intents

    # ------------------------------------------------------------------
    # Execution report entry point (fill callback — dual-observer pattern)
    # ------------------------------------------------------------------

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        listing = (report.exchange_id, report.security_id)
        if report.exec_type in (ExecType.FILL, ExecType.PARTIAL_FILL):
            for ps in self._listing_to_ps.get(listing, []):
                if ps.phase not in (Phase.ENTERING, Phase.PARTIAL_FILL):
                    continue
                for leg in ps.legs:
                    if leg.listing != listing:
                        continue
                    leg.record_fill(report.fill_price, report.filled_qty, report.fee)
                    if self._debug:
                        label = self._exchange_id_to_label.get(listing[0], f"eid={listing[0]}")
                        price_c = report.fill_price / (PRICE_SCALE // 100)
                        qty_c = report.filled_qty / SIZE_SCALE
                        filled_legs = [(self._exchange_id_to_label.get(lg.listing[0],"?"), lg.filled_qty/SIZE_SCALE) for lg in ps.legs if lg.filled_qty > 0]
                        print(
                            f"[DEBUG] {"PARTIAL-" if report.exec_type == ExecType.PARTIAL_FILL else ""}FILL {label} sid={listing[1]} "
                            f"price={price_c:.2f}¢ qty={qty_c:.2f}c "
                            f"pairing={ps.pairing.label} filled_legs={filled_legs}"
                        )
                    return self._check_fill_transitions(ps, report.timestamp_event)
        elif report.exec_type in (ExecType.REJECT, ExecType.EXPIRE):
            rejected_label = self._exchange_id_to_label.get(listing[0], "")
            for ps in self._listing_to_ps.get(listing, []):
                if ps.phase not in (Phase.ENTERING, Phase.PARTIAL_FILL):
                    continue
                if report.reject_reason == RejectReason.POST_ONLY_WOULD_CROSS:
                    rejected_leg = next((lg for lg in ps.legs if lg.listing == listing), None)
                    if rejected_leg is not None:
                        self._on_post_only_reject(ps, rejected_leg)
                    if self._debug:
                        print(
                            f"[DEBUG] POST_ONLY_WOULD_CROSS {rejected_label} sid={listing[1]} "
                            f"pairing={ps.pairing.label} — chase may retry this price"
                        )
                    return []
                any_other_filled = any(
                    lg.filled_qty > 0 for lg in ps.legs if lg.listing != listing
                )
                if self._debug:
                    print(
                        f"[DEBUG] {"REJECT" if report.exec_type == ExecType.REJECT else "EXPIRE"} {rejected_label} sid={listing[1]} "
                        f"pairing={ps.pairing.label} any_other_filled={any_other_filled}"
                    )
                if any_other_filled:
                    ps.phase = Phase.UNWINDING
                    return self._close_all_positions(ps, report.timestamp_event)
                else:
                    cancel_intents = [
                        Intent(exchange_id=lg.listing[0], security_id=lg.listing[1])
                        for lg in ps.legs if lg.listing != listing
                    ]
                    ps.phase = Phase.SCANNING
                    ps.last_cancel_ts = report.timestamp_event
                    self._reset(ps)
                    return cancel_intents
        elif report.exec_type == ExecType.CANCEL:
            canceled_label = self._exchange_id_to_label.get(listing[0], "")
            for ps in self._listing_to_ps.get(listing, []):
                if ps.phase not in (Phase.ENTERING, Phase.PARTIAL_FILL):
                    continue

                canceled_leg = next((lg for lg in ps.legs if lg.listing == listing), None)

                if canceled_leg and canceled_leg.filled_qty > 0:
                    new_target = canceled_leg.filled_qty
                    if self._debug:
                        print(
                            f"[DEBUG] TAKER_PARTIAL_CANCEL {canceled_label} sid={listing[1]} "
                            f"pairing={ps.pairing.label} filled={new_target/SIZE_SCALE:.2f}c "
                            f"target was {canceled_leg.target_qty/SIZE_SCALE:.2f}c"
                        )
                    intents = self._apply_cancel_retarget(
                        ps, canceled_leg, new_target, report.timestamp_event
                    )
                    if intents is not None:
                        return intents
                    return self._check_fill_transitions(ps, report.timestamp_event)

                any_other_filled = any(
                    lg.filled_qty > 0 for lg in ps.legs if lg.listing != listing
                )
                if self._debug:
                    print(
                        f"[DEBUG] CANCEL_NOFILL {canceled_label} sid={listing[1]} "
                        f"pairing={ps.pairing.label} any_other_filled={any_other_filled}"
                    )
                if any_other_filled:
                    ps.phase = Phase.UNWINDING
                    return self._close_all_positions(ps, report.timestamp_event)
                else:
                    cancel_intents = [
                        Intent(exchange_id=lg.listing[0], security_id=lg.listing[1])
                        for lg in ps.legs if lg.listing != listing
                    ]
                    ps.phase = Phase.SCANNING
                    ps.last_cancel_ts = report.timestamp_event
                    self._reset(ps)
                    return cancel_intents
        return []

    # ------------------------------------------------------------------
    # Book management
    # ------------------------------------------------------------------

    def _update_book(self, listing: tuple[int, int], data: Schema, ts: int) -> None:
        book = self._books[listing]
        bids: list[BookLevel] = []
        asks: list[BookLevel] = []
        for i in range(10):
            p, s = data.bid_price(i), data.bid_size(i)
            if p <= 0 or s <= 0:
                break
            bids.append(BookLevel(p, s))
        for i in range(10):
            p, s = data.ask_price(i), data.ask_size(i)
            if p <= 0 or s <= 0:
                break
            asks.append(BookLevel(p, s))
        prev_bid_px = book.best_bid()

        book.bids = bids
        book.asks = asks
        book.last_update_ts = ts

        self._now_ns = ts

        # Only prints at or through the prior best bid fill a resting bid. Size decreases
        # in the book can't be used: they are mostly cancellations, which never fill us,
        # and measured that way every leg looked certain to fill.
        if (data.action == Action.TRADE.value and data.price > 0 and data.size > 0
                and 0 < data.price <= prev_bid_px):
            vol, last = self._sell_flow.get(listing, (0.0, ts))
            decay = math.exp(-(ts - last) / self._flow_window_ns)
            self._sell_flow[listing] = (vol * decay + data.size / SIZE_SCALE, ts)

        self._cost_model.on_book_update(listing, book)
        self._price_model.on_book_update(listing, book)

    def _sell_rate(self, listing: tuple[int, int]) -> float:
        """Contracts/sec sold into the bid, exponentially weighted over flow_window_ns.

        Dividing decayed volume by the window (rather than by elapsed time) understates
        the rate during warm-up, which errs toward not entering.
        """
        vol, last = self._sell_flow.get(listing, (0.0, 0))
        if vol <= 0:
            return 0.0
        window_s = self._flow_window_ns / 1e9
        return vol * math.exp(-(self._now_ns - last) / self._flow_window_ns) / window_s

    def _fill_prob(self, listing: tuple[int, int], horizon_ns: int) -> float:
        """P(a bid joining the back of the queue fills within the horizon).

        Time to work through Q contracts at arrival rate lambda is ~Q/lambda; modelling
        the first passage as exponential gives 1 - exp(-lambda*T/Q). Standard
        queue-position valuation (Moallemi/Cont): order value is fill probability times
        the spread premium, so an entry must be discounted by the chance it never fills.

        The strategy previously assumed every resting bid fills, which is why pricing
        maker legs at the bid made 94% of ticks look like an opportunity.
        """
        book = self._books.get(listing)
        if book is None or not book.bids:
            return 0.0
        queue = book.bids[0].size / SIZE_SCALE
        if queue <= 0:
            return 1.0
        rate = self._sell_rate(listing)
        if rate <= 0:
            return 0.0
        return 1.0 - math.exp(-rate * (horizon_ns / 1e9) / queue)

    def _is_stale(self, listing: tuple[int, int], ts: int) -> bool:
        book = self._books[listing]
        return book.last_update_ts == 0 or (ts - book.last_update_ts) > self._max_staleness_ns

    # ------------------------------------------------------------------
    # Fee helpers
    # ------------------------------------------------------------------

    def _maker_fee(self, listing: tuple[int, int], price_scaled: int) -> float:
        label = self._exchange_id_to_label.get(listing[0], "")
        rate = self._maker_fee_rates.get(label, 0.0)
        p = price_scaled / PRICE_SCALE
        return rate * p * (1.0 - p)

    def _taker_fee(self, listing: tuple[int, int], price_scaled: int) -> float:
        label = self._exchange_id_to_label.get(listing[0], "")
        rate = self._taker_fee_rates.get(label, 0.0)
        p = price_scaled / PRICE_SCALE
        return rate * p * (1.0 - p)

    # ------------------------------------------------------------------
    # Edge detection (walks ask books — entry uses asks only)
    # ------------------------------------------------------------------

    def _compute_pairing_edge(self, pairing: Pairing, ts: int = 0) -> tuple[float, int]:
        """Walk ask books to find max profitable qty and avg edge per contract.

        Returns (avg_edge_dollars, qty_scaled). qty=0 means no opportunity.
        Entry asks only — bid books are not walked for entry.
        """
        legs = pairing.legs
        books = [self._books[lg] for lg in legs]

        # Staleness and min contract price gates
        if ts > 0:
            if any(self._is_stale(lg, ts) for lg in legs):
                return 0.0, 0
            if any(
                books[i].best_ask() < self._min_contract_price_scaled
                or books[i].best_ask() > PRICE_SCALE - self._min_contract_price_scaled
                for i in range(len(legs))
            ):
                return 0.0, 0

        # Need valid asks on every leg
        for book in books:
            if not book.asks:
                return 0.0, 0

        # Cross-exchange consistency: implied probabilities (ask prices) should sum to ~1.
        # Only an ask_sum ABOVE 1 is evidence of staleness. An ask_sum below 1 is the
        # arbitrage itself — a two-sided abs() guard rejected the widest opportunities
        # (ask_sum 0.949 is a 5.1c risk-free edge, 5x min_edge).
        if self._max_price_divergence > 0:
            ask_sum = sum(books[i].best_ask() / PRICE_SCALE for i in range(len(legs)))
            if ask_sum - 1.0 > self._max_price_divergence:
                return 0.0, 0

        # Price each leg at the price it will ACTUALLY pay given its execution mode.
        # A taker lifts the ask; a maker rests at the bid and is filled by incoming flow.
        # Pricing maker legs off the ask understates their edge by the full spread, which
        # on this book is ~3.6c against a ~2c edge -- it gated every maker entry on a
        # taker opportunity that the strategy then did not take.
        is_taker = [
            self._exchange_id_to_label.get(lg[0], "") in self._taker_labels for lg in legs
        ]

        def _leg_price(i: int, level: int) -> int | None:
            book = books[i]
            if is_taker[i]:
                return book.asks[level].price if level < len(book.asks) else None
            return book.bids[level].price if level < len(book.bids) else None

        def _leg_size(i: int, level: int) -> int:
            book = books[i]
            side = book.asks if is_taker[i] else book.bids
            return side[level].size if level < len(side) else 0

        for i in range(len(legs)):
            if _leg_price(i, 0) is None:
                return 0.0, 0

        qty = 0
        total_edge = 0.0
        level_indices = [0] * len(legs)
        remaining = [_leg_size(i, 0) for i in range(len(legs))]
        max_qty = self._max_position

        while qty < max_qty:
            current_asks = []
            for i, book in enumerate(books):
                px = _leg_price(i, level_indices[i])
                if px is None:
                    return total_edge / max(qty, 1), qty
                current_asks.append(px)

            # Fee-aware marginal edge at this depth
            total_cost = sum(p / PRICE_SCALE for p in current_asks)
            total_fee = sum(
                self._cost_model.total_cost(
                    legs[i], current_asks[i],
                    qty,
                    maker=self._exchange_id_to_label.get(legs[i][0], "") not in self._taker_labels,
                    other_listing=legs[1 - i] if len(legs) == 2 else None,
                )
                for i in range(len(legs))
            )
            marginal_edge = 1.0 - total_cost - total_fee

            # Discount by the chance the whole set actually fills. A maker leg only
            # earns its edge if it is worked; an unfilled leg leaves the filled ones
            # naked, and unwinding those costs the spread. Taker legs fill on arrival
            # so they carry p=1.
            if self._fill_risk_horizon_ns > 0:
                p_all = 1.0
                for i in range(len(legs)):
                    if not is_taker[i]:
                        p_all *= self._fill_prob(legs[i], self._fill_risk_horizon_ns)
                if p_all <= 0.0:
                    return 0.0, 0
                stuck = 0.0
                for i in range(len(legs)):
                    b = books[i]
                    if b.best_bid() > 0 and b.best_ask() > 0:
                        stuck += (b.best_ask() - b.best_bid()) / PRICE_SCALE
                stuck = (stuck / max(len(legs), 1)) * self._unwind_spread_mult
                marginal_edge = p_all * marginal_edge - (1.0 - p_all) * stuck

            if marginal_edge <= 0:
                break

            chunk = min(remaining)
            chunk = min(chunk, max_qty - qty)
            if chunk <= 0:
                break

            qty += chunk
            total_edge += marginal_edge * chunk

            for i in range(len(legs)):
                remaining[i] -= chunk
                if remaining[i] <= 0:
                    # Taker orders only fill at level 0 (limit to best ask).
                    # Don't walk deeper — the walk would return qty > actual fill.
                    if is_taker[i]:
                        return total_edge / max(qty, 1), qty
                    next_idx = level_indices[i] + 1
                    nxt = _leg_size(i, next_idx)
                    if nxt > 0:
                        level_indices[i] = next_idx
                        remaining[i] = nxt
                    else:
                        return total_edge / max(qty, 1), qty

        return (total_edge / max(qty, 1), qty) if qty > 0 else (0.0, 0)

    def _compute_maker_bid_prices(self, pairing: Pairing, qty: int) -> dict[tuple[int, int], int] | None:
        """Compute entry prices for each leg, respecting the budget constraint.

        PM taker legs use ask price + taker fee; Kalshi legs use best_bid + maker fee.
        Budget: sum(entry_i/PRICE_SCALE + fee_i) < 1.0 - min_edge
        """
        prices: dict[tuple[int, int], int] = {}
        for leg in pairing.legs:
            book = self._books[leg]
            is_taker = self._exchange_id_to_label.get(leg[0], "") in self._taker_labels
            if is_taker:
                ask = book.best_ask()
                if ask <= 0:
                    return None
                prices[leg] = ask
            else:
                if not book.bids or not book.asks:
                    return None
                best_bid = book.best_bid()
                best_ask = book.best_ask()
                if best_bid <= 0 or best_ask <= 0:
                    return None
                prices[leg] = self._price_model.compute_bid(leg, best_bid, best_ask)

        # Verify joint budget constraint
        total = sum(p / PRICE_SCALE for p in prices.values())
        total_fees = 0.0
        for leg, p in prices.items():
            if self._exchange_id_to_label.get(leg[0], "") in self._taker_labels:
                total_fees += self._taker_fee(leg, p)
            else:
                total_fees += self._maker_fee(leg, p)
        if total + total_fees >= 1.0 - self._min_edge:
            return None

        return prices

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _min_price_for_size(self, listing: tuple[int, int], size: int) -> int:
        mn, _, _ = self._listing_specs[listing]
        if mn <= 0 or size <= 0:
            return 0
        return (mn + size - 1) // size

    def _passes_notional(self, listing: tuple[int, int], price: int, size: int) -> bool:
        mn, _, _ = self._listing_specs[listing]
        if mn <= 0:
            return True
        # price * size, not price >= mn // size: the floored form accepted prices whose
        # actual notional fell short (33 * 3 == 99 < 100 while 100 // 3 == 33).
        return size > 0 and price * size >= mn

    def _align_lot(self, listing: tuple[int, int], size: int) -> int:
        _, lot, min_size = self._listing_specs[listing]
        if lot > 0 and size % lot != 0:
            size = (size // lot) * lot
        # The venue rejects an order under its minimum, so a sub-minimum size is as untradable as a sub-lot one.
        return size if size >= min_size else 0

    def _claimed_base(self, listing: tuple[int, int]) -> int:
        """Position on `listing` already accounted for by any pairing that holds it.

        With N>=3 outcomes a listing belongs to several pairings. Comparing against a
        single pairing's base_qty made a pairing holding nothing treat a sibling's
        parked position as an orphan and try to liquidate it.
        """
        return sum(other.base_qty for other in self._listing_to_ps.get(listing, []))

    def _listing_is_busy(self, pairing: Pairing) -> bool:
        """True if another pairing sharing one of our listings is mid-cycle.

        Note this SERIALIZES pairings that share a listing. At N=2 the two pairings are
        disjoint so they run concurrently; at N>=3 each listing appears in several
        pairings, so in practice only one is active at a time. That is deliberate — the
        fill router cannot attribute a fill on a shared listing to a specific pairing —
        but it does mean N>=3 throughput is much lower than the pairing count suggests.
        """
        for lg in pairing.legs:
            for other in self._listing_to_ps.get(lg, []):
                if other.pairing is pairing:
                    continue
                if other.phase != Phase.SCANNING:
                    return True
        return False

    def _on_scanning(self, ps: PairingState, ts: int) -> list[Intent]:
        if self._listing_is_busy(ps.pairing):
            return []

        for lg in ps.pairing.legs:
            eid, sid = lg
            pos = self.positions.get_position(eid, sid)
            current = pos.net_quantity if pos is not None else 0
            if current > self._claimed_base(lg):
                ps.legs = [LegState(listing=l) for l in ps.pairing.legs]
                ps.phase = Phase.UNWINDING
                ps.last_close_ts = 0
                return self._close_all_positions(ps, ts)

        if ps.last_cancel_ts > 0:
            if ts - ps.last_cancel_ts < _CANCEL_SETTLE_NS:
                return []
            ps.last_cancel_ts = 0

        edge, qty = self._compute_pairing_edge(ps.pairing, ts)
        if edge < self._min_edge or qty <= 0:
            return []

        current_pos = 0
        for lg in ps.pairing.legs:
            eid, sid = lg
            pos = self.positions.get_position(eid, sid)
            if pos is not None:
                current_pos = max(current_pos, pos.net_quantity)
        remaining = self._max_position - current_pos
        if remaining <= 0:
            return []
        target_qty = min(qty, remaining)
        for lg in ps.pairing.legs:
            target_qty = self._align_lot(lg, target_qty)
        if target_qty <= 0:
            return []

        bid_prices = self._compute_maker_bid_prices(ps.pairing, target_qty)
        if bid_prices is None:
            return []

        for lg in ps.pairing.legs:
            is_taker = self._exchange_id_to_label.get(lg[0], "") in self._taker_labels
            price = self._books[lg].best_ask() if is_taker else bid_prices[lg]
            if not self._passes_notional(lg, price, target_qty):
                return []

        ps.legs = [LegState(listing=lg, target_qty=target_qty) for lg in ps.pairing.legs]
        ps.entry_ts = ts
        ps.phase = Phase.ENTERING
        if self._lead_with_illiquid:
            lead = self._pick_lead(ps.pairing, bid_prices)
            for leg in ps.legs:
                leg.posted = leg.listing == lead

        if self._debug:
            print(
                f"[DEBUG] ENTER pairing={ps.pairing.label} "
                f"edge={edge*100:.3f}¢ qty={target_qty/SIZE_SCALE:.1f}c"
            )

        if self._metrics_buf is not None:
            row = self._metrics_buf.appendRow()
            self._metrics_buf.setLong(row, self._m_ts, ts)
            self._metrics_buf.setInt(row, self._m_phase, int(Phase.ENTERING))
            self._metrics_buf.setInt(row, self._m_pairing, ps.pairing.index)
            self._metrics_buf.setDouble(row, self._m_edge, edge)
            self._metrics_buf.setLong(row, self._m_qty, target_qty)

        taker_intents = []
        maker_intents = []
        for leg in ps.legs:
            if not leg.posted:
                continue
            eid = leg.listing[0]
            if self._exchange_id_to_label.get(eid, "") in self._taker_labels:
                book = self._books[leg.listing]
                limit_price = book.best_ask()
                taker_intents.append(Intent(
                    exchange_id=eid,
                    security_id=leg.listing[1],
                    take_side=Side.BID,
                    take_size=leg.target_qty,
                    take_order_type=OrderType.LIMIT,
                    take_limit_price=limit_price,
                ))
            else:
                # Quantize here too: join_best_bid lands on-tick, but the other price
                # models can return best_ask - 1, which is one raw scale unit.
                bid = self._quantize_price(leg.listing, bid_prices[leg.listing])
                leg.entry_bid = bid
                # Seed the chase dedupe, otherwise its first tick re-sends this exact
                # price and resets queue position on the order carrying most of the size.
                leg.last_chase_bid = bid
                maker_intents.append(Intent(
                    exchange_id=eid,
                    security_id=leg.listing[1],
                    bid_price=bid,
                    bid_size=leg.target_qty,
                    post_only=True,
                ))
        result = taker_intents + maker_intents
        self._log_intents(f"entry {ps.pairing.label}", result, ts)
        return result

    def _quantize_price(self, listing: tuple[int, int], price: int) -> int:
        tick = self._effective_tick(listing)
        return (price // tick) * tick

    def _effective_tick(self, listing: tuple[int, int]) -> int:
        """Tick to quantize to, floored at one cent.

        The registry reports $0.001 for every Kalshi listing, but the recorded books
        are 100% whole-cent across 1.5M observations. Trusting it produced orders at
        prices the exchange cannot accept (0.431, 0.824) which the backtest happily
        filled. Floor until the security master is corrected.
        """
        return max(self._tick_sizes.get(listing, _MIN_TICK), _MIN_TICK)

    def _on_post_only_reject(self, ps: PairingState, leg: LegState) -> None:
        """Clear the chase dedupe for a price the exchange refused.

        The chase skips a target equal to `last_chase_bid`. Leaving it set after a
        post-only reject suppressed the retry the handler's own comment promises,
        permanently, for that price.
        """
        leg.last_chase_bid = 0

    def _entry_expired(self, ps: PairingState, ts: int) -> bool:
        """ENTERING has no natural exit if nothing fills and nothing is rejected again.

        entry_ts was only ever read for the 500ms taker-settle guard, so a pairing whose
        orders were all rejected sat in ENTERING with no live orders indefinitely.
        """
        if ps.entry_ts <= 0:
            return False
        return ts - ps.entry_ts > self._imbalance_timeout_ns

    def _hedged_base(self, ps: PairingState) -> int:
        """The quantity held on EVERY leg — i.e. the part that is actually hedged.

        Taking max() here hid leg imbalance: with legs at 50 and 100 the base became
        100, `all_at_base` was satisfied, and the pairing returned to SCANNING carrying
        50 contracts of naked directional exposure. The common quantity is the only
        part that represents complete sets.
        """
        held = []
        for lg in ps.legs:
            eid, sid = lg.listing
            pos = self.positions.get_position(eid, sid)
            held.append(pos.net_quantity if pos is not None else 0)
        return min(held) if held else 0

    def _apply_cancel_retarget(
        self, ps: PairingState, canceled_leg: LegState, new_target: int, ts: int
    ) -> list[Intent] | None:
        """Retarget every leg down to what the canceled leg actually filled.

        base_qty must stay the absolute hedged position. Setting it to `new_target`
        (this entry's increment) meant a 50-lot cancel against an accumulated 2000
        position reset the base to 50, after which _close_all_positions computed
        2050 - 50 and tried to liquidate the entire book position at a 1c ask.
        Returns the intents to emit, or None to fall through to the caller.
        """
        for lg in ps.legs:
            lg.target_qty = new_target

        if not all(lg.filled_qty >= new_target for lg in ps.legs):
            return None

        ps.base_qty = self._hedged_base(ps)
        if any(lg.filled_qty > new_target for lg in ps.legs):
            ps.phase = Phase.UNWINDING
            return self._close_all_positions(ps, ts)

        ps.phase = Phase.SCANNING
        self._reset(ps)
        return []

    def _compute_chase_ceiling(self, ps: PairingState) -> int:
        """Highest price one unfilled leg may bid and still leave the pairing profitable.

        Budget is $1 per complete set, minus what the filled legs cost, minus the fees
        already paid on them, minus the fee this leg will pay, minus min_edge. Split
        across the unfilled legs so N of them bidding at the ceiling cannot together
        exceed the remaining budget — with 2 unfilled siblings the previous version
        allowed a total above $1, a guaranteed loss on a supposed arb.
        """
        filled_cost_scaled = 0
        filled_qty_total = 0
        fees_paid = 0.0
        unfilled = 0
        for lg in ps.legs:
            if lg.filled_qty > 0:
                filled_cost_scaled += lg.fill_cost // lg.filled_qty
                filled_qty_total = max(filled_qty_total, lg.filled_qty)
                fees_paid += lg.entry_fees
            if not lg.is_filled:
                unfilled += 1

        budget = PRICE_SCALE - filled_cost_scaled

        # Fees already paid, expressed per contract in scaled price units.
        if filled_qty_total > 0:
            fee_per_contract = fees_paid / (filled_qty_total / SIZE_SCALE)
            budget -= int(fee_per_contract * PRICE_SCALE)

        budget -= int(self._min_edge * PRICE_SCALE)

        if unfilled > 1:
            budget //= unfilled

        # The fee this leg will pay is itself a function of the price, so charge the
        # worst case (rate/4, the maximum of rate*p*(1-p)) rather than solving it.
        worst_maker_rate = max(self._maker_fee_rates.values(), default=0.0)
        budget -= int(worst_maker_rate / 4.0 * PRICE_SCALE)

        return budget

    def _on_entering(self, ps: PairingState, ts: int) -> list[Intent]:
        return self._check_fill_transitions(ps, ts)

    def _check_fill_transitions(self, ps: PairingState, ts: int) -> list[Intent]:
        all_filled = all(lg.is_filled for lg in ps.legs)
        any_filled = any(lg.filled_qty > 0 for lg in ps.legs)
        some_filled = any_filled and not all_filled

        if all_filled:
            base = self._hedged_base(ps)
            ps.base_qty = base
            held = []
            for lg in ps.legs:
                eid, sid = lg.listing
                pos = self.positions.get_position(eid, sid)
                held.append(pos.net_quantity if pos is not None else 0)
            if any(q > base for q in held):
                # Legs are imbalanced; the excess above the common quantity is naked.
                if self._debug:
                    print(f"[DEBUG] ALL_FILLED_IMBALANCED pairing={ps.pairing.label} "
                          f"held={[q/SIZE_SCALE for q in held]} base={base/SIZE_SCALE:.1f}c → UNWINDING")
                ps.phase = Phase.UNWINDING
                self._log_phase(ps, ts)
                return self._close_all_positions(ps, ts)
            if self._debug:
                print(f"[DEBUG] ALL_FILLED pairing={ps.pairing.label} → SCANNING base_qty={base/SIZE_SCALE:.1f}c")
            ps.phase = Phase.SCANNING
            self._reset(ps)
            return []

        if some_filled and ps.phase == Phase.ENTERING:
            filled = [(self._exchange_id_to_label.get(lg.listing[0],""), lg.filled_qty/SIZE_SCALE) for lg in ps.legs if lg.filled_qty > 0]
            if self._debug:
                print(f"[DEBUG] PARTIAL pairing={ps.pairing.label} → PARTIAL_FILL filled={filled}")
            ps.phase = Phase.PARTIAL_FILL
            ps.partial_fill_since = ts
            return []

        if some_filled and ps.phase == Phase.PARTIAL_FILL:
            elapsed = ts - ps.partial_fill_since

            completion = self._taker_completion(ps, ts)
            if completion:
                return completion
            deferred = self._post_deferred_legs(ps, ts)
            if deferred:
                return deferred

            chase_intents = []
            for lg in (ps.legs if self._enable_chase else []):
                if lg.is_filled:
                    continue
                eid = lg.listing[0]
                if self._exchange_id_to_label.get(eid, "") in self._taker_labels:
                    continue

                book = self._books[lg.listing]
                best_bid = book.best_bid()
                best_ask = book.best_ask()
                ceiling = self._compute_chase_ceiling(ps)

                if ceiling <= 0 or best_bid <= 0 or best_ask <= 0:
                    continue

                if ceiling <= best_bid:
                    # No price at or below the ceiling is even competitive with the
                    # current bid, so there is nothing profitable to chase. The old code
                    # walked from best_bid toward best_ask here, ignoring the ceiling and
                    # buying at a guaranteed loss; leave the order resting and let the
                    # imbalance timeout unwind if it never fills.
                    if self._debug and elapsed < 1_000_000_000:
                        print(
                            f"[DEBUG] CHASE_ABANDON pairing={ps.pairing.label} "
                            f"ceiling={ceiling/(PRICE_SCALE//100):.1f}¢ <= "
                            f"best_bid={best_bid/(PRICE_SCALE//100):.1f}¢ — no profitable price"
                        )
                    continue
                elif elapsed <= self._maker_patience_ns:
                    floor = lg.entry_bid
                    urgency = elapsed / self._maker_patience_ns if self._maker_patience_ns > 0 else 1.0
                    target_price = floor + int(urgency * (ceiling - floor))
                    phase = 1
                else:
                    # Phase 2 used to walk from the ceiling toward best_ask, i.e. above
                    # the price at which the pairing is still profitable. Cap at the
                    # ceiling: completing a leg at a loss is the timeout's job, not the
                    # chase's.
                    target_price = min(ceiling, best_ask)
                    phase = 2

                target_price = self._quantize_price(lg.listing, min(target_price, ceiling))
                if target_price <= 0 or target_price == lg.last_chase_bid:
                    continue

                remaining = self._align_lot(lg.listing, lg.target_qty - lg.filled_qty)
                if remaining <= 0:
                    continue
                if not self._passes_notional(lg.listing, target_price, remaining):
                    continue
                lg.last_chase_bid = target_price
                chase_intents.append(Intent(
                    exchange_id=eid,
                    security_id=lg.listing[1],
                    bid_price=target_price,
                    bid_size=remaining,
                    post_only=True,
                ))
                if self._debug:
                    print(
                        f"[DEBUG] CHASE pairing={ps.pairing.label} phase={phase} "
                        f"bid={target_price/(PRICE_SCALE//100):.1f}¢ "
                        f"ceiling={ceiling/(PRICE_SCALE//100):.1f}¢ "
                        f"ask={best_ask/(PRICE_SCALE//100):.1f}¢ "
                        f"elapsed={elapsed/1e9:.1f}s"
                    )

            # Check the timeout BEFORE returning chase intents. The phase-2 target tracks
            # best_ask, so in a moving book a new intent is emitted every tick and the
            # timeout was never reached — imbalance_timeout_ns was effectively disabled
            # exactly when it mattered.
            if self._check_imbalance_timeout(ps, ts):
                elapsed_s = elapsed / 1e9
                if self._debug:
                    print(f"[DEBUG] IMBALANCE_TIMEOUT pairing={ps.pairing.label} elapsed={elapsed_s:.1f}s → UNWINDING")
                ps.phase = Phase.UNWINDING
                self._log_phase(ps, ts)
                return self._close_all_positions(ps, ts)

            if chase_intents:
                return chase_intents
            return []

        # Guard: taker orders take ~300ms to settle. Don't cancel maker legs
        # or reset state until the taker leg has had a chance to fill or be rejected —
        # otherwise taker fills after reset creating untracked naked directional exposure.
        if (self._taker_labels
                and ps.entry_ts > 0
                and (ts - ps.entry_ts) < _PM_TAKER_SETTLE_NS
                and any(
                    self._exchange_id_to_label.get(lg.listing[0], "") in self._taker_labels
                    and not lg.is_filled
                    for lg in ps.legs
                )):
            return []

        unfilled_legs = [lg for lg in ps.legs if not lg.is_filled]
        if unfilled_legs:
            bid_prices = self._compute_maker_bid_prices(ps.pairing, unfilled_legs[0].target_qty)
            if bid_prices is None:
                if self._debug:
                    labels = [self._exchange_id_to_label.get(lg.listing[0], "?") for lg in unfilled_legs]
                    print(f"[DEBUG] BUDGET_FAIL pairing={ps.pairing.label} unfilled={labels} → SCANNING")
                intents = []
                for leg in unfilled_legs:
                    intents.append(Intent(exchange_id=leg.listing[0], security_id=leg.listing[1]))
                if any_filled:
                    ps.phase = Phase.UNWINDING
                    intents.extend(self._close_all_positions(ps, ts))
                else:
                    ps.phase = Phase.SCANNING
                    ps.last_cancel_ts = ts
                    self._reset(ps)
                return intents

        if ps.phase == Phase.ENTERING and self._entry_expired(ps, ts):
            if self._debug:
                print(f"[DEBUG] ENTRY_EXPIRED pairing={ps.pairing.label} "
                      f"elapsed={(ts - ps.entry_ts)/1e9:.1f}s any_filled={any_filled}")
            intents = [
                Intent(exchange_id=lg.listing[0], security_id=lg.listing[1])
                for lg in ps.legs if not lg.is_filled
            ]
            if any_filled:
                ps.phase = Phase.UNWINDING
                intents.extend(self._close_all_positions(ps, ts))
            else:
                ps.phase = Phase.SCANNING
                ps.last_cancel_ts = ts
                self._reset(ps)
            return intents

        return []

    def _taker_completion(self, ps: PairingState, ts: int) -> list[Intent]:
        """Cross the spread on the unfilled legs while the set is still profitable.

        Waiting passively for the final legs is where the exposure comes from: the
        filled legs are paid for, and the position is naked until the rest fill (one
        wait in the soccer backtest lasted 299s at -$1,804). If buying the remainder at
        the asks, plus taker fees, still completes the set below $1 - edge, locking a
        smaller edge now beats holding the full one at risk.

        Equal size is taken on every unfilled leg so the takes cannot themselves
        unbalance the set; whatever the touch can't cover rests at a bid kept strictly
        below the ask. The settle window stops a re-send before the takes report.
        """
        if self._taker_complete_edge is None:
            return []
        filled = [lg for lg in ps.legs if lg.is_filled]
        unfilled = [lg for lg in ps.legs if not lg.is_filled]
        if not filled or not unfilled:
            return []
        if any(not lg.posted for lg in unfilled) and any(not lg.is_filled for lg in ps.legs if lg.posted):
            return []
        if ps.completion_ts > 0 and ts - ps.completion_ts < _PM_TAKER_SETTLE_NS:
            return []

        set_cost = 0.0
        for lg in filled:
            set_cost += lg.fill_cost / lg.filled_qty / PRICE_SCALE
            set_cost += lg.entry_fees / (lg.filled_qty / SIZE_SCALE)
        take = None
        for lg in unfilled:
            book = self._books[lg.listing]
            ask = book.best_ask()
            if ask <= 0 or not book.asks:
                return []
            set_cost += ask / PRICE_SCALE + self._taker_fee(lg.listing, ask)
            remaining = self._align_lot(lg.listing, lg.target_qty - lg.filled_qty)
            avail = min(remaining, book.asks[0].size)
            take = avail if take is None else min(take, avail)
        if set_cost > 1.0 - self._taker_complete_edge:
            return []
        for lg in unfilled:
            take = self._align_lot(lg.listing, take)
        if take <= 0:
            return []

        intents = []
        for lg in unfilled:
            book = self._books[lg.listing]
            ask = book.best_ask()
            if not self._passes_notional(lg.listing, ask, take):
                return []
            rest = self._align_lot(lg.listing, self._align_lot(lg.listing, lg.target_qty - lg.filled_qty) - take)
            # Once the ask has fallen to or through our bid, resting there would cross
            # and be post-only rejected, and a reject with filled siblings unwinds the
            # whole pairing.
            ceiling = self._quantize_price(lg.listing, ask - self._effective_tick(lg.listing))
            if lg.entry_bid <= 0:
                lg.entry_bid = self._quantize_price(
                    lg.listing, self._price_model.compute_bid(lg.listing, book.best_bid(), ask))
            rest_px = min(lg.entry_bid, ceiling)
            if rest_px <= 0:
                rest = 0
            lg.posted = True
            intents.append(Intent(
                exchange_id=lg.listing[0],
                security_id=lg.listing[1],
                bid_price=rest_px if rest > 0 else 0,
                bid_size=max(rest, 0),
                post_only=rest > 0,
                take_side=Side.BID,
                take_size=take,
                take_order_type=OrderType.LIMIT,
                take_limit_price=ask,
            ))
        ps.completion_ts = ts
        if self._debug:
            print(f"[DEBUG] TAKER_COMPLETE pairing={ps.pairing.label} legs={len(unfilled)} "
                  f"take={take/SIZE_SCALE:.0f}c set_cost={set_cost*100:.2f}¢")
        self._log_intents(f"complete {ps.pairing.label}", intents, ts)
        return intents

    def _pick_lead(self, pairing: Pairing, bid_prices: dict[tuple[int, int], int]) -> tuple[int, int]:
        """The leg least likely to fill, ties broken by cheapest.

        Posting it alone first means the slow fill happens while nothing else is held;
        the exposure while the remaining (liquid) legs fill is only what the lead cost,
        and with a long shot that is a few cents a set rather than most of a dollar.
        """
        horizon = self._fill_risk_horizon_ns or 60_000_000_000
        return min(pairing.legs, key=lambda lg: (self._fill_prob(lg, horizon), bid_prices[lg]))

    def _post_deferred_legs(self, ps: PairingState, ts: int) -> list[Intent]:
        """Once every posted leg is filled, rest a bid on the next held-back leg.

        Legs go out one at a time, least fillable first. Posting all the held-back legs
        together after the lead left an illiquid sibling in the second stage (Draw
        stalled beside City in the soccer backtest), so the exposure barely moved; in
        order, the leg everything waits on last is the most liquid one.

        Prices are re-derived from the live book and the whole set must still complete
        below $1 - min_edge given what the filled legs actually cost; if not, nothing is
        posted and the next tick retries, with the imbalance timeout as the backstop.
        """
        held = [lg for lg in ps.legs if not lg.posted]
        if not held:
            return []
        if any(not lg.is_filled for lg in ps.legs if lg.posted):
            return []
        horizon = self._fill_risk_horizon_ns or 60_000_000_000
        nxt = min(held, key=lambda lg: (self._fill_prob(lg.listing, horizon), self._books[lg.listing].best_bid()))

        set_cost = 0.0
        for lg in ps.legs:
            if lg.posted:
                set_cost += lg.fill_cost / lg.filled_qty / PRICE_SCALE
                set_cost += lg.entry_fees / (lg.filled_qty / SIZE_SCALE)
        bids = {}
        for lg in held:
            book = self._books[lg.listing]
            best_bid, best_ask = book.best_bid(), book.best_ask()
            if best_bid <= 0 or best_ask <= 0:
                return []
            bid = self._quantize_price(lg.listing, self._price_model.compute_bid(lg.listing, best_bid, best_ask))
            bids[lg.listing] = bid
            set_cost += bid / PRICE_SCALE + self._maker_fee(lg.listing, bid)
        if set_cost >= 1.0 - self._min_edge:
            return []

        intents = []
        for lg in [nxt]:
            bid = bids[lg.listing]
            if not self._passes_notional(lg.listing, bid, lg.target_qty):
                return []
            lg.entry_bid = bid
            lg.last_chase_bid = bid
            lg.posted = True
            intents.append(Intent(
                exchange_id=lg.listing[0],
                security_id=lg.listing[1],
                bid_price=bid,
                bid_size=lg.target_qty,
                post_only=True,
            ))
        if self._debug:
            print(f"[DEBUG] POST_DEFERRED pairing={ps.pairing.label} held={len(held)} "
                  f"set_cost={set_cost*100:.2f}¢")
        self._log_intents(f"deferred {ps.pairing.label}", intents, ts)
        return intents

    def _on_partial_fill(self, ps: PairingState, ts: int) -> list[Intent]:
        return self._check_fill_transitions(ps, ts)

    def _on_closing(self, ps: PairingState, ts: int) -> list[Intent]:
        # Use net_quantity (settled fills) not effective_quantity (pending sells) to avoid
        # premature re-entry while close orders are still in flight.
        def _net_qty(lg: LegState) -> int:
            pos = self.positions.get_position(lg.listing[0], lg.listing[1])
            return pos.net_quantity if pos is not None else 0

        # Same floor as _close_all_positions. The retry used base_qty alone, which is 0
        # for a fresh entry, so the first close kept the complete sets and the retry 2s
        # later sold them at market.
        keep = max(ps.base_qty, self._hedged_floor(ps))
        all_at_base = all(_net_qty(lg) <= keep for lg in ps.legs)
        if all_at_base:
            if self._debug:
                print(f"[DEBUG] CLOSED pairing={ps.pairing.label} → SCANNING")
            ps.phase = Phase.SCANNING
            self._log_phase(ps, ts)
            ps.close_attempts = 0
            self._reset(ps)
            return []
        if ts - ps.last_close_ts > 2_000_000_000:
            ps.last_close_ts = ts
            ps.close_attempts += 1
            intents = []
            for leg in ps.legs:
                eid, sid = leg.listing
                eff_qty = self.positions.get_effective_quantity(eid, sid)
                qty = eff_qty - keep
                if qty > 0:
                    qty = self._align_lot((eid, sid), qty)
                    if qty > 0:
                        limit_price = max(PRICE_SCALE // 100, self._min_price_for_size((eid, sid), qty))
                        if limit_price <= 99 * PRICE_SCALE // 100:
                            intents.append(Intent(
                                exchange_id=eid, security_id=sid,
                                take_side=Side.ASK,
                                take_size=qty,
                                take_order_type=OrderType.LIMIT,
                                take_limit_price=limit_price,
                            ))
                elif qty < 0:
                    abs_qty = self._align_lot((eid, sid), -qty)
                    if abs_qty > 0:
                        limit_price = max(99 * PRICE_SCALE // 100, self._min_price_for_size((eid, sid), abs_qty))
                        if limit_price <= PRICE_SCALE:
                            intents.append(Intent(
                                exchange_id=eid, security_id=sid,
                                take_side=Side.BID,
                                take_size=abs_qty,
                                take_order_type=OrderType.LIMIT,
                                take_limit_price=limit_price,
                            ))
            self._log_intents(f"closing_retry {ps.pairing.label}", intents, ts)
            if not intents and ps.close_attempts >= _MAX_CLOSE_ATTEMPTS:
                # Nothing closeable: _align_lot floored a sub-lot remainder to zero, or
                # the min-notional floor exceeded the 99c cap. Retrying every 2s forever
                # kept the pairing out of SCANNING permanently; give up and let the
                # residual sit rather than wedging the state machine.
                if self._debug:
                    print(f"[DEBUG] CLOSE_GAVE_UP pairing={ps.pairing.label} "
                          f"attempts={ps.close_attempts} — residual left in place")
                ps.phase = Phase.SCANNING
                ps.base_qty = self._hedged_base(ps)
                ps.close_attempts = 0
                self._reset(ps)
                return []
            return intents
        return []

    # ------------------------------------------------------------------
    # Closing / unwind helpers
    # ------------------------------------------------------------------

    def _hedged_floor(self, ps: PairingState) -> int:
        """Quantity held on EVERY leg — a complete set, worth $1 at resolution.

        Inventory at or below this is hedged and must never be sold: shedding it
        converts a guaranteed payout into a spread loss. Only the excess above it is
        naked. Earlier versions sold against base_qty alone, which happily liquidated
        complete sets.
        """
        held = []
        for lg in ps.legs:
            pos = self.positions.get_position(lg.listing[0], lg.listing[1])
            held.append(pos.net_quantity if pos is not None else 0)
        return min(held) if held else 0

    def _close_all_positions(self, ps: PairingState, ts: int) -> list[Intent]:
        ps.last_close_ts = ts
        # Keep whatever is hedged; only shed naked excess above it.
        keep = max(ps.base_qty, self._hedged_floor(ps))
        # Post offers before crossing. On a zero-maker-fee venue a passive exit earns
        # the spread instead of paying it; the aggressive IOC path cost 14% of notional
        # on a 6000-lot unwind. Escalate only once passive attempts are exhausted.
        passive = ps.close_attempts < self._passive_unwind_attempts
        if self._debug:
            print(f"[DEBUG] CLOSE_ALL pairing={ps.pairing.label} phase={ps.phase.name} "
                  f"keep={keep/SIZE_SCALE:.0f}c passive={passive} attempt={ps.close_attempts}")
        intents = []
        for leg in ps.legs:
            eid, sid = leg.listing
            pos = self.positions.get_position(eid, sid)
            total_qty = pos.net_quantity if pos is not None else 0
            qty = total_qty - keep
            if qty > 0:
                qty = self._align_lot((eid, sid), qty)
                if qty > 0 and passive:
                    book = self._books.get(leg.listing)
                    best_ask = book.best_ask() if book else 0
                    best_bid = book.best_bid() if book else 0
                    if best_ask > 0:
                        # join the offer rather than hitting the bid
                        px = self._quantize_price(leg.listing, best_ask)
                        if px > best_bid > 0 and self._passes_notional(leg.listing, px, qty):
                            intents.append(Intent(
                                exchange_id=eid, security_id=sid,
                                ask_price=px, ask_size=qty,
                            ))
                            continue
                if qty > 0:
                    limit_price = max(PRICE_SCALE // 100, self._min_price_for_size((eid, sid), qty))
                    if limit_price <= 99 * PRICE_SCALE // 100:
                        intents.append(Intent(
                            exchange_id=eid, security_id=sid,
                            take_side=Side.ASK,
                            take_size=qty,
                            take_order_type=OrderType.LIMIT,
                            take_limit_price=limit_price,
                        ))
            elif qty < 0:
                abs_qty = self._align_lot((eid, sid), -qty)
                if abs_qty > 0:
                    limit_price = max(99 * PRICE_SCALE // 100, self._min_price_for_size((eid, sid), abs_qty))
                    if limit_price <= PRICE_SCALE:
                        intents.append(Intent(
                            exchange_id=eid, security_id=sid,
                            take_side=Side.BID,
                            take_size=abs_qty,
                            take_order_type=OrderType.LIMIT,
                            take_limit_price=limit_price,
                        ))
            elif self._exchange_id_to_label.get(eid, "") not in self._taker_labels:
                intents.append(Intent(exchange_id=eid, security_id=sid))
        self._log_intents(f"close_all {ps.pairing.label}", intents)
        return intents

    # ------------------------------------------------------------------
    # Imbalance timeout
    # ------------------------------------------------------------------

    def _check_imbalance_timeout(self, ps: PairingState, ts: int) -> bool:
        if ps.partial_fill_since is None:
            return False
        return (ts - ps.partial_fill_since) > self._imbalance_timeout_ns

    # ------------------------------------------------------------------
    # Adverse selection / markout tracking
    # ------------------------------------------------------------------

    def _log_phase(self, ps: PairingState, ts: int, edge: float = 0.0, qty: int = 0) -> None:
        """Record a row per phase transition.

        The buffer declares a `phase` column but the only writer hardcoded ENTERING, so
        it was a constant across every row and the parquet carried no information about
        chases, timeouts or unwinds — those were print-only and lost to the run.
        """
        if self._metrics_buf is None:
            return
        row = self._metrics_buf.appendRow()
        self._metrics_buf.setLong(row, self._m_ts, ts)
        self._metrics_buf.setInt(row, self._m_phase, int(ps.phase))
        self._metrics_buf.setInt(row, self._m_pairing, ps.pairing.index)
        self._metrics_buf.setDouble(row, self._m_edge, edge)
        self._metrics_buf.setLong(row, self._m_qty, qty)

    def _reset(self, ps: PairingState) -> None:
        ps.legs = []
        ps.partial_fill_since = None
        ps.entry_ts = 0
