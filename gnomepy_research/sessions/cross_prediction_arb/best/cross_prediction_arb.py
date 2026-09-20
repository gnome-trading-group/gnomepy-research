from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple

from gnomepy import ExecutionReport, Intent, OrderType, Scales, Side, Strategy
from gnomepy.java.enums import ExecType
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


# ---------------------------------------------------------------------------
# Cost model (pluggable)
# ---------------------------------------------------------------------------

class CostComponent:
    def cost(self, listing: tuple[int, int], price_scaled: int, qty: int, **kwargs) -> float:
        return 0.0

    def on_book_update(self, listing: tuple[int, int], book: Book) -> None:
        pass


class FeeCost(CostComponent):
    def __init__(self, maker_rates: dict[int, float], taker_rates: dict[int, float]):
        self._maker = maker_rates
        self._taker = taker_rates

    def cost(self, listing, price_scaled, qty, maker=True, **kwargs) -> float:
        eid = listing[0]
        rate = self._maker.get(eid, 0.0) if maker else self._taker.get(eid, 0.0)
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
    def compute_bid(self, listing, best_bid: int, best_ask: int, max_price: int | None = None) -> int:
        raise NotImplementedError

    def on_book_update(self, listing: tuple[int, int], book: Book) -> None:
        pass


class JoinBestBidModel(PriceModel):
    def compute_bid(self, listing, best_bid, best_ask, max_price=None):
        price = best_bid
        if max_price is not None:
            price = min(price, max_price)
        return price


class AggressiveModel(PriceModel):
    def __init__(self, improve_bps: float = 50.0, edge_share: float = 0.5):
        self._improve_bps = improve_bps
        self._edge_share = edge_share

    def compute_bid(self, listing, best_bid, best_ask, max_price=None):
        mid = (best_bid + best_ask) // 2
        improvement = int(mid * self._improve_bps / 10000)
        price = best_bid + improvement
        if max_price is not None:
            price = min(price, int(max_price * self._edge_share + best_bid * (1.0 - self._edge_share)))
        return min(price, best_ask - 1)


class FillProbTargetModel(PriceModel):
    def __init__(self, target_fill_prob: float = 0.7, fill_risk_lambda: float = 10.0):
        self._target = target_fill_prob
        self._lambda = fill_risk_lambda

    def compute_bid(self, listing, best_bid, best_ask, max_price=None):
        if self._target >= 1.0:
            return best_ask - 1
        mid = (best_bid + best_ask) // 2
        if mid <= 0:
            return best_bid
        target_spread_frac = -math.log(self._target) / self._lambda
        price = int(best_ask - target_spread_frac * mid)
        price = max(price, best_bid)
        price = min(price, best_ask - 1)
        if max_price is not None:
            price = min(price, max_price)
        return price


class OptimalEVModel(PriceModel):
    """Scan bid prices to maximize EV = P_fill * edge - (1-P_fill) * stuck_cost."""
    def __init__(self, fill_risk_lambda: float = 30.0, unwind_spread_mult: float = 2.0):
        self._lambda = fill_risk_lambda
        self._unwind_mult = unwind_spread_mult
        self._spreads: dict[tuple[int, int], float] = {}

    def on_book_update(self, listing, book: Book) -> None:
        bid, ask = book.best_bid(), book.best_ask()
        if bid > 0 and ask > 0:
            self._spreads[listing] = (ask - bid) / PRICE_SCALE

    def compute_bid(self, listing, best_bid, best_ask, max_price=None):
        mid = (best_bid + best_ask) // 2
        if mid <= 0 or best_bid >= best_ask:
            return best_bid
        spread = self._spreads.get(listing, (best_ask - best_bid) / PRICE_SCALE)
        stuck_cost = spread * self._unwind_mult
        cap = max_price if max_price is not None else best_ask - 1
        tick = max(PRICE_SCALE // 100, 1)
        best_ev = float("-inf")
        best_price = best_bid
        price = best_bid
        while price <= min(cap, best_ask - 1):
            spread_frac = (best_ask - price) / mid
            p_fill = math.exp(-self._lambda * spread_frac)
            edge = (cap - price) / PRICE_SCALE if max_price is not None else 0.0
            ev = p_fill * edge - (1.0 - p_fill) * stuck_cost
            if ev > best_ev:
                best_ev = ev
                best_price = price
            price += tick
        return best_price


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

class CrossPredictionArb(Strategy):
    def __init__(
        self,
        pm_a_listing_id: int = 222852,
        pm_b_listing_id: int = 222853,
        k_a_listing_id: int = 97203,
        k_b_listing_id: int = 97202,
        max_position: int = 100,
        min_edge_cents: float = 2.0,
        min_contract_price: float = 0.25,
        max_price_divergence_cents: float = 0.0,
        imbalance_timeout_ns: int = 30_000_000_000,
        max_staleness_ns: int = 60_000_000_000,
        processing_time_ns: int = 5_000_000,
        # Fee rates keyed by exchange_id
        maker_fee_rates: dict[int | str, float] | None = None,
        taker_fee_rates: dict[int | str, float] | None = None,
        # Cost model
        fill_risk_lambda: float = 0.0,
        unwind_spread_mult: float = 2.0,
        depth_coverage_mult: float = 0.0,
        use_cst_fill_model: bool = False,
        price_window_cents: int = 2,
        # Price model
        price_model: str = "join_best_bid",
        price_improve_bps: float = 50.0,
        price_edge_share: float = 0.5,
        target_fill_prob: float = 0.7,
        optimal_ev_lambda: float = 30.0,
        # PM taker mode — PM legs cross the ask for immediate fills
        pm_taker_mode: bool = False,
        # K_DUTCH_BOOK (P2) triggers EXEC_REPORT_FOR_UNKNOWN_ORDER OMS/engine mismatch — disable to avoid
        allow_dutch_book: bool = True,
        # Settlement risk (disabled by default — sports resolve before official expiry)
        gamma_T: float = 0.0,
        resolution_time_ns: int = 0,
        # Diagnostics
        track_markouts: bool = False,
        debug: bool = False,
    ):
        registry = RegistryClient()

        # Default fee rates matching backtest profiles
        # exchange_id 4 = Polymarket, 5 = Kalshi
        if maker_fee_rates is None:
            maker_fee_rates = {4: 0.0, 5: 0.0175}
        if taker_fee_rates is None:
            taker_fee_rates = {4: 0.07, 5: 0.07}
        self._maker_fee_rates = {int(k): v for k, v in maker_fee_rates.items()}
        self._taker_fee_rates = {int(k): v for k, v in taker_fee_rates.items()}

        self._max_position = max_position * SIZE_SCALE
        self._min_edge = min_edge_cents / 100.0
        self._min_contract_price_scaled = int(min_contract_price * PRICE_SCALE)
        self._max_price_divergence = max_price_divergence_cents / 100.0
        self._imbalance_timeout_ns = imbalance_timeout_ns
        self._max_staleness_ns = max_staleness_ns
        self._processing_time_ns = processing_time_ns
        self._gamma_T = gamma_T
        self._resolution_time_ns = resolution_time_ns
        self._track_markouts = track_markouts
        self._debug = debug

        # Resolve listing_ids → (exchange_id, security_id)
        def resolve(listing_id: int) -> tuple[int, int]:
            results = registry.get_listing(listing_id=listing_id)
            if not results:
                raise ValueError(f"No listing for listing_id={listing_id}")
            return (results[0].exchange_id, results[0].security_id)

        pm_a = resolve(pm_a_listing_id)
        pm_b = resolve(pm_b_listing_id)
        k_a = resolve(k_a_listing_id)
        k_b = resolve(k_b_listing_id)

        raw_pairings = [
            (0, "PM_A + K_B", [pm_a, k_b]),
            (1, "PM_B + K_A", [pm_b, k_a]),
        ]
        if allow_dutch_book:
            raw_pairings.append((2, "K_DUTCH_BOOK", [k_a, k_b]))
        self._pairings: list[Pairing] = []
        self._tracked_listings: set[tuple[int, int]] = set()
        for idx, label, legs in raw_pairings:
            self._pairings.append(Pairing(index=idx, label=label, legs=legs))
            for leg in legs:
                self._tracked_listings.add(leg)

        # Book storage
        self._books: dict[tuple[int, int], Book] = {lst: Book() for lst in self._tracked_listings}

        # Cost model
        cost_components: list[CostComponent] = [FeeCost(maker_fee_rates, taker_fee_rates)]
        if fill_risk_lambda > 0.0:
            cost_components.append(FillRiskCost(fill_risk_lambda, unwind_spread_mult, use_cst_fill_model))
        if depth_coverage_mult > 0.0:
            cost_components.append(DepthCoverageCost(depth_coverage_mult, price_window_cents))
        self._cost_model = CompositeCost(cost_components)

        # Price model
        if price_model == "aggressive":
            self._price_model: PriceModel = AggressiveModel(price_improve_bps, price_edge_share)
        elif price_model == "fill_prob_target":
            self._price_model = FillProbTargetModel(target_fill_prob, fill_risk_lambda or 10.0)
        elif price_model == "optimal_ev":
            self._price_model = OptimalEVModel(optimal_ev_lambda, unwind_spread_mult)
        else:
            self._price_model = JoinBestBidModel()

        self._pm_taker_mode = pm_taker_mode

        # Per-pairing state machines (concurrent — each pairing is independent)
        self._pairing_states: list[PairingState] = [
            PairingState(pairing=p) for p in self._pairings
        ]
        # Lookup: listing → pairing states that include it (for fill routing)
        self._listing_to_ps: dict[tuple[int, int], list[PairingState]] = {}
        for ps in self._pairing_states:
            for leg in ps.pairing.legs:
                self._listing_to_ps.setdefault(leg, []).append(ps)

        # Markout tracking
        self._pending_markouts: list[dict] = []

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
                    if self._track_markouts:
                        self._record_markout(listing, report.fill_price, Side.BID, report.timestamp_event)
                    return self._check_fill_transitions(ps, report.timestamp_event)
        elif (self._pm_taker_mode and listing[0] == 4
              and report.exec_type in (ExecType.REJECT, ExecType.EXPIRE)):
            for ps in self._listing_to_ps.get(listing, []):
                if ps.phase not in (Phase.ENTERING, Phase.PARTIAL_FILL):
                    continue
                any_k_filled = any(lg.filled_qty > 0 for lg in ps.legs if lg.listing[0] != 4)
                if any_k_filled:
                    ps.phase = Phase.UNWINDING
                    return self._close_all_positions(ps, report.timestamp_event)
                else:
                    cancel_intents = [
                        Intent(exchange_id=lg.listing[0], security_id=lg.listing[1])
                        for lg in ps.legs if lg.listing[0] != 4
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
        book.bids = bids
        book.asks = asks
        book.last_update_ts = ts
        self._cost_model.on_book_update(listing, book)
        self._price_model.on_book_update(listing, book)

    def _is_stale(self, listing: tuple[int, int], ts: int) -> bool:
        book = self._books[listing]
        return book.last_update_ts == 0 or (ts - book.last_update_ts) > self._max_staleness_ns

    # ------------------------------------------------------------------
    # Fee helpers
    # ------------------------------------------------------------------

    def _maker_fee(self, listing: tuple[int, int], price_scaled: int) -> float:
        rate = self._maker_fee_rates.get(listing[0], 0.0)
        p = price_scaled / PRICE_SCALE
        return rate * p * (1.0 - p)

    def _taker_fee(self, listing: tuple[int, int], price_scaled: int) -> float:
        rate = self._taker_fee_rates.get(listing[0], 0.0)
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

        # Cross-exchange consistency gate: PM ask should ≈ 1 - Kalshi ask.
        # Large divergence means PM is stale and will snap back before our taker fills,
        # causing partial fills that get unwound at a loss.
        if self._max_price_divergence > 0 and len(legs) == 2:
            pm_asks = [books[i].best_ask() / PRICE_SCALE for i in range(2) if legs[i][0] == 4]
            k_asks = [books[i].best_ask() / PRICE_SCALE for i in range(2) if legs[i][0] == 5]
            if pm_asks and k_asks:
                if abs(pm_asks[0] - (1.0 - k_asks[0])) > self._max_price_divergence:
                    return 0.0, 0

        qty = 0
        total_edge = 0.0
        level_indices = [0] * len(legs)
        remaining = [books[i].asks[0].size for i in range(len(legs))]
        max_qty = self._max_position

        while qty < max_qty:
            current_asks = []
            for i, book in enumerate(books):
                if level_indices[i] >= len(book.asks):
                    return total_edge / max(qty, 1), qty
                current_asks.append(book.asks[level_indices[i]].price)

            # Fee-aware marginal edge at this depth
            total_cost = sum(p / PRICE_SCALE for p in current_asks)
            total_fee = sum(
                self._cost_model.total_cost(
                    legs[i], current_asks[i],
                    qty,
                    maker=not (self._pm_taker_mode and legs[i][0] == 4),
                    other_listing=legs[1 - i] if len(legs) == 2 else None,
                )
                for i in range(len(legs))
            )
            settlement_penalty = self._settlement_penalty_per_contract()
            marginal_edge = 1.0 - total_cost - total_fee - settlement_penalty
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
                    # PM taker orders only fill at level 0 (limit to best ask).
                    # Don't walk deeper — the walk would return qty > PM's actual fill.
                    if self._pm_taker_mode and legs[i][0] == 4:
                        return total_edge / max(qty, 1), qty
                    next_idx = level_indices[i] + 1
                    if next_idx < len(books[i].asks):
                        level_indices[i] = next_idx
                        remaining[i] = books[i].asks[next_idx].size
                    else:
                        return total_edge / max(qty, 1), qty

        return (total_edge / max(qty, 1), qty) if qty > 0 else (0.0, 0)

    # ------------------------------------------------------------------
    # Settlement penalty (Feil-Nendel) — disabled when resolution_time_ns=0
    # ------------------------------------------------------------------

    def _settlement_penalty_per_contract(self) -> float:
        if self._gamma_T <= 0.0 or self._resolution_time_ns <= 0:
            return 0.0
        # Penalty is averaged across legs and applied per contract at scan time
        # (Full per-fill accounting happens in _on_entering/_on_partial_fill)
        return 0.0  # simplified: only apply when we have actual positions

    def _settlement_penalty(self, ts: int) -> float:
        if self._gamma_T <= 0.0 or self._resolution_time_ns <= 0:
            return 0.0
        tau = max(0, self._resolution_time_ns - ts) / 1e9
        if tau <= 0:
            return float("inf")
        time_decay = 1.0 / max(tau / 3600.0, 0.01)
        total = 0.0
        for ps in self._pairing_states:
            for leg in ps.legs:
                q = leg.filled_qty / SIZE_SCALE
                p = self._books[leg.listing].mid() / PRICE_SCALE
                p = max(min(p, 0.99), 0.01)
                total += self._gamma_T * q * q * p * (1.0 - p) * time_decay
        return total

    # ------------------------------------------------------------------
    # Budget constraint for maker bids
    # ------------------------------------------------------------------

    def _compute_maker_bid_prices(self, pairing: Pairing, qty: int) -> dict[tuple[int, int], int] | None:
        """Compute entry prices for each leg, respecting the budget constraint.

        PM taker legs use ask price + taker fee; Kalshi legs use best_bid + maker fee.
        Budget: sum(entry_i/PRICE_SCALE + fee_i) < 1.0 - min_edge
        """
        prices: dict[tuple[int, int], int] = {}
        for leg in pairing.legs:
            book = self._books[leg]
            eid = leg[0]
            if self._pm_taker_mode and eid == 4:
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
            eid = leg[0]
            if self._pm_taker_mode and eid == 4:
                total_fees += self._taker_fee(leg, p)
            else:
                total_fees += self._maker_fee(leg, p)
        if total + total_fees >= 1.0 - self._min_edge:
            return None

        return prices

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _on_scanning(self, ps: PairingState, ts: int) -> list[Intent]:
        for lg in ps.pairing.legs:
            eid, sid = lg
            pos = self.positions.get_position(eid, sid)
            current = pos.net_quantity if pos is not None else 0
            if current > ps.base_qty:
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

        bid_prices = self._compute_maker_bid_prices(ps.pairing, target_qty)
        if bid_prices is None:
            return []

        ps.legs = [LegState(listing=lg, target_qty=target_qty) for lg in ps.pairing.legs]
        ps.entry_ts = ts
        ps.phase = Phase.ENTERING

        if self._metrics_buf is not None:
            row = self._metrics_buf.appendRow()
            self._metrics_buf.setLong(row, self._m_ts, ts)
            self._metrics_buf.setInt(row, self._m_phase, int(Phase.ENTERING))
            self._metrics_buf.setInt(row, self._m_pairing, ps.pairing.index)
            self._metrics_buf.setDouble(row, self._m_edge, edge)
            self._metrics_buf.setLong(row, self._m_qty, target_qty)

        intents = []
        for leg in ps.legs:
            eid = leg.listing[0]
            if self._pm_taker_mode and eid == 4:
                book = self._books[leg.listing]
                limit_price = book.best_ask()
                intents.append(Intent(
                    exchange_id=eid,
                    security_id=leg.listing[1],
                    take_side=Side.BID,
                    take_size=leg.target_qty,
                    take_order_type=OrderType.LIMIT,
                    take_limit_price=limit_price,
                ))
            else:
                bid = bid_prices[leg.listing]
                intents.append(Intent(
                    exchange_id=eid,
                    security_id=leg.listing[1],
                    bid_price=bid,
                    bid_size=leg.target_qty,
                ))
        return intents

    def _on_entering(self, ps: PairingState, ts: int) -> list[Intent]:
        return self._check_fill_transitions(ps, ts)

    def _check_fill_transitions(self, ps: PairingState, ts: int) -> list[Intent]:
        all_filled = all(lg.is_filled for lg in ps.legs)
        any_filled = any(lg.filled_qty > 0 for lg in ps.legs)
        some_filled = any_filled and not all_filled

        if all_filled:
            base = 0
            for lg in ps.legs:
                eid, sid = lg.listing
                pos = self.positions.get_position(eid, sid)
                if pos is not None:
                    base = max(base, pos.net_quantity)
            ps.base_qty = base
            ps.phase = Phase.SCANNING
            self._reset(ps)
            return []

        if some_filled and ps.phase == Phase.ENTERING:
            ps.phase = Phase.PARTIAL_FILL
            ps.partial_fill_since = ts
            return []

        if some_filled and ps.phase == Phase.PARTIAL_FILL:
            if self._check_imbalance_timeout(ps, ts):
                ps.phase = Phase.UNWINDING
                return self._close_all_positions(ps, ts)
            # Don't run a budget check here. The PM leg already filled at its price
            # (that cost is locked); using the *current* PM ask for the budget would
            # falsely reject valid arbs whenever PM price jumps post-fill. Wait the
            # full imbalance_timeout before declaring the K leg missing.
            return []

        # Guard: PM taker orders take ~300ms to settle. Don't cancel Kalshi makers
        # or reset state until PM has had a chance to fill or be rejected — otherwise
        # PM fills after the reset, creating untracked naked directional exposure.
        if (self._pm_taker_mode
                and ps.entry_ts > 0
                and (ts - ps.entry_ts) < _PM_TAKER_SETTLE_NS
                and any(lg.listing[0] == 4 and not lg.is_filled for lg in ps.legs)):
            return []

        unfilled_legs = [lg for lg in ps.legs if not lg.is_filled]
        if unfilled_legs:
            bid_prices = self._compute_maker_bid_prices(ps.pairing, unfilled_legs[0].target_qty)
            if bid_prices is None:
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

        return []

    def _on_partial_fill(self, ps: PairingState, ts: int) -> list[Intent]:
        return self._check_fill_transitions(ps, ts)

    def _on_closing(self, ps: PairingState, ts: int) -> list[Intent]:
        # Use net_quantity (settled fills) not effective_quantity (pending sells) to avoid
        # premature re-entry while close orders are still in flight.
        def _net_qty(lg: LegState) -> int:
            pos = self.positions.get_position(lg.listing[0], lg.listing[1])
            return pos.net_quantity if pos is not None else 0

        all_at_base = all(_net_qty(lg) <= ps.base_qty for lg in ps.legs)
        if all_at_base:
            ps.phase = Phase.SCANNING
            self._reset(ps)
            return []
        if ts - ps.last_close_ts > 2_000_000_000:
            ps.last_close_ts = ts
            intents = []
            for leg in ps.legs:
                eid, sid = leg.listing
                eff_qty = self.positions.get_effective_quantity(eid, sid)
                qty = eff_qty - ps.base_qty
                if qty > 0:
                    intents.append(Intent(
                        exchange_id=eid, security_id=sid,
                        take_side=Side.ASK,
                        take_size=qty,
                        take_order_type=OrderType.LIMIT,
                        take_limit_price=1,
                    ))
                elif qty < 0:
                    intents.append(Intent(
                        exchange_id=eid, security_id=sid,
                        take_side=Side.BID,
                        take_size=-qty,
                        take_order_type=OrderType.LIMIT,
                        take_limit_price=PRICE_SCALE - 1,
                    ))
            return intents
        return []

    # ------------------------------------------------------------------
    # Closing / unwind helpers
    # ------------------------------------------------------------------

    def _close_all_positions(self, ps: PairingState, ts: int) -> list[Intent]:
        ps.last_close_ts = ts
        intents = []
        for leg in ps.legs:
            eid, sid = leg.listing
            pos = self.positions.get_position(eid, sid)
            total_qty = pos.net_quantity if pos is not None else 0
            qty = total_qty - ps.base_qty
            if qty > 0:
                intents.append(Intent(
                    exchange_id=eid, security_id=sid,
                    take_side=Side.ASK,
                    take_size=qty,
                    take_order_type=OrderType.LIMIT,
                    take_limit_price=1,
                ))
            elif qty < 0:
                intents.append(Intent(
                    exchange_id=eid, security_id=sid,
                    take_side=Side.BID,
                    take_size=-qty,
                    take_order_type=OrderType.LIMIT,
                    take_limit_price=PRICE_SCALE - 1,
                ))
            elif not (self._pm_taker_mode and eid == 4):
                intents.append(Intent(exchange_id=eid, security_id=sid))
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

    def _record_markout(self, listing: tuple[int, int], fill_price: int, side: Side, ts: int) -> None:
        self._pending_markouts.append({
            "listing": listing,
            "fill_price": fill_price,
            "side": side,
            "timestamp": ts,
            "mid_at_fill": self._books[listing].mid(),
        })

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def _reset(self, ps: PairingState) -> None:
        ps.legs = []
        ps.partial_fill_since = None
        ps.entry_ts = 0
