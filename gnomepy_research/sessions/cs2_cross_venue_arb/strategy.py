"""
CS2 cross-venue arb: buy both teams of one market on different venues when the pair costs under $1 after fees.

A session holds one market - a match winner or one map's winner - with each venue's two team listings (Polymarket
International, Kalshi, Polymarket US). The launcher starts it only once the match is confirmed live, because a match
that is never played settles differently by venue (Polymarket International pays 50/50, Kalshi and Polymarket US the
last fair price), which turns a locked pair into a loss. The strategy itself never checks whether the match is live.

On every book update with nothing in flight, it finds the venue pair and size where holding team A on one venue and
team B on another clears min_edge per pair after each venue's taker fee, walking each book only as deep as every pair
still clears it, and sends both legs at once as IOC limits at the deepest price it costed (decision.py). A team is
held by buying its listing, or on a venue in sell_rival_venues by selling the rival's listing, which Kalshi books as
NO on it.

The legs land at different venues at different times, so they can come back unequal. Once neither is in flight, any
difference between what pays if team A wins and what pays if team B wins is completed by buying the short team
wherever it is cheapest, no dearer than its best level plus completion_slippage, retried on later updates until the
two match. Completion never waits for the cooldown and never stops for the stake cap: an unhedged position is the one
risk the strategy exists not to hold.

A venue may hold a taker order before matching it (Polymarket International delays marketable orders on these
markets by about a second, its secondsDelay, during which makers can pull quotes). An attempt sends the slowest
venue's leg first and holds each other leg until it would match at about the same time, so the fast leg doesn't fill
against a price the slow one no longer gets. Held legs go out on the first callback after their time.

After each attempt the strategy waits cooldown_seconds before the next one: the recorded and live feeds never show
our own fills, so the same quotes would otherwise be taken again from a book that no longer has them. A venue pair
whose legs came back short is also left alone until one of its tops of book moves.

What it holds is read from the engine's positions, so a restarted session completes an unequal position instead of
forgetting it. It holds positions to settlement; the session's end is the launcher's or the backtest's.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import permutations

from gnomepy import ExecutionReport, Intent, OrderStatus, OrderType, Scales, Side, Strategy
from gnomepy.java.schemas import StreamRecord
from gnomepy.registry import RegistryClient

from gnomepy_research.fees import registry_fee_rates
from gnomepy_research.sessions.cs2_cross_venue_arb.decision import (
    BUY,
    Level,
    Order,
    exposure,
    plan_completion,
    plan_pair,
    team_ladder,
    whole,
)

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
ARB, COMPLETION = 0, 1


@dataclass
class _Leg:
    venue: str
    team: int  # 0 = team A, 1 = team B
    listing_id: int
    exchange_id: int
    security_id: int
    min_size: float
    tick: float = 0.01
    asks: list[Level] = field(default_factory=list)
    bids: list[Level] = field(default_factory=list)
    bad: bool = True  # no book yet is no book to trust
    in_flight: bool = False

    def top(self) -> tuple:
        return (self.asks[0] if self.asks else None, self.bids[0] if self.bids else None)


class CS2CrossVenueArb(Strategy):
    """
    Args:
        legs: one {venue, listing_team_a, listing_team_b} per venue listing the market; the same two teams in the same
            order on every venue. Set by the launcher.
        min_edge: the least locked profit per pair, after both fees, worth sending.
        max_contracts: the most pairs one attempt sends.
        max_locked_usd: the most pairs (each paying $1) the session holds; no new attempt starts past it.
        completion_slippage: how far above the short team's best level a completion may pay.
        cooldown_seconds: the least time between two attempts.
        sell_rival_venues: venues where a team may also be held by selling the rival team's listing.
        taker_delay_seconds: how long each venue holds a taker order before matching it; legs are staggered by it.
        book_depth: book levels read from each update.
    """

    def __init__(
        self,
        legs: list[dict],
        min_edge: float = 0.01,
        max_contracts: float = 1000.0,
        max_locked_usd: float = 2000.0,
        completion_slippage: float = 0.05,
        cooldown_seconds: float = 10.0,
        sell_rival_venues: tuple[str, ...] = ("kalshi",),
        taker_delay_seconds: dict[str, float] | None = None,
        book_depth: int = 10,
        processing_time_ns: int = 0,
    ):
        venues = [leg["venue"] for leg in legs]
        if len(venues) < 2 or len(set(venues)) != len(venues):
            raise ValueError(f"legs must name at least two venues, each once, got {venues}")
        if min_edge <= 0:
            raise ValueError(f"min_edge must be positive, got {min_edge}")
        self.min_edge = min_edge
        self.max_contracts = max_contracts
        self.max_locked_usd = max_locked_usd
        self.completion_slippage = completion_slippage
        self.cooldown_ns = int(cooldown_seconds * 1e9)
        self.sell_rival_venues = set(sell_rival_venues)
        delays = {"polymarket": 1.0} if taker_delay_seconds is None else taker_delay_seconds
        self._delay_ns = {venue: int(seconds * 1e9) for venue, seconds in delays.items()}
        # Legs of the current attempt waiting for their send time: (send at, order, edge, profit).
        self._held: list[tuple[int, Order, float, float]] = []
        self.book_depth = book_depth
        self._processing_time_ns = processing_time_ns
        self._fee_rates = registry_fee_rates(self)
        self._metrics_buf = None
        self._cooldown_until = 0
        # Venue pair -> the tops of book its legs came back short against; untouched until one moves.
        self._backoff: dict[tuple[str, str], tuple] = {}
        self._last_pair: tuple[str, str] | None = None

        registry = RegistryClient()
        self._venue_legs: dict[str, tuple[_Leg, _Leg]] = {}
        for leg in legs:
            self._venue_legs[leg["venue"]] = (self._resolve(registry, leg["venue"], 0, leg["listing_team_a"]),
                                              self._resolve(registry, leg["venue"], 1, leg["listing_team_b"]))
        self._legs = [leg for pair in self._venue_legs.values() for leg in pair]
        self._leg_of = {leg.listing_id: leg for leg in self._legs}

    @staticmethod
    def _resolve(registry, venue: str, team: int, listing_id: int) -> _Leg:
        listings = registry.get_listing(listing_id=int(listing_id))
        if not listings:
            raise ValueError(f"listing {listing_id} not found in registry")
        specs = registry.get_listing_spec(listing_id=int(listing_id))
        min_size = int(specs[0].min_size) / SIZE_SCALE if specs and specs[0].min_size else 0.0
        tick = int(specs[0].tick_size) / PRICE_SCALE if specs and getattr(specs[0], "tick_size", None) else 0.01
        return _Leg(venue=venue, team=team, listing_id=int(listing_id), exchange_id=listings[0].exchange_id,
                    security_id=listings[0].security_id, min_size=min_size, tick=tick)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def register_metrics(self) -> None:
        buf = self.metrics.create_stream("cs2_arb_orders")
        self._m_ts = buf.add_long_column("timestamp")
        self._m_kind = buf.add_long_column("kind")
        self._m_listing = buf.add_long_column("listing_id")
        self._m_direction = buf.add_long_column("direction")
        self._m_contracts = buf.add_double_column("contracts")
        self._m_limit = buf.add_double_column("limit_price")
        self._m_edge = buf.add_double_column("edge_first")
        self._m_profit = buf.add_double_column("planned_profit")
        self._m_gap = buf.add_double_column("gap_before")
        buf.freeze()
        self._metrics_buf = buf

    # ---- engine callbacks ----------------------------------------------------------------

    def on_market_data(self, data: StreamRecord) -> list[Intent]:
        leg = self._leg_of.get(self.streams.listing_id(data.stream_id))
        if leg is None:
            return []
        leg.asks = [Level(data.ask_price(i) / PRICE_SCALE, data.ask_size(i) / SIZE_SCALE)
                    for i in range(self.book_depth) if data.ask_size(i) > 0]
        leg.bids = [Level(data.bid_price(i) / PRICE_SCALE, data.bid_size(i) / SIZE_SCALE)
                    for i in range(self.book_depth) if data.bid_size(i) > 0]
        leg.bad = data.is_bad_data
        return self._decide(data.event_timestamp)

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        leg = self._leg_of.get(self.streams.listing_id_for(report.exchange_id, report.security_id))
        if leg is None or report.order_status not in TERMINAL:
            return []
        leg.in_flight = False
        if report.order_status != OrderStatus.FILLED and self._last_pair is not None:
            self._backoff[self._last_pair] = self._pair_tops(self._last_pair)
        return self._decide(report.timestamp_recv)

    # ---- decisions -----------------------------------------------------------------------

    def _decide(self, now_ns: int) -> list[Intent]:
        if self._held:
            due = [h for h in self._held if h[0] <= now_ns]
            self._held = [h for h in self._held if h[0] > now_ns]
            return [self._send(now_ns, ARB, order, edge, profit, 0.0) for _, order, edge, profit in due]
        if any(leg.in_flight for leg in self._legs):
            return []
        pays_a, pays_b = self._exposure()
        gap = pays_a - pays_b
        if abs(gap) >= 1:
            return self._complete(now_ns, 1 if gap > 0 else 0, abs(gap), gap)
        if now_ns < self._cooldown_until:
            return []
        locked = min(pays_a, pays_b)
        room = min(self.max_contracts, self.max_locked_usd - locked)
        if room < 1:
            return []
        best = None
        for va, vb in permutations(self._venue_legs, 2):
            tops = self._pair_tops((va, vb))
            if self._backoff.get((va, vb)) == tops:
                continue
            ladder_a, ladder_b = self._ladder(va, 0), self._ladder(vb, 1)
            plan = plan_pair(ladder_a, ladder_b, self.min_edge, room)
            plan = whole(plan, self._min_sizes()) if plan else None
            if plan and (best is None or plan.profit > best[1].profit):
                best = ((va, vb), plan)
        if best is None:
            return []
        (va, vb), plan = best
        self._last_pair = (va, vb)
        self._backoff.pop((va, vb), None)
        self._cooldown_until = now_ns + self.cooldown_ns
        logger.info("arb %s team A + %s team B: %.0f pairs, first edge %.4f, planned $%.2f", va, vb, plan.contracts,
                    plan.edge_first, plan.profit)
        orders = plan.orders_a + plan.orders_b
        slowest = max(self._delay_ns.get(self._leg_of[o.listing_id].venue, 0) for o in orders)
        now_orders = []
        for o in orders:
            wait = slowest - self._delay_ns.get(self._leg_of[o.listing_id].venue, 0)
            if wait > 0:
                self._held.append((now_ns + wait, o, plan.edge_first, plan.profit))
            else:
                now_orders.append(o)
        return [self._send(now_ns, ARB, o, plan.edge_first, plan.profit, 0.0) for o in now_orders]

    def _complete(self, now_ns: int, team: int, contracts: float, gap: float) -> list[Intent]:
        ladder = sorted((lv for venue in self._venue_legs for lv in self._ladder(venue, team)),
                        key=lambda r: (r.all_in, r.direction))
        orders = plan_completion(ladder, contracts, self.completion_slippage, self._min_sizes())
        if not orders:
            return []
        logger.info("completing %.0f contracts of team %s (gap %.2f) over %d orders", contracts, "AB"[team], gap,
                    len(orders))
        return [self._send(now_ns, COMPLETION, o, math.nan, math.nan, gap) for o in orders]

    # ---- state ---------------------------------------------------------------------------

    def _exposure(self) -> tuple[float, float]:
        holdings = []
        for leg in self._legs:
            pos = self.positions.get_position(leg.exchange_id, leg.security_id)
            holdings.append((leg.team, pos.net_quantity / SIZE_SCALE if pos else 0.0))
        return exposure(holdings)

    def _ladder(self, venue: str, team: int):
        legs = self._venue_legs[venue]
        own, rival = legs[team], legs[1 - team]
        if own.bad or rival.bad:
            return []
        rate = self._fee_rates(own.listing_id)[0]
        return team_ladder(own.listing_id, own.asks, rival.listing_id, rival.bids, rate,
                           venue in self.sell_rival_venues)

    def _pair_tops(self, pair: tuple[str, str]) -> tuple:
        return tuple(leg.top() for venue in pair for leg in self._venue_legs[venue])

    def _min_sizes(self) -> dict[int, float]:
        return {leg.listing_id: leg.min_size for leg in self._legs}

    # ---- orders --------------------------------------------------------------------------

    def _send(self, now_ns: int, kind: int, order: Order, edge: float, profit: float, gap: float) -> Intent:
        leg = self._leg_of[order.listing_id]
        leg.in_flight = True
        self._record(now_ns, kind, order, edge, profit, gap)
        return Intent(leg.exchange_id, leg.security_id, take_side=Side.BID if order.direction == BUY else Side.ASK,
                      take_size=round(order.contracts * SIZE_SCALE), take_order_type=OrderType.LIMIT,
                      take_limit_price=round(order.limit * PRICE_SCALE))

    def _record(self, now_ns: int, kind: int, order: Order, edge: float, profit: float, gap: float) -> None:
        if self._metrics_buf is None:
            return
        buf = self._metrics_buf
        buf.begin()
        buf.set_long(self._m_ts, now_ns)
        buf.set_long(self._m_kind, kind)
        buf.set_long(self._m_listing, order.listing_id)
        buf.set_long(self._m_direction, order.direction)
        buf.set_double(self._m_contracts, order.contracts)
        buf.set_double(self._m_limit, order.limit)
        buf.set_double(self._m_edge, edge)
        buf.set_double(self._m_profit, profit)
        buf.set_double(self._m_gap, gap)
        buf.commit()
