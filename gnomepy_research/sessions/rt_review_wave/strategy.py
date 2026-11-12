"""
A film's first wave of outlet reviews, traded on every venue listing its Rotten Tomatoes strikes.

gnome-launcher starts one session per film (the `rt` rule) holding, for each "at least X" strike on each venue
(Kalshi, Polymarket International, Polymarket US), the strike's YES and NO listings. The session waits, often for days,
until the film's first review wave: the third review, of a cluster seen within cluster_hours, of those the rt_reviews
source saw soon after they were published (review_feed.py, decision.py). The research behind it: over Jan-Nov 2026,
taking the side of the reviews known 5 minutes after the wave started made +11c a contract net of the taker fee on
Kalshi (51 films), fading to +3.5c by an hour, because the venues absorb a wave slowly.

From the wave's start for window_minutes it buys, on every strike whose price is inside [min_price, max_price], YES
when the share of reviews Claude called fresh is at least threshold above the baseline (the past films' median share
at that point) and NO when it is that far below. Orders are IOC limits at most max_through above the best ask, sized by
what the book offers up to that limit, max_contracts_per_strike per strike and max_usd_per_venue per venue, fees
included. Once a side is held, only that side is bought: a signal that flips as more reviews arrive stops the buying
instead of buying the other side. Positions are held to settlement.

It decides again only when something it reads has changed: a strike's top of book, the reviews known, a fill, or the
wave starting. Each strike has at most one order in flight, and an order that comes back short backs its listing off
until that listing's top of book moves, so a paper session, whose feed never shows its own fills, can't send the same
order over and over.

The session ends itself, keeping what it holds, once the window has passed and no order is in flight; a session
started after its film's wave ends at once. State that must survive a restart (what it holds, money committed) is read
from the engine's positions, so a restarted session resumes instead of buying again.

The session is woken by market data: its strikes' books and any extra stream it reads (`streams` in the session
config). Reviews arrive by a table re-read every reload_seconds, and are acted on at the next wake-up.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from gnomepy import ExecutionReport, Intent, OrderStatus, OrderType, Scales, Side, Strategy
from gnomepy.java.schemas import StreamRecord
from gnomepy.registry import RegistryClient

from gnomepy_research.fees import registry_fee_rates
from gnomepy_research.sessions.rt_review_wave.decision import NO, YES, Level, Signal, signal, take, wave_start
from gnomepy_research.sessions.rt_review_wave.review_feed import REVIEWS, ReviewFeed

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
MINUTE_NS = 60_000_000_000
HOUR_NS = 60 * MINUTE_NS
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
SIDE_NAME = {YES: "YES", NO: "NO"}
VENUES = ("kalshi", "polymarket", "polymarket_us")


@dataclass
class _Book:
    """One side (YES or NO) of one strike on one venue."""
    venue: str
    strike: int
    side: int
    listing_id: int
    exchange_id: int
    security_id: int
    tick: float
    lot: int
    min_size: int
    asks: list[Level] = field(default_factory=list)
    bids: list[Level] = field(default_factory=list)
    in_flight: bool = False
    # What the order in flight commits, contracts and USD with fees: counted against the caps until its final report,
    # whether or not the engine shows it as pending yet.
    in_flight_contracts: float = 0.0
    in_flight_usd: float = 0.0
    # The top of book when an order here last came back short; nothing is sent here until the top moves off it.
    backoff_top: tuple | None = None


def _top(book: _Book) -> tuple:
    return (book.asks[0] if book.asks else None, book.bids[0] if book.bids else None)


class RtReviewWave(Strategy):
    """
    Args:
        film_id: the film, as rt_films and rt_reviews name it.
        legs: one {venue, strike, yes_listing, no_listing} per strike per venue: the registry listings of the strike's
            two sides. Set by the launcher.
        baseline: the fresh share the signal is measured from: the past films' median at 5 minutes into their wave.
        threshold: how far from the baseline the fresh share must be to trade.
        min_reviews: reviews needed both to start a wave and to trade.
        cluster_hours: the span the wave's first min_reviews reviews must be seen within.
        max_discovery_lag_hours: a review seen later than this after it was published doesn't count towards the
            wave's start (it still counts in the share).
        window_minutes: how long after the wave's start it may buy.
        min_price, max_price: the prices it buys at; outside them a strike's result is all but settled.
        max_through: how far above the best ask an order's limit may reach.
        max_contracts_per_strike: most it holds of one strike on one venue.
        max_usd_per_venue: most it commits on one venue, fees included.
        min_order_usd: smallest order worth sending.
        reload_seconds: how often to re-read the film's reviews (0 = once, for replays).
        reviews_source: the rt_reviews table, or a parquet extract of it.
    """

    def __init__(
        self,
        film_id: str,
        legs: list[dict],
        baseline: float = 0.667,
        threshold: float = 0.10,
        min_reviews: int = 3,
        cluster_hours: float = 6.0,
        max_discovery_lag_hours: float = 6.0,
        window_minutes: float = 15.0,
        min_price: float = 0.03,
        max_price: float = 0.97,
        max_through: float = 0.03,
        max_contracts_per_strike: float = 100.0,
        max_usd_per_venue: float = 1000.0,
        min_order_usd: float = 1.0,
        book_depth: int = 10,
        processing_time_ns: int = 0,
        reload_seconds: float = 5.0,
        reviews_source: str = REVIEWS,
    ):
        if not legs:
            raise ValueError("legs must name at least one strike")
        unknown = {leg["venue"] for leg in legs} - set(VENUES)
        if unknown:
            raise ValueError(f"unknown venues {sorted(unknown)}; expected {VENUES}")
        if not 0 < min_price < max_price < 1:
            raise ValueError(f"need 0 < min_price < max_price < 1, got {min_price}, {max_price}")
        self.film_id = film_id
        self.baseline = baseline
        self.threshold = threshold
        self.min_reviews = min_reviews
        self.cluster_ns = int(cluster_hours * HOUR_NS)
        self.max_lag_ns = int(max_discovery_lag_hours * HOUR_NS)
        self.window_ns = int(window_minutes * MINUTE_NS)
        self.min_price, self.max_price = min_price, max_price
        self.max_through = max_through
        self.max_contracts_per_strike = max_contracts_per_strike
        self.max_usd_per_venue = max_usd_per_venue
        self.min_order_usd = min_order_usd
        self.book_depth = book_depth
        self._processing_time_ns = processing_time_ns
        self._fee_rates = registry_fee_rates(self)
        self._orders_buf = None
        self._decisions_buf = None
        self._dirty = True
        self._decided_reviews: tuple | None = None
        self._recorded: tuple | None = None
        self._wave_ns: int | None = None

        registry = RegistryClient()
        self._books: list[_Book] = []
        for leg in legs:
            for side, key in ((YES, "yes_listing"), (NO, "no_listing")):
                self._books.append(self._resolve(registry, leg["venue"], int(leg["strike"]), side, leg[key]))
        self._book_of = {b.listing_id: b for b in self._books}
        self._feed = ReviewFeed(film_id, reviews_source, reload_seconds)

    @staticmethod
    def _resolve(registry, venue: str, strike: int, side: int, listing_id: int) -> _Book:
        listings = registry.get_listing(listing_id=int(listing_id))
        if not listings:
            raise ValueError(f"listing {listing_id} not found in registry")
        specs = registry.get_listing_spec(listing_id=int(listing_id))
        spec = specs[0] if specs else None
        tick = int(spec.tick_size) / PRICE_SCALE if spec and spec.tick_size else 0.01
        lot = int(spec.lot_size) if spec and spec.lot_size else SIZE_SCALE
        min_size = int(spec.min_size) if spec and spec.min_size else 0
        return _Book(venue, strike, side, int(listing_id), listings[0].exchange_id, listings[0].security_id, tick,
                     lot, min_size)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def register_metrics(self) -> None:
        orders = self.metrics.create_stream("rt_review_wave_orders")
        self._o_ts = orders.add_long_column("timestamp")
        self._o_listing = orders.add_long_column("listing_id")
        self._o_strike = orders.add_long_column("strike")
        self._o_side = orders.add_long_column("side")
        self._o_venue = orders.add_long_column("venue")
        self._o_best = orders.add_double_column("best_ask")
        self._o_limit = orders.add_double_column("limit_price")
        self._o_contracts = orders.add_double_column("contracts")
        self._o_usd = orders.add_double_column("usd")
        self._o_held = orders.add_double_column("held_contracts")
        self._o_signal = orders.add_double_column("signal")
        orders.freeze()
        self._orders_buf = orders

        decisions = self.metrics.create_stream("rt_review_wave_decisions")
        self._d_ts = decisions.add_long_column("timestamp")
        self._d_wave = decisions.add_long_column("wave_start")
        self._d_reviews = decisions.add_long_column("reviews")
        self._d_share = decisions.add_double_column("fresh_share")
        self._d_signal = decisions.add_double_column("signal")
        self._d_side = decisions.add_long_column("side")
        self._d_held_side = decisions.add_long_column("held_side")
        decisions.freeze()
        self._decisions_buf = decisions

    def _taker_rate(self, book: _Book) -> float:
        return self._fee_rates(book.listing_id)[0]

    # ---- state read from the engine's positions --------------------------------------

    def _book_state(self, book: _Book) -> tuple[float, float]:
        """(contracts held or on order, committed USD incl. fees and orders still working)."""
        pos = self.positions.get_position(book.exchange_id, book.security_id)
        filled = pos.net_quantity / SIZE_SCALE if pos else 0.0
        avg = pos.avg_entry_price / PRICE_SCALE if pos and pos.net_quantity else 0.0
        fees = pos.total_fees / PRICE_SCALE if pos else 0.0
        if book.in_flight:
            return filled + book.in_flight_contracts, filled * avg + fees + book.in_flight_usd
        return filled, filled * avg + fees

    # ---- engine callbacks ----------------------------------------------------------------

    def on_market_data(self, data: StreamRecord) -> list[Intent]:
        book = self._book_of.get(self.streams.listing_id(data.stream_id))
        if book is not None:
            before = _top(book)
            book.asks = [Level(data.ask_price(i) / PRICE_SCALE, data.ask_size(i) / SIZE_SCALE)
                         for i in range(self.book_depth) if data.ask_size(i) > 0]
            book.bids = [Level(data.bid_price(i) / PRICE_SCALE, data.bid_size(i) / SIZE_SCALE)
                         for i in range(self.book_depth) if data.bid_size(i) > 0]
            top = _top(book)
            if top != before:
                self._dirty = True
            if book.backoff_top is not None and top != book.backoff_top:
                book.backoff_top = None
                self._dirty = True
        return self._evaluate(data.event_timestamp)

    def on_stream_event(self, stream_id: int, record: StreamRecord) -> list[Intent]:
        # Another stream's records only wake the session, so reviews are acted on between this film's book updates.
        return self._evaluate(record.event_timestamp)

    def _evaluate(self, now_ns: int) -> list[Intent]:
        if self.is_exiting:
            return []
        reviews = self._feed.known(now_ns)
        start = wave_start(reviews, self.min_reviews, self.cluster_ns, self.max_lag_ns)
        if start is None:
            return []
        if self._wave_ns != start:
            self._wave_ns = start
            self._dirty = True
            logger.info("film %s: wave started at %d with %d reviews known", self.film_id, start, len(reviews))
        in_flight = any(b.in_flight for b in self._books)
        if now_ns >= start + self.window_ns:
            if not in_flight:
                self.request_exit(f"film {self.film_id}: its wave's {self.window_ns // MINUTE_NS} min window passed")
            return []
        key = tuple(reviews)
        if key != self._decided_reviews:
            self._dirty = True
        if not self._dirty:
            return []
        self._dirty = False
        self._decided_reviews = key

        sig = signal(reviews, self.baseline, self.threshold, self.min_reviews)
        states = {b.listing_id: self._book_state(b) for b in self._books}
        held = {side for side in (YES, NO) if any(states[b.listing_id][0] > 0 for b in self._books if b.side == side)}
        held_side = next(iter(held)) if len(held) == 1 else 0
        self._record_decision(now_ns, start, sig, held_side)
        if sig.side == 0 or (held_side and sig.side != held_side) or len(held) > 1:
            return []

        spent = {venue: 0.0 for venue in VENUES}
        for b in self._books:
            spent[b.venue] += states[b.listing_id][1]
        intents = []
        for b in self._books:
            if b.side != sig.side or b.in_flight or b.backoff_top is not None or not b.asks:
                continue
            best = b.asks[0].price
            if not self.min_price <= best <= self.max_price:
                continue
            limit = min(self.max_price, _on_tick(best + self.max_through, b.tick))
            contracts, _ = states[b.listing_id]
            t = take(b.asks, limit, self.max_contracts_per_strike - contracts, self.max_usd_per_venue - spent[b.venue],
                     self._taker_rate(b), b.lot / SIZE_SCALE)
            if t is None or t.cost < self.min_order_usd or round(t.contracts * SIZE_SCALE) < b.min_size:
                continue
            spent[b.venue] += t.cost
            intents += self._order(now_ns, b, best, t.limit_price, t.contracts, t.cost, contracts, sig)
        return intents

    def _order(self, now_ns: int, book: _Book, best: float, limit: float, contracts: float, usd: float, held: float,
               sig: Signal) -> list[Intent]:
        book.in_flight, book.in_flight_contracts, book.in_flight_usd = True, contracts, usd
        self._record_order(now_ns, book, best, limit, contracts, usd, held, sig.signal)
        logger.info("film %s BUY %s %s >=%d (listing %d): %d reviews, fresh %.3f, signal %+.3f | best ask %.3f, "
                    "limit %.3f, %.2f contracts $%.2f | held %.2f", self.film_id, book.venue, SIDE_NAME[book.side],
                    book.strike, book.listing_id, sig.reviews, sig.fresh_share, sig.signal, best, limit, contracts,
                    usd, held)
        return [Intent(book.exchange_id, book.security_id, take_side=Side.BID, take_size=round(contracts * SIZE_SCALE),
                       take_order_type=OrderType.LIMIT, take_limit_price=round(limit * PRICE_SCALE))]

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        book = self._book_of.get(self.streams.listing_id_for(report.exchange_id, report.security_id))
        if book is None:
            return []
        if getattr(report, "filled_qty", 0) or report.order_status == OrderStatus.FILLED:
            self._dirty = True
        if report.order_status in TERMINAL:
            book.in_flight = False
            self._dirty = True
            if report.order_status != OrderStatus.FILLED:
                book.backoff_top = _top(book)
                logger.info("film %s %s %s >=%d (listing %d): order %s with contracts left, backing off until its top "
                            "of book moves", self.film_id, book.venue, SIDE_NAME[book.side], book.strike,
                            book.listing_id, report.order_status.name)
        return []

    # ---- metrics ---------------------------------------------------------------------------

    def _record_decision(self, now_ns: int, start: int, sig: Signal, held_side: int) -> None:
        key = (start, sig.reviews, sig.side, held_side)
        if self._decisions_buf is None or key == self._recorded:
            return
        self._recorded = key
        buf = self._decisions_buf
        buf.begin()
        buf.set_long(self._d_ts, now_ns)
        buf.set_long(self._d_wave, start)
        buf.set_long(self._d_reviews, sig.reviews)
        buf.set_double(self._d_share, sig.fresh_share)
        buf.set_double(self._d_signal, sig.signal)
        buf.set_long(self._d_side, sig.side)
        buf.set_long(self._d_held_side, held_side)
        buf.commit()

    def _record_order(self, now_ns: int, book: _Book, best: float, limit: float, contracts: float, usd: float,
                      held: float, sig: float) -> None:
        if self._orders_buf is None:
            return
        buf = self._orders_buf
        buf.begin()
        buf.set_long(self._o_ts, now_ns)
        buf.set_long(self._o_listing, book.listing_id)
        buf.set_long(self._o_strike, book.strike)
        buf.set_long(self._o_side, book.side)
        buf.set_long(self._o_venue, VENUES.index(book.venue))
        buf.set_double(self._o_best, best)
        buf.set_double(self._o_limit, limit)
        buf.set_double(self._o_contracts, contracts)
        buf.set_double(self._o_usd, usd)
        buf.set_double(self._o_held, held)
        buf.set_double(self._o_signal, sig)
        buf.commit()


def _on_tick(price: float, tick: float) -> float:
    """The price rounded down to the tick, so a limit never reaches past max_through."""
    return math.floor(price / tick + 1e-9) * tick if tick > 0 else price
