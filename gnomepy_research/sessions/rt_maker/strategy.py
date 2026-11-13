"""
A resting maker on a film's Kalshi Rotten Tomatoes strikes, for Kalshi's liquidity rewards.

gnome-launcher starts one session per film (the `rt_maker` rule) holding the YES listing of each of the film's Kalshi
"at least X" strikes. On every strike with a liquidity-reward program in force, the session rests quote_size contracts
at the best bid and the best ask other makers show (quotes.py): a YES bid, and a YES ask, which Kalshi books as a NO
bid, so both sides of the reward's scoring are covered. A side stops once the strike's position reaches max_position,
and quotes that would commit more of max_film_usd than is left are cut back to the side that trades the position back
towards flat. Positions are held; nothing is ever taken.

Kalshi pays the rewards for resting size near the touch on both sides; the risk is being filled by a taker who knows
something the book hasn't priced. On these markets that is mostly a critic's review (gnome-pipelines' rt_reviews,
re-read every reload_seconds), and signal_mode says what a review does:
- pull: for pull_minutes after any review, every quote is pulled, then both sides come back.
- skew: the review signal (models/rt/signal.py: Claude's fresh share against the baseline) leans the quotes. A review
  with a side quotes only that side, the YES bid for YES and the YES ask for NO, for skew_hours (None: until the next
  review); a review without a side, or one that flips the last side leaned, pulls both for pull_minutes first. With
  pull_every_review, every review pulls both for pull_minutes, and the side is quoted alone only after that.
In both, a change in the Tomatometer's count (rt_scores) pulls every quote for pull_minutes.

Why skew: after a film's reviews arrive, Kalshi under-reacts in their direction, so both sides at the touch sell to
the takers who read them. Over 62 settled films (Apr-Nov 2026, ~/gnome-research-work/rt/outlets/maker_skew.py, trades
only, held to settlement) quoting the signal's side until the next review beat pulling for 30 minutes by $40k (+5.8
standard errors, better on 32 films, worse on 7); about half of that is from not quoting both sides after the wave
and half from the side the signal picks (the opposite side gains nothing).

Which strikes pay comes from the registry's incentive programs (Kalshi's, synced every 15 minutes), re-read every
programs_reload_seconds; with rewarded_only off, every strike is quoted.

A live book shows our own resting orders and a paper session's doesn't: own_orders_in_book says which, so the best
price other makers show is read correctly in both (set it false for paper).

The session never ends itself: the launcher stops it when the film's markets close. Positions are read from the
engine, so a restarted session picks up what it holds.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

from gnomepy import ExecutionReport, Intent, OrderStatus, Scales, Strategy
from gnomepy.incentives import IncentiveTimeline, load_incentive_timelines
from gnomepy.java.schemas import StreamRecord
from gnomepy.registry import RegistryClient

from gnomepy_research.models.rt.review_feed import REVIEWS, SCORES, ReviewFeed
from gnomepy_research.sessions.rt_maker.quotes import (NONE, PULL, Level, Quote, ReviewState, adds, held_cost, lean,
                                                       one_side, quote, reducing, review_state)

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
MINUTE_NS = 60_000_000_000
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
VENUE = "kalshi"
HOUR_NS = 60 * MINUTE_NS
PULL_MODE, SKEW_MODE = "pull", "skew"
# Why a strike's quote is what it is, in the quotes metric.
QUOTING, PULLED, NOT_REWARDED, CAPPED, LEANING = 0, 1, 2, 3, 4


@dataclass
class _Strike:
    strike: int
    listing_id: int
    exchange_id: int
    security_id: int
    lot: float
    min_size: float
    bids: list[Level] = field(default_factory=list)
    asks: list[Level] = field(default_factory=list)
    # The quote last sent, which the OMS keeps resting until told otherwise; None once an order may have ended.
    sent: Quote | None = None


class RtMaker(Strategy):
    """
    Args:
        film_id: the film, as rt_films and rt_reviews name it.
        legs: one {venue, strike, yes_listing, no_listing} per strike per venue, as the launcher sets them; only
            Kalshi's YES listings are quoted.
        quote_size: contracts rested on each side.
        max_position: most YES contracts held, or sold, on one strike.
        max_film_usd: most money the film's positions and quotes may commit.
        pull_minutes: how long every quote stays pulled after a review is seen (one the signal gives a side doesn't
            pull in skew mode) or the Tomatometer's count changes.
        signal_mode: "pull" or "skew", what a review does (above).
        baseline, threshold, min_reviews: the signal's, as rt_review_wave's: the fresh share it is measured from, how
            far from it a side needs, and the reviews needed for any side.
        skew_hours: how long a review's side is quoted alone; None quotes it until the next review.
        pull_every_review: in skew mode, pull both sides for pull_minutes after every review, even one with a side
            (the side is quoted alone after the pull): the first minutes of a wave are when the market overshoots.
        min_price, max_price: a side priced outside them isn't quoted: the strike is all but settled.
        min_join_size: least size of others' a level needs to be joined.
        own_orders_in_book: whether the book this session reads shows its own orders (live) or not (paper).
        rewarded_only: quote only strikes with a reward program in force.
        reload_seconds: how often to re-read the film's reviews and Tomatometer (0 = once, for replays).
        programs_reload_seconds: how often to re-read the reward programs (0 = once).
        reviews_source, scores_source: the rt_reviews and rt_scores tables, or parquet extracts of them.
    """

    def __init__(
        self,
        film_id: str,
        legs: list[dict],
        quote_size: float = 500.0,
        max_position: float = 500.0,
        max_film_usd: float = 10_000.0,
        pull_minutes: float = 30.0,
        signal_mode: str = PULL_MODE,
        baseline: float = 0.667,
        threshold: float = 0.10,
        min_reviews: int = 3,
        skew_hours: float | None = None,
        pull_every_review: bool = False,
        min_price: float = 0.03,
        max_price: float = 0.97,
        min_join_size: float = 1.0,
        own_orders_in_book: bool = True,
        rewarded_only: bool = True,
        book_depth: int = 10,
        processing_time_ns: int = 0,
        reload_seconds: float = 5.0,
        programs_reload_seconds: float = 600.0,
        reviews_source: str = REVIEWS,
        scores_source: str = SCORES,
    ):
        kalshi = [leg for leg in legs if leg["venue"] == VENUE]
        if not kalshi:
            raise ValueError(f"legs name no {VENUE} strikes")
        if not 0 < min_price < max_price < 1:
            raise ValueError(f"need 0 < min_price < max_price < 1, got {min_price}, {max_price}")
        if quote_size <= 0 or max_position <= 0:
            raise ValueError("quote_size and max_position must be positive")
        if signal_mode not in (PULL_MODE, SKEW_MODE):
            raise ValueError(f"signal_mode must be {PULL_MODE!r} or {SKEW_MODE!r}, got {signal_mode!r}")
        if skew_hours is not None and skew_hours <= 0:
            raise ValueError(f"skew_hours must be positive or None, got {skew_hours}")
        self.film_id = film_id
        self.quote_size = quote_size
        self.max_position = max_position
        self.max_film_usd = max_film_usd
        self.pull_ns = int(pull_minutes * MINUTE_NS)
        self.signal_mode = signal_mode
        self.baseline, self.threshold, self.min_reviews = baseline, threshold, min_reviews
        self.skew_ns = None if skew_hours is None else int(skew_hours * HOUR_NS)
        self.pull_every_review = pull_every_review
        self._state_key: tuple[int, int] | None = None
        self._state: ReviewState | None = None
        self._mode: str | int | None = None
        self.min_price, self.max_price = min_price, max_price
        self.min_join_size = min_join_size
        self.own_orders_in_book = own_orders_in_book
        self.rewarded_only = rewarded_only
        self.book_depth = book_depth
        self._processing_time_ns = processing_time_ns
        self._quotes_buf = None
        self._pulled_until: int | None = None

        self._registry = RegistryClient()
        self._strikes = sorted((self._resolve(int(leg["strike"]), int(leg["yes_listing"])) for leg in kalshi),
                               key=lambda s: s.strike)
        self._strike_of = {s.listing_id: s for s in self._strikes}
        self._programs: dict[int, IncentiveTimeline] = {}
        if rewarded_only:
            self._programs = self._load_programs()
            if programs_reload_seconds > 0:
                threading.Thread(target=self._reload_programs, args=(programs_reload_seconds,),
                                 name=f"rt-programs-{film_id}", daemon=True).start()
        self._feed = ReviewFeed(film_id, reviews_source, reload_seconds, scores_source)

    def _resolve(self, strike: int, listing_id: int) -> _Strike:
        listings = self._registry.get_listing(listing_id=listing_id)
        if not listings:
            raise ValueError(f"listing {listing_id} not found in registry")
        specs = self._registry.get_listing_spec(listing_id=listing_id)
        spec = specs[0] if specs else None
        lot = int(spec.lot_size) / SIZE_SCALE if spec and spec.lot_size else 1.0
        min_size = int(spec.min_size) / SIZE_SCALE if spec and spec.min_size else 0.0
        return _Strike(strike, listing_id, listings[0].exchange_id, listings[0].security_id, lot, min_size)

    def _load_programs(self) -> dict[int, IncentiveTimeline]:
        return load_incentive_timelines(self._registry, [s.listing_id for s in self._strikes])

    def _reload_programs(self, every: float) -> None:
        while True:
            time.sleep(every)
            try:
                self._programs = self._load_programs()
            except Exception:
                # A failed refresh keeps the programs already held rather than stopping the session.
                logger.exception("reloading the reward programs of %s failed", self.film_id)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def register_metrics(self) -> None:
        quotes = self.metrics.create_stream("rt_maker_quotes")
        self._q_ts = quotes.add_long_column("timestamp")
        self._q_listing = quotes.add_long_column("listing_id")
        self._q_strike = quotes.add_long_column("strike")
        self._q_bid = quotes.add_double_column("bid_price")
        self._q_bid_size = quotes.add_double_column("bid_size")
        self._q_ask = quotes.add_double_column("ask_price")
        self._q_ask_size = quotes.add_double_column("ask_size")
        self._q_position = quotes.add_double_column("position")
        self._q_reason = quotes.add_long_column("reason")
        self._q_signal = quotes.add_double_column("signal")
        quotes.freeze()
        self._quotes_buf = quotes

    # ---- state ---------------------------------------------------------------------------------

    def _position(self, s: _Strike) -> tuple[float, float]:
        """(YES contracts held, negative when sold; their average price)."""
        pos = self.positions.get_position(s.exchange_id, s.security_id)
        if pos is None or not pos.net_quantity:
            return 0.0, 0.0
        return pos.net_quantity / SIZE_SCALE, pos.avg_entry_price / PRICE_SCALE

    def _rewarded(self, s: _Strike, now_ns: int) -> bool:
        if not self.rewarded_only:
            return True
        timeline = self._programs.get(s.listing_id)
        return bool(timeline and timeline.in_force(now_ns))

    def _pull_until(self, now_ns: int) -> int | None:
        """The end of the pull the latest review or Tomatometer change started, as seen by now."""
        known = self._feed.known(now_ns)
        seen = [t for t in (known[-1].seen_ns if known else None, self._feed.count_changed(now_ns)) if t is not None]
        return max(seen) + self.pull_ns if seen else None

    def _review_state(self, now_ns: int) -> ReviewState | None:
        """The latest review's state, recomputed only when a review has been seen since the last call."""
        known = self._feed.known(now_ns)
        key = (len(known), known[-1].seen_ns) if known else (0, 0)
        if key != self._state_key:
            self._state_key = key
            self._state = review_state(known, self.baseline, self.threshold, self.min_reviews)
        return self._state

    def _skew(self, now_ns: int) -> tuple[bool, int, float]:
        """(pulled, the side to quote alone or 0, the latest signal) in skew mode."""
        state = self._review_state(now_ns)
        mode = lean(state, now_ns, self.pull_ns, self.skew_ns, self.pull_every_review)
        if mode != self._mode:
            logger.info("film %s: quoting %s (signal %s)", self.film_id, mode,
                        "none" if state is None else f"{state.signal.signal:+.3f} from {state.signal.reviews} reviews")
            self._mode = mode
        changed = self._feed.count_changed(now_ns)
        counting = changed is not None and now_ns < changed + self.pull_ns
        side = mode if isinstance(mode, int) else 0
        return mode == PULL or counting, side, math.nan if state is None else state.signal.signal

    # ---- engine callbacks ---------------------------------------------------------------------

    def on_market_data(self, data: StreamRecord) -> list[Intent]:
        s = self._strike_of.get(self.streams.listing_id(data.stream_id))
        if s is not None:
            s.bids = [Level(data.bid_price(i) / PRICE_SCALE, data.bid_size(i) / SIZE_SCALE)
                      for i in range(self.book_depth) if data.bid_size(i) > 0]
            s.asks = [Level(data.ask_price(i) / PRICE_SCALE, data.ask_size(i) / SIZE_SCALE)
                      for i in range(self.book_depth) if data.ask_size(i) > 0]
        return self._evaluate(data.event_timestamp)

    def on_stream_event(self, stream_id: int, record: StreamRecord) -> list[Intent]:
        # Another stream's records only wake the session, so a review is acted on between this film's book updates.
        return self._evaluate(record.event_timestamp)

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        s = self._strike_of.get(self.streams.listing_id_for(report.exchange_id, report.security_id))
        if s is not None and report.order_status in TERMINAL:
            # One of the strike's orders is gone, so the quote is sent again at the next decision.
            s.sent = None
        return []

    def _evaluate(self, now_ns: int) -> list[Intent]:
        if self.is_exiting:
            return []
        side, sig = 0, math.nan
        if self.signal_mode == SKEW_MODE:
            pulled, side, sig = self._skew(now_ns)
        else:
            until = self._pull_until(now_ns)
            pulled = until is not None and now_ns < until
            if pulled and until != self._pulled_until:
                logger.info("film %s: a review or a Tomatometer change; quotes pulled for %d min", self.film_id,
                            self.pull_ns // MINUTE_NS)
            self._pulled_until = until
        positions = {s.listing_id: self._position(s) for s in self._strikes}
        budget = self.max_film_usd - sum(held_cost(*p) for p in positions.values())
        intents = []
        for s in self._strikes:
            position = positions[s.listing_id][0]
            reason = QUOTING
            if pulled:
                q, reason = NONE, PULLED
            elif not self._rewarded(s, now_ns):
                q, reason = NONE, NOT_REWARDED
            else:
                q = quote(s.bids, s.asks, position, s.sent if self.own_orders_in_book else None, self.quote_size,
                          self.max_position, self.min_price, self.max_price, self.min_join_size, s.lot, s.min_size)
                if side:
                    q, reason = one_side(q, side), LEANING
                cost = adds(q, position)
                if cost > budget:
                    q, reason = reducing(q, position, s.lot, s.min_size), CAPPED
                    cost = adds(q, position)
                budget -= cost
            if q != s.sent:
                intents.append(self._send(now_ns, s, q, position, reason, sig))
        return intents

    def _send(self, now_ns: int, s: _Strike, q: Quote, position: float, reason: int, sig: float) -> Intent:
        s.sent = q
        self._record(now_ns, s, q, position, reason, sig)
        return Intent(s.exchange_id, s.security_id,
                      bid_price=round(q.bid_price * PRICE_SCALE) if q.bid_size else 0,
                      bid_size=round(q.bid_size * SIZE_SCALE),
                      ask_price=round(q.ask_price * PRICE_SCALE) if q.ask_size else 0,
                      ask_size=round(q.ask_size * SIZE_SCALE))

    def _record(self, now_ns: int, s: _Strike, q: Quote, position: float, reason: int, sig: float) -> None:
        if self._quotes_buf is None:
            return
        buf = self._quotes_buf
        buf.begin()
        buf.set_long(self._q_ts, now_ns)
        buf.set_long(self._q_listing, s.listing_id)
        buf.set_long(self._q_strike, s.strike)
        buf.set_double(self._q_bid, q.bid_price)
        buf.set_double(self._q_bid_size, q.bid_size)
        buf.set_double(self._q_ask, q.ask_price)
        buf.set_double(self._q_ask_size, q.ask_size)
        buf.set_double(self._q_position, position)
        buf.set_long(self._q_reason, reason)
        buf.set_double(self._q_signal, sig)
        buf.commit()
