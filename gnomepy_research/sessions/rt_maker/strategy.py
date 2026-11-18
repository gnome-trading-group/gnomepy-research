"""
A resting maker on films' Kalshi Rotten Tomatoes strikes, for Kalshi's liquidity rewards.

gnome-launcher (the `rt_maker` rule) packs the open films into as few sessions as a paper gateway can hold, each session
holding the YES listing of each of its films' Kalshi "at least X" strikes. Every film is quoted as if it had the session
to itself: its own reviews, pulls, lean and budget. On every strike with a liquidity-reward program in force, the
session rests quote_size contracts at the best bid and the best ask other makers show (quotes.py): a YES bid, and a YES
ask, which Kalshi books as a NO bid, so both sides of the reward's scoring are covered. A side stops once the strike's
position reaches max_position, and quotes that would commit more of the film's max_film_usd than is left are cut back
to the side that trades the position back towards flat. Positions are held; nothing is ever taken. A film past its
markets' close is no longer quoted; the launcher drops it at the session's next relaunch.

Fills are where the maker loses, so a side can also keep away from them (quotes.py): queue_policy "requeue" pulls a
side its estimated queue place puts near the front and sends it again from the back, "behind" rests each side a tick
behind the price it would join, and inventory skew steps the side that adds to a position back a tick per skew_unit
contracts held. All are off by default ("join", no skew).

A book update decides its own film's quotes. Every record also decides the quotes of any other film whose review state
moved (a review or Tomatometer change seen, a pull or lean ending, its close passing) or one of whose orders ended, so a
film reacts at the next record of any of the session's strikes, never waiting for its own book to move. A strike whose
order was rejected is quoted again only REJECT_BACKOFF_NS later.

Kalshi pays the rewards for resting size near the touch on both sides; the risk is being filled by a taker who knows
something the book hasn't priced. On these markets that is mostly a critic's review (gnome-pipelines' rt_reviews,
re-read every reload_seconds), and signal_mode says what a review does:
- pull: for pull_minutes after any review, every quote is pulled, then both sides come back.
- skew: the review signal (models/rt/signal.py: Claude's fresh share against the baseline) leans the quotes. A review
  with a side quotes only that side, the YES bid for YES and the YES ask for NO, for skew_hours (None: until the next
  review); a review without a side, or one that flips the last side leaned, pulls both for pull_minutes first. With
  pull_every_review, every review pulls both for pull_minutes, and the side is quoted alone only after that.
In both, a change in the Tomatometer's count (rt_scores) pulls every quote for pull_minutes.

The reviews the feed sees don't explain most of the maker's losses: they come in a few hours in which a film's whole
ladder reprices at once, on news the feed never shows (backtests of 14 films over a week: the worst 5% of film-hours
held 84-98% of the 1 h markout losses, and none had a review in the feed near it). With guard_ticks set,
any strike whose mid moves guard_ticks or more within guard_seconds pulls every quote of its film for guard_minutes,
so the first strike to move warns its siblings.

Why skew: after a film's reviews arrive, Kalshi under-reacts in their direction, so both sides at the touch sell to
the takers who read them. Over 62 settled films (Apr-Nov 2026, ~/gnome-research-work/rt/outlets/maker_skew.py, trades
only, held to settlement) quoting the signal's side until the next review beat pulling for 30 minutes by $40k (+5.8
standard errors, better on 32 films, worse on 7); about half of that is from not quoting both sides after the wave
and half from the side the signal picks (the opposite side gains nothing).

Which strikes pay comes from the registry's incentive programs (Kalshi's, synced every 15 minutes), re-read every
programs_reload_seconds, PROGRAMS_BATCH listings a request so a session of many films keeps each request small; with
rewarded_only off, every strike is quoted.

A live book shows our own resting orders and a paper session's doesn't: own_orders_in_book says which, so the best
price other makers show is read correctly in both (set it false for paper).

The session never ends itself: the launcher stops it once none of its films is open, or relaunches it when the set of
films changes. Positions are read from the engine, so a restarted session picks up what it holds; each strike's is
read again after its fills and otherwise at most every POSITION_REFRESH_NS, as a book update decides every strike of
its film.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import pandas as pd
from gnomepy import ExecutionReport, Intent, OrderStatus, Scales, Strategy
from gnomepy.incentives import IncentiveTimeline, load_incentive_timelines
from gnomepy.java.schemas import StreamRecord
from gnomepy.registry import RegistryClient

from gnomepy_research.models.rt.review_feed import REVIEWS, SCORES, ReviewFeed
from gnomepy_research.sessions.rt_maker.quotes import (BEHIND, JOIN, NONE, PULL, QUEUE_POLICIES, REQUEUE, Level, Quote,
                                                       QueueSpot, ReviewState, adds, advance, held_cost, joined, lean,
                                                       near_front, one_side, others_at, placed, quote, reducing,
                                                       review_state, skew_ticks)

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
MINUTE_NS = 60_000_000_000
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
VENUE = "kalshi"
HOUR_NS = 60 * MINUTE_NS
PULL_MODE, SKEW_MODE = "pull", "skew"
# Why a strike's quote is what it is, in the quotes metric.
QUOTING, PULLED, NOT_REWARDED, CAPPED, LEANING, CLOSED, REQUEUED, GUARDED = 0, 1, 2, 3, 4, 5, 6, 7
# Listings per reward-programs request: a request returns every version of each listing's programs.
PROGRAMS_BATCH = 25
# How long a strike whose order was rejected waits before it is quoted again. A refusal that would recur (a gateway
# cap, a halt) resent at once loops at the record rate, and every round is ledger events the session must write.
REJECT_BACKOFF_NS = 5_000_000_000
FILLS = {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}
# How stale a strike's position may be between its fills. A book update decides every strike of its film, so reading
# each from the engine every time is the session's main cost; live, a correction booked outside the session shows up
# this much later.
POSITION_REFRESH_NS = 1_000_000_000
NEVER = 1 << 62


@dataclass
class _Strike:
    strike: int
    listing_id: int
    exchange_id: int
    security_id: int
    lot: float
    min_size: float
    tick: float
    bids: list[Level] = field(default_factory=list)
    asks: list[Level] = field(default_factory=list)
    # The quote last sent, which the OMS keeps resting until told otherwise; None once an order may have ended.
    sent: Quote | None = None
    # Not quoted again before this, after a rejected order.
    retry_at_ns: int = 0
    # Each resting side's estimated place in its level's queue; None while it isn't resting.
    bid_spot: QueueSpot | None = None
    ask_spot: QueueSpot | None = None
    # When each side was last pulled to be sent again from the back.
    bid_requeued_ns: int = -(1 << 62)
    ask_requeued_ns: int = -(1 << 62)
    # (YES contracts held, their average price) as last read, and when; None until read and after a fill.
    position: tuple[float, float] = (0.0, 0.0)
    position_ns: int | None = None
    # Whether a reward program was in force, which programs said so, and until when that holds.
    rewarded: bool = False
    rewarded_timeline: IncentiveTimeline | None = None
    rewarded_until_ns: int = -1
    # (when, mid) at each change of its mid over the guard's lookback, oldest first; the first is the mid as of the
    # lookback's start once the history reaches back that far.
    mids: deque = field(default_factory=deque)


@dataclass
class _Film:
    film_id: str
    close_ns: int | None
    strikes: list[_Strike]
    pulled_until: int | None = None
    state_key: tuple[int, int] | None = None
    state: ReviewState | None = None
    mode: str | int | None = None
    # (closed, pulled, guarded, side) when its quotes were last decided; a change decides them again.
    regime: tuple[bool, bool, bool, int] | None = None
    # One of its orders ended since its quotes were last decided.
    stale: bool = True
    # Every quote stays pulled until then, after one of its strikes moved too far too fast.
    guarded_until: int = 0


class RtMaker(Strategy):
    """
    Args:
        films: one {film_id, close, strikes} per film, as the launcher sets them: the film as rt_films and rt_reviews
            name it, its markets' close (ISO-8601, or None), and its Kalshi strikes as {strike: YES listing id}. Kept
            that small because a session's whole config travels in its host's user data, which EC2 caps at 16 KB.
        quote_size: contracts rested on each side.
        max_position: most YES contracts held, or sold, on one strike.
        max_film_usd: most money each film's positions and quotes may commit.
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
        queue_policy: "join" (rest where the side joins), "requeue" (pull a side near the front of its queue and send
            it again from the back) or "behind" (rest each side a tick behind the price it would join).
        requeue_min_ahead, requeue_ahead_fraction: requeue a side once less than requeue_min_ahead contracts, or less
            than requeue_ahead_fraction of everyone else's size at its level, is estimated ahead of it.
        assume_cancel_ahead: the share (0 to 1) of a level's shrinks that weren't trades taken to be cancels ahead of
            our side. 1 errs towards the front, so a side is requeued early rather than filled.
        requeue_cooldown_seconds: least time between two requeues of one side.
        max_requeues_per_second: requeues the whole session may make a second, so they stay inside its order limits.
        skew_unit, max_skew_ticks: the side that adds to a strike's position rests a tick further back per skew_unit
            contracts held, at most max_skew_ticks (0 for either: no skew).
        guard_ticks, guard_seconds, guard_minutes: a strike's mid moving guard_ticks ticks or more within
            guard_seconds pulls every quote of its film for guard_minutes (0 ticks: no guard).
        own_orders_in_book: whether the book this session reads shows its own orders (live) or not (paper).
        rewarded_only: quote only strikes with a reward program in force.
        reload_seconds: how often to re-read the films' reviews and Tomatometers (0 = once, for replays).
        programs_reload_seconds: how often to re-read the reward programs (0 = once).
        reviews_source, scores_source: the rt_reviews and rt_scores tables, or parquet extracts of them.
    """

    def __init__(
        self,
        films: list[dict],
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
        queue_policy: str = JOIN,
        requeue_min_ahead: float = 0.0,
        requeue_ahead_fraction: float = 0.0,
        assume_cancel_ahead: float = 1.0,
        requeue_cooldown_seconds: float = 5.0,
        max_requeues_per_second: float = 10.0,
        skew_unit: float = 0.0,
        max_skew_ticks: int = 0,
        guard_ticks: int = 0,
        guard_seconds: float = 60.0,
        guard_minutes: float = 5.0,
        own_orders_in_book: bool = True,
        rewarded_only: bool = True,
        book_depth: int = 10,
        processing_time_ns: int = 0,
        reload_seconds: float = 5.0,
        programs_reload_seconds: float = 600.0,
        reviews_source: str = REVIEWS,
        scores_source: str = SCORES,
    ):
        if not films:
            raise ValueError("films is empty")
        if len({f["film_id"] for f in films}) != len(films):
            raise ValueError("a film is named twice")
        for f in films:
            if not f["strikes"]:
                raise ValueError(f"film {f['film_id']} has no {VENUE} strikes")
        if not 0 < min_price < max_price < 1:
            raise ValueError(f"need 0 < min_price < max_price < 1, got {min_price}, {max_price}")
        if quote_size <= 0 or max_position <= 0:
            raise ValueError("quote_size and max_position must be positive")
        if signal_mode not in (PULL_MODE, SKEW_MODE):
            raise ValueError(f"signal_mode must be {PULL_MODE!r} or {SKEW_MODE!r}, got {signal_mode!r}")
        if skew_hours is not None and skew_hours <= 0:
            raise ValueError(f"skew_hours must be positive or None, got {skew_hours}")
        if queue_policy not in QUEUE_POLICIES:
            raise ValueError(f"queue_policy must be one of {QUEUE_POLICIES}, got {queue_policy!r}")
        if not 0 <= assume_cancel_ahead <= 1 or not 0 <= requeue_ahead_fraction <= 1:
            raise ValueError("assume_cancel_ahead and requeue_ahead_fraction must be within [0, 1]")
        if requeue_min_ahead < 0 or skew_unit < 0 or max_skew_ticks < 0 or max_requeues_per_second <= 0:
            raise ValueError("requeue_min_ahead, skew_unit and max_skew_ticks must not be negative, and "
                             "max_requeues_per_second must be positive")
        if guard_ticks < 0 or guard_seconds <= 0 or guard_minutes < 0:
            raise ValueError("guard_ticks and guard_minutes must not be negative, and guard_seconds must be positive")
        self.quote_size = quote_size
        self.max_position = max_position
        self.max_film_usd = max_film_usd
        self.pull_ns = int(pull_minutes * MINUTE_NS)
        self.signal_mode = signal_mode
        self.baseline, self.threshold, self.min_reviews = baseline, threshold, min_reviews
        self.skew_ns = None if skew_hours is None else int(skew_hours * HOUR_NS)
        self.pull_every_review = pull_every_review
        self.min_price, self.max_price = min_price, max_price
        self.min_join_size = min_join_size
        self.queue_policy = queue_policy
        self.requeue_min_ahead, self.requeue_ahead_fraction = requeue_min_ahead, requeue_ahead_fraction
        self.assume_cancel_ahead = assume_cancel_ahead
        self.requeue_cooldown_ns = int(requeue_cooldown_seconds * 1e9)
        self.max_requeues_per_second = max_requeues_per_second
        self._requeue_tokens, self._tokens_ns = max_requeues_per_second, 0
        self.skew_unit, self.max_skew_ticks = skew_unit, max_skew_ticks
        self.guard_ticks = guard_ticks
        self.guard_lookback_ns = int(guard_seconds * 1e9)
        self.guard_hold_ns = int(guard_minutes * MINUTE_NS)
        self.own_orders_in_book = own_orders_in_book
        self.rewarded_only = rewarded_only
        self.book_depth = book_depth
        self._processing_time_ns = processing_time_ns
        self._quotes_buf = None
        self._now_ns = 0

        self._registry = RegistryClient()
        self._films = [_Film(f["film_id"], pd.Timestamp(f["close"]).value if f.get("close") else None,
                             sorted((self._resolve(int(strike), int(listing_id))
                                     for strike, listing_id in f["strikes"].items()), key=lambda s: s.strike))
                       for f in films]
        self._strike_of: dict[int, _Strike] = {}
        self._film_of: dict[int, _Film] = {}
        for film in self._films:
            for s in film.strikes:
                if s.listing_id in self._strike_of:
                    raise ValueError(f"listing {s.listing_id} is in two films")
                self._strike_of[s.listing_id] = s
                self._film_of[s.listing_id] = film
        self._programs: dict[int, IncentiveTimeline] = {}
        if rewarded_only:
            self._programs = self._load_programs()
            if programs_reload_seconds > 0:
                threading.Thread(target=self._reload_programs, args=(programs_reload_seconds,),
                                 name="rt-programs", daemon=True).start()
        self._feed = ReviewFeed([f.film_id for f in self._films], reviews_source, reload_seconds, scores_source)

    def _resolve(self, strike: int, listing_id: int) -> _Strike:
        listings = self._registry.get_listing(listing_id=listing_id)
        if not listings:
            raise ValueError(f"listing {listing_id} not found in registry")
        specs = self._registry.get_listing_spec(listing_id=listing_id)
        spec = specs[0] if specs else None
        lot = int(spec.lot_size) / SIZE_SCALE if spec and spec.lot_size else 1.0
        min_size = int(spec.min_size) / SIZE_SCALE if spec and spec.min_size else 0.0
        tick = int(spec.tick_size) / PRICE_SCALE if spec and spec.tick_size else 0.01
        return _Strike(strike, listing_id, listings[0].exchange_id, listings[0].security_id, lot, min_size, tick)

    def _load_programs(self) -> dict[int, IncentiveTimeline]:
        ids = list(self._strike_of)
        programs: dict[int, IncentiveTimeline] = {}
        for i in range(0, len(ids), PROGRAMS_BATCH):
            programs.update(load_incentive_timelines(self._registry, ids[i:i + PROGRAMS_BATCH]))
        return programs

    def _reload_programs(self, every: float) -> None:
        while True:
            time.sleep(every)
            try:
                self._programs = self._load_programs()
            except Exception:
                # A failed refresh keeps the programs already held rather than stopping the session.
                logger.exception("reloading the reward programs of %d listings failed", len(self._strike_of))

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
        self._q_bid_ahead = quotes.add_double_column("bid_ahead")
        self._q_ask_ahead = quotes.add_double_column("ask_ahead")
        quotes.freeze()
        self._quotes_buf = quotes

    # ---- state ---------------------------------------------------------------------------------

    def _position(self, s: _Strike) -> tuple[float, float]:
        """(YES contracts held, negative when sold; their average price)."""
        if s.position_ns is None or self._now_ns - s.position_ns >= POSITION_REFRESH_NS:
            pos = self.positions.get_position(s.exchange_id, s.security_id)
            s.position = ((0.0, 0.0) if pos is None or not pos.net_quantity
                          else (pos.net_quantity / SIZE_SCALE, pos.avg_entry_price / PRICE_SCALE))
            s.position_ns = self._now_ns
        return s.position

    def _rewarded(self, s: _Strike, now_ns: int) -> bool:
        if not self.rewarded_only:
            return True
        timeline = self._programs.get(s.listing_id)
        if timeline is not s.rewarded_timeline or now_ns >= s.rewarded_until_ns:
            s.rewarded = bool(timeline and timeline.in_force(now_ns))
            s.rewarded_timeline, s.rewarded_until_ns = timeline, _next_program_change(timeline, now_ns)
        return s.rewarded

    def _pull_until(self, film: _Film, now_ns: int) -> int | None:
        """The end of the pull the film's latest review or Tomatometer change started, as seen by now."""
        known = self._feed.known(film.film_id, now_ns)
        seen = [t for t in (known[-1].seen_ns if known else None, self._feed.count_changed(film.film_id, now_ns))
                if t is not None]
        return max(seen) + self.pull_ns if seen else None

    def _review_state(self, film: _Film, now_ns: int) -> ReviewState | None:
        """The film's latest review's state, recomputed only when a review has been seen since the last call."""
        known = self._feed.known(film.film_id, now_ns)
        key = (len(known), known[-1].seen_ns) if known else (0, 0)
        if key != film.state_key:
            film.state_key = key
            film.state = review_state(known, self.baseline, self.threshold, self.min_reviews)
        return film.state

    def _skew(self, film: _Film, now_ns: int) -> tuple[bool, int, float]:
        """(pulled, the side to quote alone or 0, the latest signal) of a film in skew mode."""
        state = self._review_state(film, now_ns)
        mode = lean(state, now_ns, self.pull_ns, self.skew_ns, self.pull_every_review)
        if mode != film.mode:
            logger.info("film %s: quoting %s (signal %s)", film.film_id, mode,
                        "none" if state is None else f"{state.signal.signal:+.3f} from {state.signal.reviews} reviews")
            film.mode = mode
        changed = self._feed.count_changed(film.film_id, now_ns)
        counting = changed is not None and now_ns < changed + self.pull_ns
        side = mode if isinstance(mode, int) else 0
        return mode == PULL or counting, side, math.nan if state is None else state.signal.signal

    def _pull(self, film: _Film, now_ns: int) -> bool:
        """Whether a film is pulled in pull mode."""
        until = self._pull_until(film, now_ns)
        pulled = until is not None and now_ns < until
        if pulled and until != film.pulled_until:
            logger.info("film %s: a review or a Tomatometer change; quotes pulled for %d min", film.film_id,
                        self.pull_ns // MINUTE_NS)
        film.pulled_until = until
        return pulled

    # ---- engine callbacks ---------------------------------------------------------------------

    def on_market_data(self, data: StreamRecord) -> list[Intent]:
        listing_id = self.streams.listing_id(data.stream_id)
        s = self._strike_of.get(listing_id)
        if s is not None:
            s.bids = _levels(data.bid_price, data.bid_size, self.book_depth)
            s.asks = _levels(data.ask_price, data.ask_size, self.book_depth)
            self._advance(s, data)
            self._watch(s, self._film_of[listing_id], data.event_timestamp)
        return self._decide(data.event_timestamp, self._film_of.get(listing_id))

    def _watch(self, s: _Strike, film: _Film, now_ns: int) -> None:
        """Guards the strike's film when its mid has moved guard_ticks or more since guard_seconds ago."""
        if not self.guard_ticks or not s.bids or not s.asks:
            return
        mid = (s.bids[0].price + s.asks[0].price) / 2
        history = s.mids
        if history and history[-1][1] == mid:
            return
        history.append((now_ns, mid))
        start = now_ns - self.guard_lookback_ns
        while len(history) > 1 and history[1][0] <= start:
            history.popleft()
        then_ns, then = history[0]
        if then_ns <= start and abs(mid - then) >= self.guard_ticks * s.tick - 1e-9:
            if now_ns >= film.guarded_until:
                logger.info("film %s: strike %d moved %+.2f in %ds; every quote pulled for %d min", film.film_id,
                            s.strike, mid - then, self.guard_lookback_ns // 1_000_000_000,
                            self.guard_hold_ns // MINUTE_NS)
            film.guarded_until = max(film.guarded_until, now_ns + self.guard_hold_ns)

    def _advance(self, s: _Strike, data: StreamRecord) -> None:
        """Moves each resting side's queue place by the record: a trade print at its price, and its level's size."""
        if s.bid_spot is None and s.ask_spot is None:
            return
        trade = data.action == "Trade"
        price = data.price / PRICE_SCALE if trade else math.nan
        size = data.size / SIZE_SCALE if trade else 0.0
        for spot, levels in ((s.bid_spot, s.bids), (s.ask_spot, s.asks)):
            if spot is None:
                continue
            traded = size if trade and math.isclose(price, spot.price) else 0.0
            others = others_at(levels, spot.price, spot.size if self.own_orders_in_book else 0.0)
            advance(spot, data.event_timestamp, others, traded, self.assume_cancel_ahead)

    def on_stream_event(self, stream_id: int, record: StreamRecord) -> list[Intent]:
        # Another stream's records only wake the session, so a review is acted on between the films' book updates.
        return self._decide(record.event_timestamp, None)

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        listing_id = self.streams.listing_id_for(report.exchange_id, report.security_id)
        s = self._strike_of.get(listing_id)
        if s is None:
            return []
        if report.order_status in FILLS and report.filled_qty:
            s.position_ns = None
            # Reports don't say which side; the fill's price does, as our bid and ask never share one.
            price = report.fill_price / PRICE_SCALE
            for name in ("bid_spot", "ask_spot"):
                spot = getattr(s, name)
                if spot is not None and math.isclose(spot.price, price):
                    if report.order_status == OrderStatus.FILLED:
                        setattr(s, name, None)
                    else:
                        spot.ahead, spot.size = 0.0, report.leaves_qty / SIZE_SCALE
        if report.order_status in TERMINAL:
            # One of the strike's orders is gone, so the quote is sent again at the film's next decision.
            s.sent = None
            self._film_of[listing_id].stale = True
            if report.order_status == OrderStatus.REJECTED:
                s.retry_at_ns = self._now_ns + REJECT_BACKOFF_NS
                s.bid_spot = s.ask_spot = None
        return []

    def _decide(self, now_ns: int, touched: _Film | None) -> list[Intent]:
        """The quotes of the film whose book moved, and of every film whose regime moved or whose orders ended."""
        if self.is_exiting:
            return []
        self._now_ns = max(self._now_ns, now_ns)
        intents = []
        for film in self._films:
            closed = film.close_ns is not None and now_ns >= film.close_ns
            side, sig = 0, math.nan
            if self.signal_mode == SKEW_MODE:
                pulled, side, sig = self._skew(film, now_ns)
            else:
                pulled = self._pull(film, now_ns)
            guarded = now_ns < film.guarded_until
            regime = (closed, pulled, guarded, side)
            if film is touched or film.stale or regime != film.regime:
                film.regime, film.stale = regime, False
                intents += self._quote_film(film, now_ns, closed, pulled, guarded, side, sig)
        return intents

    def _quote_film(self, film: _Film, now_ns: int, closed: bool, pulled: bool, guarded: bool, side: int,
                    sig: float) -> list[Intent]:
        positions = {s.listing_id: self._position(s) for s in film.strikes}
        budget = self.max_film_usd - sum(held_cost(*p) for p in positions.values())
        intents = []
        for s in film.strikes:
            position = positions[s.listing_id][0]
            reason = QUOTING
            if closed:
                q, reason = NONE, CLOSED
            elif pulled or guarded:
                q, reason = NONE, PULLED if pulled else GUARDED
            elif not self._rewarded(s, now_ns):
                q, reason = NONE, NOT_REWARDED
            else:
                q = quote(s.bids, s.asks, position, s.sent if self.own_orders_in_book else None, self.quote_size,
                          self.max_position, self.min_price, self.max_price, self.min_join_size, s.lot, s.min_size)
                if side:
                    q, reason = one_side(q, side), LEANING
                q = placed(q, position, s.tick, 1 if self.queue_policy == BEHIND else 0,
                           skew_ticks(position, self.skew_unit, self.max_skew_ticks), self.min_price, self.max_price)
                cost = adds(q, position)
                if cost > budget:
                    q, reason = reducing(q, position, s.lot, s.min_size), CAPPED
                    cost = adds(q, position)
                budget -= cost
                if self.queue_policy == REQUEUE:
                    q, requeued = self._requeue(s, q, now_ns)
                    if requeued:
                        reason = REQUEUED
            if q != s.sent:
                if now_ns < s.retry_at_ns:
                    # Decided again at every record until it may be sent.
                    film.stale = True
                    continue
                intents.append(self._send(now_ns, s, q, position, reason, sig))
        return intents

    def _requeue(self, s: _Strike, q: Quote, now_ns: int) -> tuple[Quote, bool]:
        """The quote with any side near the front of its queue pulled, so the next decision sends it from the back."""
        self._requeue_tokens = min(self.max_requeues_per_second, self._requeue_tokens + (now_ns - self._tokens_ns)
                                   * self.max_requeues_per_second / 1e9)
        self._tokens_ns = now_ns
        bid, ask, requeued = (q.bid_price, q.bid_size), (q.ask_price, q.ask_size), False
        for name, (price, size) in (("bid", bid), ("ask", ask)):
            spot = getattr(s, f"{name}_spot")
            if (not size or spot is None or not math.isclose(spot.price, price) or self._requeue_tokens < 1
                    or now_ns < getattr(s, f"{name}_requeued_ns") + self.requeue_cooldown_ns
                    or not near_front(spot, self.requeue_min_ahead, self.requeue_ahead_fraction)):
                continue
            self._requeue_tokens -= 1
            setattr(s, f"{name}_requeued_ns", now_ns)
            setattr(s, f"{name}_spot", None)
            if name == "bid":
                bid = (0.0, 0.0)
            else:
                ask = (0.0, 0.0)
            requeued = True
        return Quote(bid[0], bid[1], ask[0], ask[1]), requeued

    def _place(self, s: _Strike, q: Quote) -> None:
        """Each side's queue place after sending q: kept where it rests at the same price no larger (a smaller size
        keeps its place), from the back of the level where it is new, moved or larger, gone where it isn't quoted."""
        for name, price, size, levels in (("bid_spot", q.bid_price, q.bid_size, s.bids),
                                          ("ask_spot", q.ask_price, q.ask_size, s.asks)):
            spot = getattr(s, name)
            if not size:
                setattr(s, name, None)
            elif spot is not None and math.isclose(spot.price, price) and size <= spot.size:
                spot.size = size
            else:
                setattr(s, name, joined(price, size, others_at(levels, price, 0.0)))

    def _send(self, now_ns: int, s: _Strike, q: Quote, position: float, reason: int, sig: float) -> Intent:
        s.sent = q
        self._place(s, q)
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
        buf.set_double(self._q_bid_ahead, s.bid_spot.ahead if s.bid_spot else math.nan)
        buf.set_double(self._q_ask_ahead, s.ask_spot.ahead if s.ask_spot else math.nan)
        buf.commit()


def _levels(price_at, size_at, depth: int) -> list[Level]:
    """A side of the book best first, to its first empty level, each level's size read once."""
    levels = []
    for i in range(depth):
        size = size_at(i)
        if size <= 0:
            break
        levels.append(Level(price_at(i) / PRICE_SCALE, size / SIZE_SCALE))
    return levels


def _next_program_change(timeline: IncentiveTimeline | None, now_ns: int) -> int:
    """When any of the timeline's programs next starts, ends or changes terms after now_ns: until then, which are in
    force can't change."""
    upcoming = [t for program in (timeline.programs if timeline else ()) for v in program.versions
                for t in (v.effective_from, v.period_start, v.period_end) if t > now_ns]
    return min(upcoming, default=NEVER)
