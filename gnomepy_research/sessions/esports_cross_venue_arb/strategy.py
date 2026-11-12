"""
Esports cross-venue arb (CS2, Dota 2, Valorant): buy both teams of one market on different venues when the pair costs
under $1 after fees.

A session holds one market - a match winner or one map's winner - with each venue's two team listings (Polymarket
International, Kalshi, Polymarket US). Nothing may be locked before play starts, because a match that is never played
settles differently by venue (Polymarket International pays 50/50, Kalshi and Polymarket US the last fair price),
which turns a locked pair into a loss. Two ways a session knows play has started:

- given the match's Polymarket event slug or Kalshi event ticker (gnome-launcher's esports_game rule launches it
  shortly before the scheduled start), it follows the venues' live feeds (game_state.py) and opens nothing until one
  says the match is live. A feed quiet for stale_seconds, or one saying the match isn't live after it was, stops new
  pairs until it says live again. Once one says the match is over it opens nothing more, sends the legs it was
  holding back and completes any unequal position, and ends itself once nothing is in flight and ended_settle_seconds
  have passed (a fill racing the end still arrives), or ended_max_seconds after the end whatever it holds. A match
  that ends without ever having been live (a forfeit or a cancellation: nothing was traded) ends it at once, and one
  never live never_live_hours after its scheduled start (a postponement no feed reports) ends it then;
- given recorded states (game_states), a replay follows them as of each event;
- given neither, it trades from its first callback: for backtests whose scenarios start once play has begun.

Branched from cs2_cross_venue_arb (which the HLTV-driven cs2 rule still launches once a match is live) to trade any
esports league the launcher scans: CS2, Dota 2, Valorant.

Completing an unequal position is never held back by the feed: an unhedged position is the one risk the strategy
exists not to hold. Each change of the match's state is recorded to the game_state metrics stream, so a replay of the
session can run the same gate (game_states: the recorded rows).

On every book update with nothing in flight, it finds the venue pair and size where holding team A on one venue and
team B on another clears min_edge per pair after each venue's taker fee, walking each book only as deep as every pair
still clears it, and sends both legs at once as IOC limits at the deepest price it costed (decision.py). A team is
held by buying its listing, or on a venue in sell_rival_venues by selling the rival's listing, which Kalshi books as
NO on it.

The legs land at different venues at different times, so they can come back unequal. Once neither is in flight, any
difference between what pays if team A wins and what pays if team B wins is completed by buying the short team
wherever it is cheapest, no dearer than its best level plus completion_slippage, retried on later updates until the
two match. A completion goes to the venues that match a taker order at once, and to a delayed one (Polymarket
International) only when they have nothing within the cap. One that comes back unfilled is retried no sooner than
completion_retry_seconds later, and an order the venue rejected is never sent again unchanged: the reject would only
repeat, once per book update. Completion never waits for the cooldown and never stops for the stake cap: an unhedged
position is the one risk the strategy exists not to hold.

Every limit is put on the tick the engine enforces, rounded toward the book (a buy up, a sale down), so it still
reaches the level it was costed on: Polymarket International moves a market between a cent and a tenth of one, and a
book quoted on the finer tick than the listing's spec says would otherwise be rejected order after order. A limit
that rounds to 0 or $1 is not sent; an attempt with such a leg is not made.

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
from datetime import datetime, timezone
from itertools import permutations

from gnomepy import ExecutionReport, Intent, OrderStatus, OrderType, Scales, Side, Strategy
from gnomepy.java.schemas import StreamRecord
from gnomepy.registry import RegistryClient

from gnomepy_research.fees import registry_fee_rates
from gnomepy_research.sessions.esports_cross_venue_arb import game_state as gs
from gnomepy_research.sessions.esports_cross_venue_arb.decision import (
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
# What the match's state allows: new pairs, only completing what is held, or nothing (the session is ending).
OPEN, CLOSED, ENDING, EXIT = "open", "closed", "ending", "exit"
DEFAULT_POLYMARKET_DELAY = 1.0


def _key(order: Order, limit: int) -> tuple[int, int, float, int]:
    return order.listing_id, order.direction, order.contracts, limit


def make_watch(polymarket_slug: str | None, kalshi_event_ticker: str | None, game_states: list | None):
    """How the session follows the match: the rows a session recorded (a replay), the venues' live feeds, or none."""
    if game_states is not None:
        return gs.ReplayWatch([tuple(row) for row in game_states])
    if polymarket_slug or kalshi_event_ticker:
        return gs.GameWatch(polymarket_slug, kalshi_event_ticker)
    return None


@dataclass
class _Leg:
    venue: str
    team: int  # 0 = team A, 1 = team B
    listing_id: int
    exchange_id: int
    security_id: int
    min_size: float
    tick: float = 0.01
    # The venue's market id (Polymarket's condition id), where its listing id carries one.
    market_id: str | None = None
    asks: list[Level] = field(default_factory=list)
    bids: list[Level] = field(default_factory=list)
    bad: bool = True  # no book yet is no book to trust
    in_flight: bool = False
    # What the order in flight is: its kind (ARB, COMPLETION) and (listing, direction, contracts, scaled limit).
    sent: tuple[int, tuple[int, int, float, int]] | None = None

    def top(self) -> tuple:
        return (self.asks[0] if self.asks else None, self.bids[0] if self.bids else None)


class EsportsCrossVenueArb(Strategy):
    """
    Args:
        legs: one {venue, listing_team_a, listing_team_b} per venue listing the market; the same two teams in the same
            order on every venue. Set by the launcher.
        min_edge: the least locked profit per pair, after both fees, worth sending.
        max_contracts: the most pairs one attempt sends.
        max_locked_usd: the most pairs (each paying $1) the session holds; no new attempt starts past it.
        completion_slippage: how far above the short team's best level a completion may pay.
        completion_retry_seconds: the least wait after a completion came back unfilled before the next.
        cooldown_seconds: the least time between two attempts.
        sell_rival_venues: venues where a team may also be held by selling the rival team's listing.
        taker_delay_seconds: how long each venue holds a taker order before matching it; legs are staggered by it.
            Unset, Polymarket International's is read from the market (its secondsDelay, 1 s on every esports match
            winner listed since 2026-11), else 1 s.
        book_depth: book levels read from each update.
        league, subject_id: what the esports_game rule launched this session for; recorded, not used.
        polymarket_slug, kalshi_event_ticker: the match's Polymarket event and Kalshi event, whose live feeds tell the
            session whether it is being played. Set by the esports_game rule; with neither, the session trades at once.
        scheduled_start: when the match is scheduled (ISO); with never_live_hours, ends a session whose match never
            goes live.
        never_live_hours: how long past the scheduled start a match never yet live may be before the session ends.
        stale_seconds: no new pairs once no feed reporting the match has answered for this long.
        ended_settle_seconds, ended_max_seconds: once the match is over, how long the session stays for fills racing
            the end (it leaves after this once nothing is in flight and its position is even), and the most it stays.
        game_states: [[observed_ns, status, source], ...] as a session recorded them, followed in place of the live
            feeds (replays).

    A session given fewer than two venues has nothing to arb: the launcher launches whatever venues have the match's
    listings registered, so this ends the session on its first callback rather than failing it.
    """

    def __init__(
        self,
        legs: list[dict],
        min_edge: float = 0.01,
        max_contracts: float = 1000.0,
        max_locked_usd: float = 2000.0,
        completion_slippage: float = 0.05,
        completion_retry_seconds: float = 1.0,
        cooldown_seconds: float = 10.0,
        sell_rival_venues: tuple[str, ...] = ("kalshi",),
        taker_delay_seconds: dict[str, float] | None = None,
        book_depth: int = 10,
        processing_time_ns: int = 0,
        league: str | None = None,
        subject_id: int | None = None,
        polymarket_slug: str | None = None,
        kalshi_event_ticker: str | None = None,
        scheduled_start: str | None = None,
        never_live_hours: float = 3.0,
        stale_seconds: float = 60.0,
        ended_settle_seconds: float = 30.0,
        ended_max_seconds: float = 600.0,
        game_states: list | None = None,
    ):
        venues = [leg["venue"] for leg in legs]
        if len(set(venues)) != len(venues):
            raise ValueError(f"legs must name each venue once, got {venues}")
        self.league = league
        self.subject_id = subject_id
        start = datetime.fromisoformat(scheduled_start.replace("Z", "+00:00")) if scheduled_start else None
        if start is not None and start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        self._give_up_ns = int((start.timestamp() + never_live_hours * 3600) * 1e9) if start else None
        self._stale_ns = int(stale_seconds * 1e9)
        self._ended_settle_ns = int(ended_settle_seconds * 1e9)
        self._ended_max_ns = int(ended_max_seconds * 1e9)
        self._ended_ns: int | None = None
        self._was_live = False
        self._watch = make_watch(polymarket_slug, kalshi_event_ticker, game_states)
        self._game = gs.GameState()
        self._game_metrics = None
        self._exiting = False
        self._unusable = None if len(venues) >= 2 else f"only {venues} list this market; nothing to arb"
        if min_edge <= 0:
            raise ValueError(f"min_edge must be positive, got {min_edge}")
        self.min_edge = min_edge
        self.max_contracts = max_contracts
        self.max_locked_usd = max_locked_usd
        self.completion_slippage = completion_slippage
        self._completion_retry_ns = int(completion_retry_seconds * 1e9)
        self._completion_wait_until = 0
        # Completions the venue rejected, as (listing, direction, contracts, scaled limit): never resent unchanged.
        self._rejected_completions: set[tuple[int, int, float, int]] = set()
        self.cooldown_ns = int(cooldown_seconds * 1e9)
        self.sell_rival_venues = set(sell_rival_venues)
        self._delay_ns: dict[str, int] = {}
        # Legs of the current attempt waiting for their send time: (send at, order, scaled limit, edge, profit).
        self._held: list[tuple[int, Order, int, float, float]] = []
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
        if taker_delay_seconds is None:
            taker_delay_seconds = {"polymarket": self._polymarket_delay(polymarket_slug)}
        self._delay_ns = {venue: int(seconds * 1e9) for venue, seconds in taker_delay_seconds.items()}

    def _polymarket_delay(self, slug: str | None) -> float:
        """Polymarket International's taker delay on this market (its secondsDelay), else DEFAULT_POLYMARKET_DELAY."""
        legs = self._venue_legs.get("polymarket")
        if not slug or not legs:
            return DEFAULT_POLYMARKET_DELAY
        try:
            delay = gs.polymarket_seconds_delay(slug, legs[0].market_id)
        except Exception:
            logger.warning("Polymarket market %s unreadable; assuming a %.0f s taker delay", slug,
                           DEFAULT_POLYMARKET_DELAY, exc_info=True)
            return DEFAULT_POLYMARKET_DELAY
        logger.info("Polymarket taker delay on %s: %s s", slug, delay)
        return DEFAULT_POLYMARKET_DELAY if delay is None else float(delay)

    @staticmethod
    def _resolve(registry, venue: str, team: int, listing_id: int) -> _Leg:
        listings = registry.get_listing(listing_id=int(listing_id))
        if not listings:
            raise ValueError(f"listing {listing_id} not found in registry")
        specs = registry.get_listing_spec(listing_id=int(listing_id))
        min_size = int(specs[0].min_size) / SIZE_SCALE if specs and specs[0].min_size else 0.0
        tick = int(specs[0].tick_size) / PRICE_SCALE if specs and getattr(specs[0], "tick_size", None) else 0.01
        native = getattr(listings[0], "exchange_security_id", None) or ""
        return _Leg(venue=venue, team=team, listing_id=int(listing_id), exchange_id=listings[0].exchange_id,
                    security_id=listings[0].security_id, min_size=min_size, tick=tick,
                    market_id=native.split(":")[0] or None)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def register_metrics(self) -> None:
        buf = self.metrics.create_stream("esports_arb_orders")
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
        game = self.metrics.create_stream("game_state")
        self._g_ts = game.add_long_column("timestamp")
        self._g_observed = game.add_long_column("observed_ns")
        self._g_status = game.add_long_column("status")
        self._g_source = game.add_long_column("source")
        game.freeze()
        self._game_metrics = game

    # ---- engine callbacks ----------------------------------------------------------------

    def _exit_if_unusable(self) -> bool:
        if self._exiting:
            return True
        if self._unusable is None:
            return False
        self._exit(self._unusable)
        return True

    def _exit(self, reason: str) -> None:
        if not self._exiting:
            self._exiting = True
            logger.info("exiting: %s", reason)
            self.request_exit(reason)

    def _game_gate(self, now_ns: int) -> str:
        """What the match's state allows now (OPEN, CLOSED, ENDING, EXIT), recording each change of it."""
        if self._watch is None:
            return OPEN
        state = self._watch.state(now_ns)
        if state != self._game:
            self._game = state
            self._record_game(now_ns, state)
        self._was_live = self._was_live or self._watch.was_live(now_ns)
        if state.status == gs.ENDED:
            if not self._was_live and not self._holds_anything():
                self._exit(f"the match ended without being played ({state.source})")
                return EXIT
            if self._ended_ns is None:
                # From when the feed said so, not when a callback first came: books can be quiet after the end.
                self._ended_ns = min(now_ns, state.observed_ns) if state.observed_ns else now_ns
            if now_ns - self._ended_ns >= self._ended_max_ns:
                self._exit(f"the match is over ({state.source}); left after ended_max_seconds")
                return EXIT
            return ENDING
        if state.status == gs.LIVE:
            self._was_live = True
        elif not self._was_live and self._give_up_ns is not None and now_ns >= self._give_up_ns:
            self._exit("the match wasn't live by its scheduled start plus never_live_hours")
            return EXIT
        if state.status != gs.LIVE or now_ns - self._watch.heard_ns(now_ns) > self._stale_ns:
            return CLOSED
        return OPEN

    def _holds_anything(self) -> bool:
        """Whether the session holds a position or has an order out or waiting: a restarted session that first hears
        the match ended must still complete what it holds."""
        return bool(self._held) or any(leg.in_flight for leg in self._legs) or any(self._exposure())

    def _leave_once_settled(self, now_ns: int) -> None:
        """Ends the session once the match is over, settle time has passed, and the caller found nothing left to do."""
        if self._ended_ns is not None and now_ns - self._ended_ns >= self._ended_settle_ns:
            self._exit(f"the match is over ({self._game.source})")

    def on_market_data(self, data: StreamRecord) -> list[Intent]:
        if self._exit_if_unusable():
            return []
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
        if self._exit_if_unusable():
            return []
        leg = self._leg_of.get(self.streams.listing_id_for(report.exchange_id, report.security_id))
        if leg is None or report.order_status not in TERMINAL:
            return []
        leg.in_flight = False
        self._note_outcome(leg, report)
        return self._decide(report.timestamp_recv)

    def _note_outcome(self, leg: _Leg, report: ExecutionReport) -> None:
        """What a terminal report on a taker order means for the next one: a pair that came back short backs off
        until its tops move, a completion that came back unfilled waits, and one rejected is not sent again."""
        if leg.sent is None or report.order_status == OrderStatus.FILLED:
            return
        kind, key = leg.sent
        if kind == COMPLETION:
            self._completion_wait_until = report.timestamp_recv + self._completion_retry_ns
            if report.order_status == OrderStatus.REJECTED:
                self._rejected_completions.add(key)
        elif self._last_pair is not None:
            self._backoff[self._last_pair] = self._pair_tops(self._last_pair)

    # ---- decisions -----------------------------------------------------------------------

    def _decide(self, now_ns: int) -> list[Intent]:
        gate = self._game_gate(now_ns)
        if gate == EXIT:
            return []
        if self._held:
            due = [h for h in self._held if h[0] <= now_ns]
            self._held = [h for h in self._held if h[0] > now_ns]
            return [self._send(now_ns, ARB, order, limit, edge, profit, 0.0) for _, order, limit, edge, profit in due]
        if any(leg.in_flight for leg in self._legs):
            return []
        pays_a, pays_b = self._exposure()
        gap = pays_a - pays_b
        if abs(gap) >= 1:
            return self._complete(now_ns, 1 if gap > 0 else 0, abs(gap), gap)
        if gate == ENDING:
            self._leave_once_settled(now_ns)
            return []
        if gate != OPEN or now_ns < self._cooldown_until:
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
            if plan and any(self._take_limit(o) is None for o in plan.orders_a + plan.orders_b):
                plan = None
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
                self._held.append((now_ns + wait, o, self._take_limit(o), plan.edge_first, plan.profit))
            else:
                now_orders.append(o)
        return [self._send(now_ns, ARB, o, self._take_limit(o), plan.edge_first, plan.profit, 0.0) for o in now_orders]

    def _complete(self, now_ns: int, team: int, contracts: float, gap: float) -> list[Intent]:
        sends = self._completion(now_ns, team, contracts)
        if not sends:
            return []
        logger.info("completing %.0f contracts of team %s (gap %.2f) over %d orders", contracts, "AB"[team], gap,
                    len(sends))
        return [self._send(now_ns, COMPLETION, o, limit, math.nan, math.nan, gap) for o, limit in sends]

    def _completion_venue_sets(self) -> list[list[str]]:
        """The venues a completion may use, in order of preference: those matching a taker order at once, then all."""
        fast = [venue for venue in self._venue_legs if self._delay_ns.get(venue, 0) <= 0]
        return [fast] if len(fast) == len(self._venue_legs) else [fast, list(self._venue_legs)]

    def _completion(self, now_ns: int, team: int, contracts: float) -> list[tuple[Order, int]]:
        """The orders, with their scaled limits, that buy the short team now: none while a retry waits."""
        if now_ns < self._completion_wait_until:
            return []
        for venues in self._completion_venue_sets():
            ladder = sorted((lv for venue in venues for lv in self._ladder(venue, team)),
                            key=lambda r: (r.all_in, r.direction))
            sends = []
            for o in plan_completion(ladder, contracts, self.completion_slippage, self._min_sizes()):
                limit = self._take_limit(o)
                if limit is not None and _key(o, limit) not in self._rejected_completions:
                    sends.append((o, limit))
            if sends:
                return sends
        return []

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

    def _take_limit(self, order: Order) -> int | None:
        """The order's scaled limit on the engine's tick, rounded toward the book; None where that leaves no price."""
        leg = self._leg_of[order.listing_id]
        # compliant_price rounds an ask up and a bid down: the way toward the book for a buy and a sale.
        side = Side.ASK if order.direction == BUY else Side.BID
        limit = self.positions.compliant_price(leg.exchange_id, leg.security_id, round(order.limit * PRICE_SCALE),
                                               side)
        return limit if 0 < limit < PRICE_SCALE else None

    def _send(self, now_ns: int, kind: int, order: Order, limit: int, edge: float, profit: float,
              gap: float) -> Intent:
        leg = self._leg_of[order.listing_id]
        leg.in_flight = True
        leg.sent = (kind, _key(order, limit))
        self._record(now_ns, kind, order, edge, profit, gap)
        return Intent(leg.exchange_id, leg.security_id, take_side=Side.BID if order.direction == BUY else Side.ASK,
                      take_size=round(order.contracts * SIZE_SCALE), take_order_type=OrderType.LIMIT,
                      take_limit_price=limit)

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

    def _record_game(self, now_ns: int, state: gs.GameState) -> None:
        logger.info("match state: %s (%s)", state.status, state.source or "no source")
        if self._game_metrics is None:
            return
        buf = self._game_metrics
        buf.begin()
        buf.set_long(self._g_ts, now_ns)
        buf.set_long(self._g_observed, state.observed_ns)
        buf.set_long(self._g_status, gs.STATUS_CODES[state.status])
        buf.set_long(self._g_source, gs.SOURCE_CODES[state.source])
        buf.commit()
