"""
Esports cross-venue arb, maker on Polymarket International: rest a bid on each team there, and when one fills buy the other
team at once on Kalshi or Polymarket US.

Polymarket International holds marketable orders on these markets for about a second (secondsDelay) while resting
orders post at once, so taking there arrives late against quotes that have already moved, and resting there gets the
second's protection instead. Each team's bid is priced from what the other team costs to take elsewhere right now,
so a fill locks min_edge per pair if the hedge is still there (quoting.py). Bids are post-only and re-priced whenever
the hedge books move; they are pulled while a fill is being hedged and once the session holds max_locked_usd.

A fill is hedged through the parent's completion, limited to the other venues (a Polymarket International take would
wait out its delay), no dearer than the best hedge plus completion_slippage, retried until the two teams match (no
sooner than completion_retry_seconds after one came back unfilled, and never resent unchanged after a reject). It
never takes on its own.

Bids are rounded onto the tick the engine says is in force (Polymarket International moves a market from a cent to a
tenth of one as its price nears the ends, so the spec read at start can be wrong later). A bid the venue rejects is not
sent again at the same price and size: the reject would only repeat.

Bids are pulled, and stay down, while either price they hang on has moved more than calm_move within the last
calm_seconds: the cost of the hedge, or the quoted team's own best bid. Books that jump on a round's result run through
a resting bid faster than any cancel, and the hedge priced a moment before is gone by the time the fill arrives.

Bids rest only while the parent's match state allows new pairs (live, feed answering), and are pulled for good once
either team is priced past quote_guard of a sure thing: the match is all but decided, its last round is the biggest
jump of it, and Polymarket International keeps filling resting bids for minutes after the result is known.
"""
from __future__ import annotations

import logging

from gnomepy import ExecutionReport, Intent, OrderStatus, Scales, Side

from gnomepy_research.sessions.esports_cross_venue_arb.strategy import (
    COMPLETION,
    ENDING,
    EXIT,
    OPEN,
    TERMINAL,
    EsportsCrossVenueArb,
)
from gnomepy_research.sessions.esports_cross_venue_arb__pm_maker.quoting import MoveWindow, maker_bid

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE


class EsportsCrossVenueArbPMMaker(EsportsCrossVenueArb):
    """
    A session whose legs don't include maker_venue ends itself, as one with a single venue does.

    Args (beyond the parent's, which keep their meaning; cooldown_seconds and taker_delay_seconds go unused):
        maker_venue: the venue quoted.
        quote_size: the most contracts bid on one team.
        depth_band: how far past the best hedge level the size may reach.
        requote_size_change: how far the hedge must shrink below a resting bid's size before it is resent smaller.
        calm_seconds, calm_move: no bid while the hedge cost or the quoted team's best bid has ranged more than
            calm_move over the last calm_seconds; calm_move of None quotes through every move.
        quote_guard: no bid, ever again, once any team's mid (both sides quoted) is above 1 - quote_guard or below
            quote_guard; None quotes to the end.
    """

    def __init__(self, legs: list[dict], maker_venue: str = "polymarket", quote_size: float = 200.0,
                 depth_band: float = 0.01, requote_size_change: float = 0.5, calm_seconds: float = 5.0,
                 calm_move: float | None = None, quote_guard: float | None = 0.02, **kwargs):
        super().__init__(legs, **kwargs)
        self.quote_guard = quote_guard
        self._decided = False
        if maker_venue not in self._venue_legs:
            self._unusable = self._unusable or f"{maker_venue} doesn't list this market; nothing to quote"
        self.maker_venue = maker_venue
        self.quote_size = quote_size
        self.depth_band = depth_band
        self.requote_size_change = requote_size_change
        self.calm_move = calm_move
        self._moves = {leg.listing_id: (MoveWindow(calm_seconds), MoveWindow(calm_seconds))
                       for leg in self._venue_legs.get(maker_venue, ())}
        self._quoted: dict[int, tuple[float, float] | None] = {leg.listing_id: None
                                                               for leg in self._venue_legs.get(maker_venue, ())}
        self._rejected: dict[int, tuple[float, float] | None] = {}

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        if self._exit_if_unusable():
            return []
        leg = self._leg_of.get(self.streams.listing_id_for(report.exchange_id, report.security_id))
        if leg is None:
            return []
        terminal = report.order_status in TERMINAL
        if leg.venue == self.maker_venue:
            if report.order_status == OrderStatus.REJECTED:
                self._rejected[leg.listing_id] = self._quoted[leg.listing_id]
            if terminal:
                self._quoted[leg.listing_id] = None
        elif terminal:
            leg.in_flight = False
            self._note_outcome(leg, report)
        if terminal or report.filled_qty:
            return self._decide(report.timestamp_recv)
        return []

    def _decide(self, now_ns: int) -> list[Intent]:
        gate = self._game_gate(now_ns)
        if gate == EXIT:
            return self._pull_quotes()
        hedges = {leg.listing_id: self._hedge(leg) for leg in self._venue_legs[self.maker_venue]}
        for leg in self._venue_legs[self.maker_venue]:
            self._sample(leg, now_ns, hedges[leg.listing_id])
        if any(leg.in_flight for leg in self._legs):
            return []
        pays_a, pays_b = self._exposure()
        gap = pays_a - pays_b
        if abs(gap) >= 1:
            return self._pull_quotes() + self._complete(now_ns, 1 if gap > 0 else 0, abs(gap), gap)
        if gate == ENDING:
            pulls = self._pull_quotes()
            if not pulls:
                self._leave_once_settled(now_ns)
            return pulls
        if min(pays_a, pays_b) >= self.max_locked_usd or gate != OPEN or self._all_but_decided():
            return self._pull_quotes()
        intents = []
        for leg in self._venue_legs[self.maker_venue]:
            want = self._target(leg, hedges[leg.listing_id])
            have = self._quoted[leg.listing_id]
            if want is not None and want == self._rejected.get(leg.listing_id):
                continue
            # A resend loses the bid's place in the queue, so the same price is only resent to shrink it to what the
            # hedge can still cover.
            if want == have or (want and have and want[0] == have[0]
                                and want[1] > (1 - self.requote_size_change) * have[1]):
                continue
            intents.append(self._quote(leg, want))
        return intents

    def _all_but_decided(self) -> bool:
        if self.quote_guard is None or self._decided:
            return self._decided
        for leg in self._legs:
            if leg.bad or not leg.bids or not leg.asks:
                continue
            # The mid, not one side: a stray order on a thin book alone mustn't take the quotes down for good.
            mid = (leg.bids[0].price + leg.asks[0].price) / 2
            if mid > 1 - self.quote_guard or mid < self.quote_guard:
                self._decided = True
                logger.info("quotes down for good: team %s on %s is priced at %.3f", "AB"[leg.team], leg.venue, mid)
        return self._decided

    def _hedge(self, leg) -> list:
        return sorted((lv for venue in self._venue_legs if venue != self.maker_venue
                       for lv in self._ladder(venue, 1 - leg.team)), key=lambda r: (r.all_in, r.direction))

    def _target(self, leg, hedge) -> tuple[float, float] | None:
        if leg.bad or self._jumpy(leg):
            return None
        maker_rate = self._fee_rates(leg.listing_id)[1]
        best_ask = leg.asks[0].price if leg.asks else None
        size = min(self.quote_size, self.max_locked_usd)
        bid = maker_bid(hedge, size, self.min_edge, maker_rate, leg.tick, best_ask, self.depth_band)
        if bid is None or bid[1] < max(1.0, leg.min_size):
            return None
        price = self.positions.compliant_price(leg.exchange_id, leg.security_id, round(bid[0] * PRICE_SCALE),
                                               Side.BID) / PRICE_SCALE
        if price <= 0 or (best_ask is not None and price >= best_ask):
            return None
        return price, bid[1]

    def _sample(self, leg, now_ns: int, hedge) -> None:
        hedge_moves, own_moves = self._moves[leg.listing_id]
        if hedge:
            hedge_moves.add(now_ns, hedge[0].all_in)
        if leg.bids and not leg.bad:
            own_moves.add(now_ns, leg.bids[0].price)

    def _jumpy(self, leg) -> bool:
        if self.calm_move is None:
            return False
        return max(m.range() for m in self._moves[leg.listing_id]) > self.calm_move + 1e-9

    def _quote(self, leg, want: tuple[float, float] | None) -> Intent:
        self._quoted[leg.listing_id] = want
        if want is None:
            return Intent(leg.exchange_id, leg.security_id)
        price, size = want
        return Intent(leg.exchange_id, leg.security_id, bid_price=round(price * PRICE_SCALE),
                      bid_size=round(size * SIZE_SCALE), post_only=True)

    def _pull_quotes(self) -> list[Intent]:
        return [self._quote(leg, None) for leg in self._venue_legs[self.maker_venue]
                if self._quoted[leg.listing_id] is not None]

    def _completion_venue_sets(self) -> list[list[str]]:
        return [[venue for venue in self._venue_legs if venue != self.maker_venue]]

    def _complete(self, now_ns: int, team: int, contracts: float, gap: float) -> list[Intent]:
        sends = self._completion(now_ns, team, contracts)
        if sends:
            logger.info("hedging %.0f contracts of team %s (gap %.2f) over %d orders", contracts, "AB"[team], gap,
                        len(sends))
        return [self._send(now_ns, COMPLETION, o, limit, float("nan"), float("nan"), gap) for o, limit in sends]
