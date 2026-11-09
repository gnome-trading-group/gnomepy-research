"""
CS2 cross-venue arb, maker on Polymarket International: rest a bid on each team there, and when one fills buy the other
team at once on Kalshi or Polymarket US.

Polymarket International holds marketable orders on these markets for about a second (secondsDelay) while resting
orders post at once, so taking there arrives late against quotes that have already moved, and resting there gets the
second's protection instead. Each team's bid is priced from what the other team costs to take elsewhere right now,
so a fill locks min_edge per pair if the hedge is still there (quoting.py). Bids are post-only and re-priced whenever
the hedge books move; they are pulled while a fill is being hedged and once the session holds max_locked_usd.

A fill is hedged through the parent's completion, limited to the other venues (a Polymarket International take would
wait out its delay), no dearer than the best hedge plus completion_slippage, retried until the two teams match. It
never takes on its own.

Bids are rounded onto the tick the engine says is in force (Polymarket International moves a market from a cent to a
tenth of one as its price nears the ends, so the spec read at start can be wrong later). A bid the venue rejects is not
sent again at the same price and size: the reject would only repeat.
"""
from __future__ import annotations

import logging

from gnomepy import ExecutionReport, Intent, OrderStatus, Scales, Side

from gnomepy_research.sessions.cs2_cross_venue_arb.decision import plan_completion
from gnomepy_research.sessions.cs2_cross_venue_arb.strategy import COMPLETION, TERMINAL, CS2CrossVenueArb
from gnomepy_research.sessions.cs2_cross_venue_arb__pm_maker.quoting import maker_bid

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE


class CS2CrossVenueArbPMMaker(CS2CrossVenueArb):
    """
    Args (beyond the parent's, which keep their meaning; cooldown_seconds and taker_delay_seconds go unused):
        maker_venue: the venue quoted.
        quote_size: the most contracts bid on one team.
        depth_band: how far past the best hedge level the size may reach.
        requote_size_change: how far the hedge must shrink below a resting bid's size before it is resent smaller.
    """

    def __init__(self, legs: list[dict], maker_venue: str = "polymarket", quote_size: float = 200.0,
                 depth_band: float = 0.01, requote_size_change: float = 0.5, **kwargs):
        super().__init__(legs, **kwargs)
        if maker_venue not in self._venue_legs:
            raise ValueError(f"maker_venue {maker_venue} is not one of the legs' venues")
        self.maker_venue = maker_venue
        self.quote_size = quote_size
        self.depth_band = depth_band
        self.requote_size_change = requote_size_change
        self._quoted: dict[int, tuple[float, float] | None] = {leg.listing_id: None
                                                               for leg in self._venue_legs[maker_venue]}
        self._rejected: dict[int, tuple[float, float] | None] = {}

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
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
        if terminal or report.filled_qty:
            return self._decide(report.timestamp_recv)
        return []

    def _decide(self, now_ns: int) -> list[Intent]:
        if any(leg.in_flight for leg in self._legs):
            return []
        pays_a, pays_b = self._exposure()
        gap = pays_a - pays_b
        if abs(gap) >= 1:
            return self._pull_quotes() + self._complete(now_ns, 1 if gap > 0 else 0, abs(gap), gap)
        if min(pays_a, pays_b) >= self.max_locked_usd:
            return self._pull_quotes()
        intents = []
        for leg in self._venue_legs[self.maker_venue]:
            want = self._target(leg)
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

    def _target(self, leg) -> tuple[float, float] | None:
        if leg.bad:
            return None
        hedge = sorted((lv for venue in self._venue_legs if venue != self.maker_venue
                        for lv in self._ladder(venue, 1 - leg.team)), key=lambda r: (r.all_in, r.direction))
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

    def _complete(self, now_ns: int, team: int, contracts: float, gap: float) -> list[Intent]:
        ladder = sorted((lv for venue in self._venue_legs if venue != self.maker_venue
                         for lv in self._ladder(venue, team)), key=lambda r: (r.all_in, r.direction))
        orders = plan_completion(ladder, contracts, self.completion_slippage, self._min_sizes())
        if orders:
            logger.info("hedging %.0f contracts of team %s (gap %.2f) over %d orders", contracts, "AB"[team], gap,
                        len(orders))
        return [self._send(now_ns, COMPLETION, o, float("nan"), float("nan"), gap) for o in orders]
