"""
CS2 series winner in the last hour before kickoff, quoted passively instead of taken: the cs2_series strategy's model,
blend, legs and window, with resting bids in place of IOC takes.

On each decision (a top-of-book change on any leg, a fill, a new model price, the window opening) it states the quote
it wants on every listing and sends only the changes; the OMS keeps the resting orders in line with them.
- The fair price never comes from a book we might be making: it blends the model with the mid of a liquid venue (spread
  at most liquid_spread), Polymarket International's when it qualifies, else the tightest; with no liquid venue it is
  the model alone. A wide book's mid is the midpoint of nobody's prices (one match had Polymarket International at
  0.01-0.88 while Polymarket US quoted 0.32/0.33).
- On a liquid venue someone else sets the price: a team is bid when its best bid leaves at least min_edge below fair
  after the maker fee (PM US pays makers a rebate), joining it, else resting at the highest price that does.
- On a wide venue we set the price: both teams are bid at fair less half_spread, inside the spread, so we become its
  best bid and best ask. Never at or above a liquid venue's ask for the team, or anyone could buy there and sell to us.
  A venue with no book at all (a feed we don't have) is never quoted.
- wide_only quotes wide venues alone: joining liquid books gets picked off (fills there are followed by the price moving
  against them), while setting the price where nobody else quotes is where the model's fair price earns its spread.
- Nothing ever crosses its own book.
- One-sided quoting (liquid venues only) bids the team the model favours against the mid; two-sided bids both, so a fill
  on the other team unwinds the position at a locked profit before it adds exposure the other way.
- A team is bid on every venue where a bid qualifies, so a wide book is priced even while a liquid one is joined. The
  budget is split between the teams being quoted, and a team's size shared equally between its venues: together the bids never exceed the Kelly top-up at the best price on the
  net position (shares of the team minus shares of the other), plus whatever unwinds an opposite position, nor
  max_stake_usd across the match, resting quotes counted as spent. Handing the whole size to the best-priced venue
  would leave the others, a wide book in particular, unquoted.
- A resting quote is left alone while it is still worth having: on a liquid venue at or below its fair price less the
  edge and still at or above the best bid; on a wide venue within EDGE_SLACK of fair less half_spread. It moves up only
  to rejoin a best bid that rose past it, down only when the edge demands, and is resized only when the size it should
  have moved by more than a quarter. Every modify sends an order to the back of the queue, so re-pricing on each tick of
  the fair price would never fill.
- A resting quote may keep up to EDGE_SLACK less edge than min_edge before it is moved down: the fair price wiggles with
  the mid, and a quote joined near its limit would otherwise be moved down and back up on every wiggle.
- The favoured team changes only when the model leans the other way against the mid by more than a cent.
- The reference mid holds for up to a minute when its venue's book briefly stops qualifying.
- All quotes are pulled when the model's price changes (a lineup change makes the resting quotes stale before the
  market reprices), and for good pull_before_kickoff_min before kickoff; the session ends at the entry cutoff once no
  quote is left resting.
"""
from __future__ import annotations

import math

from gnomepy import Intent
from gnomepy_research.sessions.cs2_series.decision import cost_per_share, kelly_shares, max_entry_price
from gnomepy_research.sessions.cs2_series.strategy import (MID_VENUE, PRICE_SCALE, SIZE_SCALE, UPCOMING, CS2Series, _Leg,
                                                          _mid)

MINUTE_NS = 60_000_000_000
HOME_MID_HOLD_NS = MINUTE_NS
RESIZE_FRACTION = 0.25
FAVOURED_HYSTERESIS = 0.01
EDGE_SLACK = 0.005


def _floor(value: float, step: float) -> float:
    return math.floor(value / step + 1e-9) * step


class CS2SeriesMM(CS2Series):
    """
    Args: those of CS2Series, plus
        two_sided: on liquid venues, bid both teams, not just the one the model favours (wide venues always get both).
        half_spread: on wide venues, how far below each team's fair price its bid sits.
        liquid_spread: the widest spread at which a venue's book counts as liquid.
        wide_only: quote wide venues only, never joining a liquid book.
        pull_before_kickoff_min: stop quoting this long before kickoff.
    """

    def __init__(self, *args, two_sided: bool = False, half_spread: float = 0.03, liquid_spread: float = 0.05,
                 wide_only: bool = False, pull_before_kickoff_min: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.two_sided = two_sided
        self.half_spread = half_spread
        self.liquid_spread = liquid_spread
        self.wide_only = wide_only
        self.pull_ns = int(pull_before_kickoff_min * MINUTE_NS)
        # What each listing was last asked to rest: (price, shares), or None for nothing.
        self._sent: dict[int, tuple[float, float] | None] = {leg.listing_id: None for leg in self._legs}
        self._quoted_q: float | None = None
        self._home_mid: tuple[float, int] | None = None
        self._favoured: int | None = None

    def _maker_rate(self, leg: _Leg) -> float:
        return self._fee_rates(leg.listing_id)[1]

    def on_execution_report(self, report) -> list[Intent]:
        if getattr(report, "filled_qty", 0):
            self._dirty = True
        return []

    def _filled(self, leg: _Leg) -> tuple[float, float]:
        """(shares held, what they cost including fees)."""
        pos = self.positions.get_position(leg.exchange_id, leg.security_id)
        if not pos or not pos.net_quantity:
            return 0.0, 0.0
        shares = pos.net_quantity / SIZE_SCALE
        return shares, shares * pos.avg_entry_price / PRICE_SCALE + pos.total_fees / PRICE_SCALE

    def _resting(self) -> bool:
        return any(self._leg_state(leg)[1] > 1e-9 for leg in self._legs)

    def _intents(self, wanted: dict[int, tuple[float, float] | None]) -> list[Intent]:
        out = []
        for leg in self._legs:
            want = wanted.get(leg.listing_id)
            if want == self._sent[leg.listing_id]:
                continue
            self._sent[leg.listing_id] = want
            if want is None:
                out.append(Intent(leg.exchange_id, leg.security_id))
            else:
                price, shares = want
                out.append(Intent(leg.exchange_id, leg.security_id, bid_price=round(price * PRICE_SCALE),
                                  bid_size=round(shares * SIZE_SCALE)))
        return out

    def _pull(self) -> list[Intent]:
        return self._intents({})

    @staticmethod
    def _venue_mid(a: _Leg, b: _Leg) -> tuple[float, float] | None:
        """(team A mid, spread) of a venue: team A's book, else team B's as 1 - its mid."""
        m = _mid(a)
        if m is None and (mb := _mid(b)) is not None:
            m = (1.0 - mb[0], mb[1])
        return m

    def _liquid(self, venue: str) -> bool:
        a, b = dict(self._venues)[venue]
        m = self._venue_mid(a, b)
        return m is not None and m[1] <= self.liquid_spread + 1e-9

    def _reference_mid(self, now_ns: int) -> float | None:
        """Team A's mid on a liquid venue (Polymarket International first, then the tightest), the last one for up to
        HOME_MID_HOLD_NS while none qualifies, else None (the model alone)."""
        liquid = []
        for venue, (a, b) in self._venues:
            m = self._venue_mid(a, b)
            if m is not None and m[1] <= self.liquid_spread + 1e-9:
                liquid.append((venue != MID_VENUE, m[1], m[0]))
        if liquid:
            mid = min(liquid)[2]
            self._home_mid = (mid, now_ns)
            return mid
        if self._home_mid is not None and now_ns - self._home_mid[1] <= HOME_MID_HOLD_NS:
            return self._home_mid[0]
        return None

    def _price(self, leg: _Leg, fair: float, liquid_asks: dict[int, float]) -> float | None:
        """Where a bid for the leg's team should rest. On a liquid venue: the existing bid while it's still at or above the
        best bid and within the edge (less EDGE_SLACK), else the best bid or the highest price that leaves min_edge,
        whichever is lower. On a wide venue: fair less half_spread (the existing bid while within EDGE_SLACK of it),
        below every liquid venue's ask for the team. Never at or above the venue's own best ask."""
        rate = self._maker_rate(leg)
        sent = self._sent.get(leg.listing_id)
        own_ask = leg.asks[0].price if leg.asks else None
        below_ask = lambda p: own_ask is None or p < own_ask - 1e-9
        if self._liquid(leg.venue):
            if not leg.bids:
                return None
            cap = _floor(max_entry_price(fair, self.min_edge, rate), leg.tick)
            target = min(leg.bids[0].price, cap)
            keep_cap = max_entry_price(fair, self.min_edge - EDGE_SLACK, rate)
            if sent is not None and target - 1e-9 <= sent[0] <= keep_cap + 1e-9 and below_ask(sent[0]):
                return sent[0]
        else:
            a, b = dict(self._venues)[leg.venue]
            if not (a.bids or a.asks or b.bids or b.asks):
                # No book at all: a feed we don't have, not a market waiting for a price.
                return None
            target = _floor(max_entry_price(fair, self.half_spread, rate), leg.tick)
            ceiling = liquid_asks.get(leg.team)
            if ceiling is not None:
                target = min(target, _floor(ceiling - leg.tick, leg.tick))
            if sent is not None and abs(sent[0] - target) <= EDGE_SLACK + 1e-9 and below_ask(sent[0]) and \
                    (ceiling is None or sent[0] < ceiling - 1e-9):
                return sent[0]
        if not below_ask(target):
            target = _floor(own_ask - leg.tick, leg.tick)
        return target if target > 0 else None

    def _evaluate(self, now_ns: int) -> list[Intent]:
        state, quote = self._feed.state(now_ns), self._feed.quote(now_ns)
        if state is not None and (state.status != UPCOMING or now_ns >= state.kickoff_ns - self.cutoff_ns):
            pulled = self._pull()
            if not pulled and not self._resting():
                self.request_exit(f"match {self.match_id} is {state.status}" if state.status != UPCOMING
                                  else f"match {self.match_id} reached its entry cutoff")
            return pulled
        if state is None or quote is None:
            return []
        self._record_quote(now_ns, quote)
        if now_ns >= state.kickoff_ns - self.pull_ns:
            return self._pull()
        ready = (now_ns - state.at_ns <= self.max_state_age_ns and now_ns >= state.kickoff_ns - self.window_ns
                 and quote.lineup_known and not math.isnan(quote.q)
                 and 0 <= now_ns - self._model_cutoff_ns <= self.max_model_age_ns)
        if not ready:
            return self._pull()
        if self._quoted_q is not None and quote.q != self._quoted_q:
            # The model's price moved: the resting quotes were priced on the old one.
            self._quoted_q = quote.q
            self._dirty = True
            return self._pull()
        if not self._was_ready or quote.at_ns != self._decided_quote_at:
            self._dirty = True
        self._was_ready = True
        if not self._dirty:
            return []
        self._dirty = False
        self._decided_quote_at = quote.at_ns
        self._quoted_q = quote.q
        ref = self._reference_mid(now_ns)
        fair_a = self._blend.fair(ref, quote.q) if ref is not None else quote.q
        mid = ref if ref is not None else fair_a
        fairs = (fair_a, 1.0 - fair_a)
        filled = {leg.listing_id: self._filled(leg) for leg in self._legs}
        shares = [sum(filled[leg.listing_id][0] for leg in self._legs if leg.team == t) for t in (0, 1)]
        filled_cost = sum(cost for _, cost in filled.values())
        lean = fair_a - mid
        if self._favoured is None or (self._favoured == 1 and lean > FAVOURED_HYSTERESIS) or \
                (self._favoured == 0 and lean < -FAVOURED_HYSTERESIS):
            self._favoured = 0 if lean >= 0 else 1
        liquid_teams = (0, 1) if self.two_sided else (self._favoured,)
        liquid_asks = {}
        for leg in self._legs:
            if self._liquid(leg.venue) and leg.asks:
                liquid_asks[leg.team] = min(liquid_asks.get(leg.team, 1.0), leg.asks[0].price)
        wanted: dict[int, tuple[float, float] | None] = {}
        budget = self.max_stake_usd - filled_cost
        kelly_bankroll = self.kelly_fraction * self.bankroll_usd
        quotable = {}
        for team in (0, 1):
            options = []
            for leg in self._legs:
                if leg.team != team:
                    continue
                if self._liquid(leg.venue) and (self.wide_only or team not in liquid_teams):
                    continue
                price = self._price(leg, fairs[team], liquid_asks)
                if price is not None:
                    options.append(((price, -self._maker_rate(leg)), leg, price))
            if options:
                quotable[team] = sorted(options, key=lambda o: o[0], reverse=True)
        for k, (team, options) in enumerate(quotable.items()):
            best_unit = cost_per_share(options[0][2], self._maker_rate(options[0][1]))
            net = shares[team] - shares[1 - team]
            left = max(0.0, -net) + max(0.0, kelly_shares(fairs[team], best_unit, kelly_bankroll, max(net, 0.0),
                                                          max(net, 0.0) * best_unit))
            # The budget left is split between the teams still to quote, so bidding both sides of a book never hands
            # the whole stake to whichever team comes first.
            team_budget = max(budget, 0.0) / (len(quotable) - k)
            for n, (_, leg, price) in enumerate(options):
                unit = cost_per_share(price, self._maker_rate(leg))
                share = len(options) - n
                size = _floor(min(left / share, team_budget / share / unit, max(budget, 0.0) / unit), leg.lot / SIZE_SCALE)
                sent = self._sent.get(leg.listing_id)
                if sent is not None and sent[1] * unit <= budget + 1e-9 and abs(size - sent[1]) <= RESIZE_FRACTION * sent[1]:
                    size = sent[1]
                if size * SIZE_SCALE < max(leg.min_size, 1) or size * unit < self.min_order_usd:
                    continue
                wanted[leg.listing_id] = (price, size)
                budget -= size * unit
                team_budget -= size * unit
                left -= size
                if self._sent[leg.listing_id] != (price, size):
                    self._record_order(now_ns, leg, 1, quote.q, mid, fairs[team],
                                       leg.bids[0].price if leg.bids else math.nan, fairs[team] - price, price, size,
                                       size * unit, shares[team], kelly_bankroll)
        return self._intents(wanted)
