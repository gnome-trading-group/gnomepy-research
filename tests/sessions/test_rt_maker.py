"""rt_maker: the quoting rules, and the session's quotes against fake books, reviews and reward programs."""
from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from gnomepy import OrderStatus, Scales
from gnomepy.price_grid import PriceBand, PriceGrid
from gnomepy.incentives import IncentiveProgram, IncentiveTimeline, ProgramVersions
from gnomepy_research.sessions.rt_maker import strategy as mod
from gnomepy_research.models.rt.review_feed import Review
from gnomepy_research.models.rt.signal import NO, YES
from gnomepy_research.sessions.rt_maker.quotes import (BOTH, NONE, PULL, Level, Quote, QueueSpot, adds, advance,
                                                       join_price, joined, lean, lots, moved_steps, near_front,
                                                       one_side, placed, quote, reducing, review_state, skew_ticks)
from tests.sessions.conftest import CENT_GRID, FakeStreams, stream_of

P, S = Scales.PRICE, Scales.SIZE
MINUTE = 60_000_000_000
T0 = pd.Timestamp("2026-11-14T14:00:00Z").value
FILM = "ebenezer"
KALSHI, PM = 5, 4
STRIKES = (40, 60)
LEGS = ([{"venue": "kalshi", "strike": k, "yes_listing": 5000 + k, "no_listing": 5100 + k} for k in STRIKES]
        + [{"venue": "polymarket", "strike": 60, "yes_listing": 4060, "no_listing": 4160}])
FILM_B = "nosferatu"
LEGS_B = [{"venue": "kalshi", "strike": k, "yes_listing": 6000 + k, "no_listing": 6100 + k} for k in STRIKES]
LISTINGS = {leg[key]: (KALSHI if leg["venue"] == "kalshi" else PM, leg[key] + 10_000)
            for leg in LEGS + LEGS_B for key in ("yes_listing", "no_listing")}
YES40, YES60 = 5040, 5060
B40, B60 = 6040, 6060
HOUR = 60 * MINUTE
PULL_NS = 30 * MINUTE


# ---- the quoting rules ---------------------------------------------------------------------------------------------

def test_the_join_price_skips_a_level_holding_only_our_own_size():
    bids = [Level(0.45, 500), Level(0.44, 800)]
    assert join_price(bids, None, 1) == 0.45
    assert join_price(bids, (0.45, 500), 1) == 0.44, "alone at 0.45: the others are at 0.44"
    assert join_price(bids, (0.45, 200), 1) == 0.45
    assert join_price([Level(0.45, 3)], None, 5) is None


def test_sizes_round_down_to_the_lot_and_vanish_under_the_minimum():
    assert lots(499.7, 1, 0) == 499
    assert lots(0.4, 1, 0) == 0
    assert lots(4, 1, 5) == 0


def test_a_quote_joins_both_touches_capped_by_the_position_limit():
    bids, asks = [Level(0.45, 900)], [Level(0.47, 900)]
    assert quote(bids, asks, 0, None, 500, 500, 0.03, 0.97, 1, 1, 0) == Quote(0.45, 500, 0.47, 500)
    assert quote(bids, asks, 300, None, 500, 500, 0.03, 0.97, 1, 1, 0) == Quote(0.45, 200, 0.47, 500)
    assert quote(bids, asks, 500, None, 500, 500, 0.03, 0.97, 1, 1, 0) == Quote(0, 0, 0.47, 500), "long to the limit"
    assert quote(bids, asks, -500, None, 500, 500, 0.03, 0.97, 1, 1, 0) == Quote(0.45, 500, 0, 0), "short to it"


def test_a_side_priced_outside_the_range_is_left_alone():
    assert quote([Level(0.02, 900)], [Level(0.04, 900)], 0, None, 500, 500, 0.03, 0.97, 1, 1, 0) == Quote(0, 0, 0.04, 500)
    assert quote([Level(0.97, 900)], [Level(0.98, 900)], 0, None, 500, 500, 0.03, 0.97, 1, 1, 0) == Quote(0.97, 500, 0, 0)


def test_a_book_left_crossed_by_our_own_orders_is_not_quoted():
    assert quote([Level(0.50, 10)], [Level(0.50, 10)], 0, None, 500, 500, 0.03, 0.97, 1, 1, 0) == NONE


def test_what_a_quote_adds_counts_only_size_beyond_trading_the_position_back():
    q = Quote(0.40, 500, 0.42, 500)
    assert adds(q, 0) == pytest.approx(500 * 0.40 + 500 * 0.58)
    assert adds(q, 200) == pytest.approx(500 * 0.40 + 300 * 0.58)
    assert reducing(q, 200, 1, 0) == Quote(0, 0, 0.42, 200)
    assert reducing(q, -100, 1, 0) == Quote(0.40, 100, 0, 0)
    assert reducing(q, 0, 1, 0) == NONE


# ---- keeping away from fills ----------------------------------------------------------------------------------------

def test_a_trade_the_feed_shows_as_a_shrink_and_a_print_comes_off_the_queue_once():
    for cancel_ahead in (0.0, 1.0):
        shrink_first = joined(0.45, 500, 900)
        advance(shrink_first, 1, 700, 0, cancel_ahead)
        advance(shrink_first, 1, 700, 200, cancel_ahead)
        print_first = joined(0.45, 500, 900)
        advance(print_first, 1, 900, 200, cancel_ahead)
        advance(print_first, 1, 700, 0, cancel_ahead)
        assert shrink_first.ahead == print_first.ahead == 700


def test_a_shrink_without_a_trade_takes_the_assumed_share_of_cancels_off_the_size_ahead():
    for cancel_ahead, ahead in ((1.0, 100), (0.5, 200), (0.0, 300)):
        spot = QueueSpot(0.45, 500, ahead=300, level=900)
        advance(spot, 1, 700, 0, cancel_ahead)
        assert spot.ahead == ahead


def test_size_joining_the_level_lands_behind_us_and_the_level_bounds_what_is_ahead():
    spot = QueueSpot(0.45, 500, ahead=300, level=900)
    advance(spot, 1, 1500, 0, 1.0)
    assert spot.ahead == 300
    advance(spot, 2, 100, 0, 0.0)
    assert spot.ahead == 100, "no more ahead than the level holds"


def test_a_side_is_near_the_front_under_either_threshold():
    spot = QueueSpot(0.45, 500, ahead=100, level=1000)
    assert near_front(spot, 200, 0)
    assert not near_front(spot, 0, 0.05)
    assert near_front(spot, 0, 0.2)


def test_inventory_skew_steps_back_a_tick_per_unit_up_to_the_most():
    assert skew_ticks(0, 250, 2) == 0
    assert skew_ticks(499, 250, 2) == 1
    assert skew_ticks(-1000, 250, 2) == 2
    assert skew_ticks(1000, 0, 2) == 0


def test_placing_steps_the_side_that_adds_to_the_position_back_and_both_back_when_behind():
    q = Quote(0.45, 500, 0.47, 500)
    assert placed(q, 300, CENT_GRID, 0, 1, 0.03, 0.97) == Quote(0.44, 500, 0.47, 500), "long: the bid adds to it"
    assert placed(q, -300, CENT_GRID, 0, 2, 0.03, 0.97) == Quote(0.45, 500, 0.49, 500), "short: the ask adds to it"
    assert placed(q, 0, CENT_GRID, 1, 0, 0.03, 0.97) == Quote(0.44, 500, 0.48, 500)
    assert placed(Quote(0.03, 500, 0.05, 500), 0, CENT_GRID, 1, 0, 0.03, 0.97) == Quote(0, 0, 0.06, 500), \
        "out of range"


TAPERED = PriceGrid.of([
    PriceBand(0, P // 10, P // 1000),
    PriceBand(P // 10, 9 * P // 10, P // 100),
    PriceBand(9 * P // 10, P, P // 1000),
])


def test_placing_steps_back_by_the_grid_where_each_step_lands():
    # A bid joining at 10c steps back into the tail's tenths of a cent; an ask at 89c steps up past the band edge.
    q = Quote(0.10, 500, 0.89, 500)
    assert placed(q, 0, TAPERED, 2, 0, 0.01, 0.99) == Quote(0.098, 500, 0.901, 500)


def test_a_move_is_counted_in_grid_steps_from_the_lower_price():
    assert moved_steps(CENT_GRID, 0.505, 0.535, 3)
    assert not moved_steps(CENT_GRID, 0.505, 0.525, 3), "two cents is short of three steps"
    assert moved_steps(TAPERED, 0.05, 0.053, 3), "three tenths of a cent are three steps in the tail"
    assert not moved_steps(CENT_GRID, 0.05, 0.053, 3)


# ---- leaning on the review signal ------------------------------------------------------------------------------------

def rv(i, seen_min, fresh=True):
    return Review(f"u{i}", T0 + seen_min * MINUTE, T0 + seen_min * MINUTE, fresh)


def mode(reviews, minute, skew_hours=None, pull_every_review=False):
    state = review_state(reviews, 0.667, 0.10, 3)
    return lean(state, T0 + minute * MINUTE, PULL_NS, None if skew_hours is None else int(skew_hours * HOUR),
                pull_every_review)


def test_before_any_review_both_sides_are_quoted():
    assert mode([], 0) == BOTH


def test_a_review_before_the_signal_has_enough_reviews_pulls_then_both_come_back():
    two = [rv(1, 0), rv(2, 5)]
    assert mode(two, 6) == PULL
    assert mode(two, 34) == PULL
    assert mode(two, 35) == BOTH, "30 min after the latest review"


def test_a_side_is_quoted_alone_at_once_and_until_the_next_review():
    fresh = [rv(1, 0), rv(2, 1), rv(3, 2)]
    assert mode(fresh, 2) == YES, "no pull: the review that gave the side leans straight away"
    assert mode(fresh, 10 * 60) == YES, "unbounded: still leaning hours later"
    rotten = [rv(1, 0, False), rv(2, 1, False), rv(3, 2)]
    assert mode(rotten, 3) == NO


def test_a_bounded_lean_returns_to_both_sides_after_skew_hours():
    fresh = [rv(1, 0), rv(2, 1), rv(3, 2)]
    assert mode(fresh, 61, skew_hours=1) == YES
    assert mode(fresh, 62, skew_hours=1) == BOTH


def test_a_review_that_leaves_no_side_ends_the_lean_with_a_pull():
    reviews = [rv(1, 0), rv(2, 1), rv(3, 2), rv(4, 60, False), rv(5, 61, False)]
    # 3 fresh of 5 = 0.6: within 0.10 of 0.667, no side.
    assert mode(reviews, 62) == PULL
    assert mode(reviews, 91) == BOTH


def test_a_flip_pulls_both_first_then_leans_the_new_way():
    reviews = [rv(1, 0), rv(2, 1), rv(3, 2)] + [rv(i, 60, False) for i in range(4, 10)]
    # 3 fresh of 9 = 0.33: NO, after leaning YES.
    assert review_state(reviews, 0.667, 0.10, 3).flip
    assert mode(reviews, 61) == PULL
    assert mode(reviews, 90) == NO


def test_a_flip_counts_against_the_last_side_leaned_even_with_a_no_side_review_between():
    reviews = ([rv(1, 0), rv(2, 1), rv(3, 2), rv(4, 60, False), rv(5, 61, False)]
               + [rv(i, 120, False) for i in range(6, 10)])
    # YES, then no side (3/5), then NO (3/9): a flip from YES.
    state = review_state(reviews, 0.667, 0.10, 3)
    assert (state.side, state.flip) == (NO, True)
    assert mode(reviews, 121) == PULL


def test_a_side_without_a_flip_never_pulls_even_after_a_no_side_review_in_between():
    reviews = ([rv(1, 0), rv(2, 1), rv(3, 2), rv(4, 60, False), rv(5, 61, False)]
               + [rv(i, 120) for i in range(6, 10)])
    # YES, then no side (3/5), then YES again (7/9): the same side as the last leaned is no flip.
    state = review_state(reviews, 0.667, 0.10, 3)
    assert (state.side, state.flip) == (YES, False)
    assert mode(reviews, 120) == YES


def test_reviews_seen_at_the_same_moment_count_as_one():
    # Seen one by one, the first two rotten reviews would say NO and the next fresh one would flip to YES.
    same = [rv(1, 0, False), rv(2, 0, False)] + [rv(i, 0) for i in range(3, 11)]
    assert review_state(same[:3], 0.667, 0.10, 3).side == NO, "one by one, the third review alone says NO"
    state = review_state(same, 0.667, 0.10, 3)
    assert (state.side, state.flip) == (YES, False)


def test_pulling_on_every_review_pulls_even_a_review_with_a_side_then_leans():
    fresh = [rv(1, 0), rv(2, 1), rv(3, 2)]
    assert mode(fresh, 2, pull_every_review=True) == PULL
    assert mode(fresh, 31, pull_every_review=True) == PULL
    assert mode(fresh, 32, pull_every_review=True) == YES, "30 min after the review that gave the side"
    assert mode(fresh + [rv(4, 300)], 301, pull_every_review=True) == PULL, "a later review of the same side pulls too"
    assert mode(fresh + [rv(4, 300)], 330, pull_every_review=True) == YES
    no_side = [rv(1, 0), rv(2, 5)]
    assert mode(no_side, 35, pull_every_review=True) == BOTH


def test_one_side_keeps_the_bid_for_yes_and_the_ask_for_no():
    q = Quote(0.45, 500, 0.47, 400)
    assert one_side(q, YES) == Quote(0.45, 500, 0, 0)
    assert one_side(q, NO) == Quote(0, 0, 0.47, 400)
    assert one_side(q, 0) == q


# ---- the session ---------------------------------------------------------------------------------------------------

@dataclass
class _Listing:
    exchange_id: int
    security_id: int


@dataclass
class _Spec:
    lot_size: int = S
    min_size: int = 0


class _Registry:
    def get_listing(self, *, listing_id, **_):
        return [_Listing(*LISTINGS[listing_id])]

    def get_listing_spec(self, *, listing_id, **_):
        return [_Spec()]


class _Timeline:
    programs = ()

    def __init__(self, paying):
        self.paying = paying

    def in_force(self, when):
        return [object()] if self.paying else []


@dataclass
class _Pos:
    net_quantity: int
    avg_entry_price: int


class _Positions:
    def __init__(self):
        self.held: dict = {}
        self.no_spec: set = set()

    def set(self, listing, contracts, price):
        self.held[LISTINGS[listing]] = _Pos(round(contracts * S), round(price * P))

    def get_position(self, exchange_id, security_id):
        return self.held.get((exchange_id, security_id))

    def price_grid(self, exchange_id, security_id):
        return None if (exchange_id, security_id) in self.no_spec else CENT_GRID


class _Control:
    def isExiting(self):
        return False


class _Book:
    def __init__(self, listing, bids, asks, ts, trade=None):
        self.stream_id = stream_of(listing)
        self.event_timestamp = ts
        self._bids, self._asks = bids, asks
        # The record's own event: a book change, or a trade print of (price, contracts).
        self.action = "Trade" if trade else "Add"
        self.price = int(round(trade[0] * P)) if trade else 0
        self.size = int(trade[1] * S) if trade else 0

    def bid_price(self, i):
        return int(round(self._bids[i][0] * P)) if i < len(self._bids) else 0

    def bid_size(self, i):
        return int(self._bids[i][1] * S) if i < len(self._bids) else 0

    def ask_price(self, i):
        return int(round(self._asks[i][0] * P)) if i < len(self._asks) else 0

    def ask_size(self, i):
        return int(self._asks[i][1] * S) if i < len(self._asks) else 0


def _review(url, seen_min, fresh=True, film=FILM):
    ts = pd.Timestamp(T0 + seen_min * MINUTE, tz="UTC")
    return {"url": url, "film_id": film, "published_at": ts, "first_seen_at": ts, "is_review": True, "fresh": fresh}


def _films(*films):
    """The `films` argument: FILM alone by default, or (film_id, legs, close minute or None) each, its strikes the
    legs' Kalshi ones, as the launcher gives them."""
    return [{"film_id": f, "close": None if close is None else pd.Timestamp(T0 + close * MINUTE, tz="UTC").isoformat(),
             "strikes": {str(leg["strike"]): leg["yes_listing"] for leg in legs if leg["venue"] == "kalshi"}}
            for f, legs, close in (films or [(FILM, LEGS, None)])]


TWO_FILMS = _films((FILM, LEGS, None), (FILM_B, LEGS_B, None))


@pytest.fixture
def make(tmp_path):
    def _make(reviews=(), scores=(), rewarded=(YES40, YES60, B40, B60), films=None, **kw):
        path, scores_path = tmp_path / "rt_reviews.parquet", tmp_path / "rt_scores.parquet"
        pd.DataFrame(list(reviews) or [{**_review("old", -24 * 60), "film_id": "other-film"}]).to_parquet(path)
        pd.DataFrame(list(scores), columns=["film_id", "liked", "not_liked", "first_seen_at"]).to_parquet(scores_path)
        timelines = {lid: _Timeline(lid in rewarded) for lid in (YES40, YES60, B40, B60)}
        with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()), \
                patch.object(mod, "load_incentive_timelines", lambda registry, ids: {i: timelines[i] for i in ids}):
            s = mod.RtMaker(films=films or _films(), reviews_source=str(path), scores_source=str(scores_path),
                            **{"reload_seconds": 0, "programs_reload_seconds": 0, **kw})
        s._position_view = _Positions()
        s._streams = FakeStreams(LISTINGS)
        s._control = _Control()
        return s
    return _make


def book_at(listing, at, bids=((0.45, 900),), asks=((0.47, 900),)):
    return _Book(listing, list(bids), list(asks), at)


def books(s, minute, bids=((0.45, 900),), asks=((0.47, 900),), listings=(YES40, YES60, 4060), trade=None):
    out = []
    for lid in listings:
        out += s.on_market_data(_Book(lid, list(bids), list(asks), T0 + minute * MINUTE, trade))
    return out


def by_listing(intents):
    return {next(lid for lid, key in LISTINGS.items() if key == (i.exchange_id, i.security_id)): i for i in intents}


def report(intent, status, fill=None, leaves=0.0):
    """An execution report on the intent's listing: a fill of (price, contracts), with leaves contracts resting."""
    return SimpleNamespace(exchange_id=intent.exchange_id, security_id=intent.security_id, order_status=status,
                           fill_price=int(round(fill[0] * P)) if fill else 0,
                           filled_qty=int(fill[1] * S) if fill else 0, leaves_qty=int(leaves * S))


def sides(intent):
    return (intent.bid_price / P, intent.bid_size / S, intent.ask_price / P, intent.ask_size / S)


def test_every_rewarded_kalshi_strike_is_quoted_at_both_touches(make):
    s = make()
    out = by_listing(books(s, 0))
    assert set(out) == {YES40, YES60}, "Polymarket's strike isn't quoted"
    assert sides(out[YES60]) == pytest.approx((0.45, 500, 0.47, 500))


def test_a_strike_without_a_reward_program_is_not_quoted(make):
    s = make(rewarded=(YES60,))
    out = by_listing(books(s, 0))
    assert sides(out[YES40]) == (0, 0, 0, 0)
    assert sides(out[YES60])[1] == 500


def test_a_strike_with_no_spec_is_not_quoted(make):
    s = make()
    s.positions.no_spec.add(LISTINGS[YES40])
    out = by_listing(books(s, 0))
    assert sides(out[YES40]) == (0, 0, 0, 0), "the engine refuses every order on a listing with no spec"
    assert sides(out[YES60])[1] == 500


def test_an_unchanged_quote_is_not_sent_again_until_one_of_its_orders_ends(make):
    s = make()
    books(s, 0)
    assert books(s, 1) == []
    intent = by_listing(books(s, 2, bids=((0.44, 900),)))[YES40]
    assert sides(intent)[0] == pytest.approx(0.44), "the touch moved"
    s.on_execution_report(report(intent, OrderStatus.FILLED, fill=(0.44, 500)))
    again = by_listing(books(s, 3, bids=((0.44, 900),)))
    assert set(again) == {YES40}, "filled: sent again"


def test_a_review_pulls_every_quote_for_the_pull_window(make):
    s = make(reviews=[_review("a", 10)])
    assert sides(by_listing(books(s, 5))[YES40])[1] == 500
    pulled = by_listing(books(s, 11))
    assert all(sides(i) == (0, 0, 0, 0) for i in pulled.values()) and set(pulled) == {YES40, YES60}
    assert books(s, 39) == [], "still pulled"
    assert sides(by_listing(books(s, 41))[YES60])[1] == 500, "30 min after the review"


def test_a_tomatometer_change_pulls_too(make):
    seen = pd.Timestamp(T0 + 10 * MINUTE, tz="UTC")
    s = make(scores=[(FILM, 3, 1, seen)])
    books(s, 5)
    assert all(sides(i) == (0, 0, 0, 0) for i in by_listing(books(s, 11)).values())


GUARD = {"guard_ticks": 3, "guard_seconds": 60, "guard_minutes": 5}


def test_one_strike_moving_too_fast_pulls_every_strike_of_its_film_for_the_guard(make):
    s = make(**GUARD)
    books(s, 0)
    moved = by_listing(s.on_market_data(book_at(YES40, T0 + 2 * MINUTE, bids=((0.48, 900),), asks=((0.50, 900),))))
    assert set(moved) == {YES40, YES60} and all(sides(i) == (0, 0, 0, 0) for i in moved.values())
    assert books(s, 6, listings=(YES60,)) == [], "still guarded"
    assert sides(by_listing(books(s, 8, listings=(YES60,)))[YES60])[1] == 500, "5 min after the move"


def test_a_move_spread_wider_than_the_lookback_does_not_guard(make):
    s = make(**GUARD)
    for minute, bid in ((0, 0.45), (2, 0.46), (4, 0.47), (6, 0.48)):
        out = by_listing(s.on_market_data(book_at(YES40, T0 + minute * MINUTE, bids=((bid, 900),),
                                                  asks=((bid + 0.02, 900),))))
        assert sides(out[YES40])[1] == 500, f"a cent every 2 min, at minute {minute}"


def test_the_guard_pulls_only_the_film_that_moved(make):
    s = make(films=TWO_FILMS, **GUARD)
    books(s, 0, listings=(YES40, YES60, B40, B60))
    s.on_market_data(book_at(YES40, T0 + 2 * MINUTE, bids=((0.48, 900),), asks=((0.50, 900),)))
    other = by_listing(s.on_market_data(book_at(B40, T0 + 3 * MINUTE, bids=((0.44, 900),))))
    assert sides(other[B40])[1] == 500


def test_the_guard_is_off_unless_its_ticks_are_set(make):
    s = make()
    books(s, 0)
    moved = by_listing(s.on_market_data(book_at(YES40, T0 + 2 * MINUTE, bids=((0.48, 900),), asks=((0.50, 900),))))
    assert sides(moved[YES40]) == pytest.approx((0.48, 500, 0.50, 500))


def test_a_negative_guard_is_refused(make):
    with pytest.raises(ValueError):
        make(guard_ticks=-1)


def test_a_side_stops_at_the_position_limit(make):
    s = make()
    books(s, 0)
    s.positions.set(YES40, 500, 0.45)
    s._strike_of[YES40].sent = None
    assert sides(by_listing(books(s, 1))[YES40]) == pytest.approx((0, 0, 0.47, 500))


def test_a_position_is_held_between_fills_and_read_again_after_one(make):
    s = make()
    out = by_listing(books(s, 0))
    s.positions.set(YES40, 500, 0.45)
    s._strike_of[YES40].sent = None
    # The same moment, no fill: the position last read stands.
    assert sides(by_listing(books(s, 0))[YES40]) == pytest.approx((0.45, 500, 0.47, 500))
    s.on_execution_report(report(out[YES40], OrderStatus.PARTIALLY_FILLED, fill=(0.45, 500)))
    assert sides(by_listing(books(s, 0))[YES40]) == pytest.approx((0, 0, 0.47, 500))


def test_a_position_changed_without_a_fill_is_read_within_the_refresh(make):
    s = make()
    books(s, 0)
    s.positions.set(YES40, 500, 0.45)
    s._strike_of[YES40].sent = None
    at = T0 + mod.POSITION_REFRESH_NS
    out = by_listing(s.on_market_data(book_at(YES40, at)))
    assert sides(out[YES40]) == pytest.approx((0, 0, 0.47, 500))


def _program(start_min, end_min, known_min):
    return IncentiveProgram("k1", "liquidity", "KXRT", T0 + start_min * MINUTE, T0 + end_min * MINUTE, 100, "PERIOD",
                            {}, None, T0 + known_min * MINUTE)


def test_a_reward_program_counts_from_its_start_to_its_end_as_known_then(make):
    s = make()
    strike = s._strike_of[YES40]
    # Known from minute 5, paying minutes 10 to 20; a second version from minute 15 ends it at 18.
    s._programs = {YES40: IncentiveTimeline([ProgramVersions("k1", [_program(10, 20, 5), _program(10, 18, 15)])])}
    paying = {m: s._rewarded(strike, T0 + m * MINUTE) for m in (0, 9, 10, 14, 15, 17, 18, 30)}
    assert paying == {0: False, 9: False, 10: True, 14: True, 15: True, 17: True, 18: False, 30: False}


def test_reloaded_reward_programs_are_read_at_once(make):
    s = make()
    strike = s._strike_of[YES40]
    assert s._rewarded(strike, T0)
    s._programs = {YES40: _Timeline(False)}
    assert not s._rewarded(strike, T0)


def test_quotes_past_the_film_budget_keep_only_the_side_that_trades_back(make):
    s = make(max_film_usd=600)
    s.positions.set(YES40, 200, 0.45)
    out = by_listing(books(s, 0))
    # Held: 200 x 0.45 = $90. The quote on 40 adds 300 x 0.45 + 300 x 0.53 = $294, leaving $216: too little for 60.
    assert sides(out[YES40]) == pytest.approx((0.45, 300, 0.47, 500))
    assert sides(out[YES60]) == (0, 0, 0, 0)


def test_live_a_level_we_are_alone_on_is_left_for_the_others_best(make):
    live = make()
    books(live, 0)
    alone = by_listing(books(live, 1, bids=((0.45, 500), (0.44, 900)), listings=(YES40,)))
    assert sides(alone[YES40])[0] == pytest.approx(0.44), "the 500 at 0.45 is ours"
    paper = make(own_orders_in_book=False)
    books(paper, 0)
    assert books(paper, 1, bids=((0.45, 500), (0.44, 900))) == [], "a paper book never shows ours: 500 others"


def test_a_film_without_kalshi_strikes_is_refused(make):
    with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()):
        with pytest.raises(ValueError):
            mod.RtMaker(films=_films((FILM, [LEGS[-1]], None)), rewarded_only=False, reload_seconds=0)


def _all_pulled(intents):
    return all(sides(i) == (0, 0, 0, 0) for i in by_listing(intents).values())


def test_skew_quotes_only_the_bid_once_the_reviews_say_yes(make):
    s = make(reviews=[_review("a", 10), _review("b", 11), _review("c", 12)], signal_mode="skew")
    assert sides(by_listing(books(s, 5))[YES40]) == pytest.approx((0.45, 500, 0.47, 500))
    assert _all_pulled(books(s, 10)), "the first review: no side yet, pulled"
    out = by_listing(books(s, 12))
    assert sides(out[YES40]) == pytest.approx((0.45, 500, 0, 0)), "the third: YES, the bid alone, no pull"
    assert books(s, 600) == [], "still leaning ten hours on"


def test_skew_quotes_only_the_ask_once_the_reviews_say_no(make):
    s = make(reviews=[_review("a", 10, False), _review("b", 11, False), _review("c", 12, False)], signal_mode="skew")
    books(s, 5)
    assert sides(by_listing(books(s, 12))[YES60]) == pytest.approx((0, 0, 0.47, 500))


def test_skew_hours_bounds_the_lean(make):
    s = make(reviews=[_review("a", 10), _review("b", 11), _review("c", 12)], signal_mode="skew", skew_hours=1.0)
    books(s, 12)
    assert sides(by_listing(books(s, 73))[YES40]) == pytest.approx((0.45, 500, 0.47, 500))


def test_skew_a_tomatometer_change_pulls_then_the_lean_resumes(make):
    reviews = [_review("a", 10), _review("b", 11), _review("c", 12)]
    s = make(reviews=reviews, scores=[(FILM, 3, 0, pd.Timestamp(T0 + 20 * MINUTE, tz="UTC"))], signal_mode="skew")
    books(s, 12)
    assert _all_pulled(books(s, 20))
    assert sides(by_listing(books(s, 51))[YES40]) == pytest.approx((0.45, 500, 0, 0))


def test_skew_the_leaned_side_still_stops_at_the_position_limit(make):
    s = make(reviews=[_review("a", 10), _review("b", 11), _review("c", 12)], signal_mode="skew")
    s.positions.set(YES40, 500, 0.45)
    assert sides(by_listing(books(s, 12))[YES40]) == (0, 0, 0, 0)


def test_skew_a_position_against_the_signal_is_traded_back_by_the_leaned_side(make):
    s = make(reviews=[_review("a", 10, False), _review("b", 11, False), _review("c", 12, False)],
             signal_mode="skew")
    s.positions.set(YES40, 300, 0.45)
    assert sides(by_listing(books(s, 12))[YES40]) == pytest.approx((0, 0, 0.47, 500)), "NO: sells the YES held"


def test_pull_mode_ignores_the_signal(make):
    s = make(reviews=[_review("a", 10), _review("b", 11), _review("c", 12)])
    books(s, 5)
    assert _all_pulled(books(s, 12))
    assert sides(by_listing(books(s, 43))[YES40]) == pytest.approx((0.45, 500, 0.47, 500))


def test_an_unknown_signal_mode_or_skew_hours_is_refused(make):
    with pytest.raises(ValueError):
        make(signal_mode="lean")
    with pytest.raises(ValueError):
        make(signal_mode="skew", skew_hours=0)


def test_skew_pulling_on_every_review_sits_out_the_wave_then_leans(make):
    s = make(reviews=[_review("a", 10), _review("b", 11), _review("c", 12)], signal_mode="skew",
             pull_every_review=True)
    books(s, 5)
    assert _all_pulled(books(s, 12)), "the review that gave the side pulls too"
    assert sides(by_listing(books(s, 43))[YES40]) == pytest.approx((0.45, 500, 0, 0)), "then the bid alone"


# ---- many films in one session ---------------------------------------------------------------------------------------

BOTH_BOOKS = (YES40, YES60, B40, B60)


def test_a_book_update_decides_only_its_own_films_quotes(make):
    s = make(films=TWO_FILMS)
    assert set(by_listing(books(s, 0, listings=BOTH_BOOKS))) == set(BOTH_BOOKS)
    moved = by_listing(books(s, 1, bids=((0.44, 900),), listings=(YES40,)))
    assert set(moved) == {YES40}, "film B's quotes are left resting"


def test_each_film_has_its_own_budget(make):
    s = make(films=TWO_FILMS, max_film_usd=600)
    s.positions.set(YES40, 200, 0.45)
    out = by_listing(books(s, 0, listings=BOTH_BOOKS))
    assert sides(out[YES60]) == (0, 0, 0, 0), "film A's budget is spent"
    assert sides(out[B40]) == pytest.approx((0.45, 500, 0.47, 500)), "film B's is its own"


def test_a_review_pulls_only_its_own_film(make):
    s = make(films=TWO_FILMS, reviews=[_review("a", 10)])
    books(s, 5, listings=BOTH_BOOKS)
    out = by_listing(books(s, 11, listings=BOTH_BOOKS))
    assert set(out) == {YES40, YES60} and _all_pulled(out.values()), "film B keeps quoting, unchanged"


def test_a_pull_ending_on_one_film_is_acted_on_at_any_films_next_record(make):
    s = make(films=TWO_FILMS, reviews=[_review("b", 10, film=FILM_B)])
    books(s, 5, listings=BOTH_BOOKS)
    assert set(by_listing(books(s, 11, listings=(YES40,)))) == {B40, B60}, "pulled on film A's record"
    back = by_listing(books(s, 41, listings=(YES40,)))
    assert set(back) == {B40, B60}, "back 30 min on, though film B's books haven't moved"
    assert sides(back[B60]) == pytest.approx((0.45, 500, 0.47, 500))


def test_an_ended_order_requotes_its_film_at_the_next_record_of_any_film(make):
    s = make(films=TWO_FILMS)
    intent = by_listing(books(s, 0, listings=BOTH_BOOKS))[B40]
    s.on_execution_report(report(intent, OrderStatus.CANCELED))
    assert set(by_listing(books(s, 1, listings=(YES40,)))) == {B40}


def test_a_film_past_its_close_is_cancelled_once_then_left_alone(make):
    s = make(films=_films((FILM, LEGS, None), (FILM_B, LEGS_B, 20)))
    books(s, 5, listings=BOTH_BOOKS)
    closed = by_listing(books(s, 21, listings=(YES40,)))
    assert set(closed) == {B40, B60} and _all_pulled(closed.values())
    assert books(s, 22, listings=BOTH_BOOKS) == []


def test_reward_programs_are_read_in_batches(make, tmp_path):
    empty = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["film_id", "liked", "not_liked", "first_seen_at"]).to_parquet(empty)
    asked = []
    with patch.object(mod, "PROGRAMS_BATCH", 3), patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()), \
            patch.object(mod, "load_incentive_timelines", lambda registry, ids: asked.append(list(ids)) or {}):
        mod.RtMaker(films=TWO_FILMS, reload_seconds=0, programs_reload_seconds=0,
                    reviews_source=str(empty), scores_source=str(empty))
    assert asked == [[YES40, YES60, B40], [B60]]


def test_one_feed_thread_and_one_programs_thread_serve_every_film(make):
    before = [t.name for t in threading.enumerate()]
    make(films=TWO_FILMS, reload_seconds=3600, programs_reload_seconds=3600)
    started = Counter(t.name for t in threading.enumerate()) - Counter(before)
    assert started == Counter({"rt-feed": 1, "rt-programs": 1})


def test_a_film_named_twice_or_a_listing_in_two_films_is_refused(make):
    with pytest.raises(ValueError):
        make(films=_films((FILM, LEGS, None), (FILM, LEGS_B, None)))
    with pytest.raises(ValueError):
        make(films=_films((FILM, LEGS, None), (FILM_B, LEGS[:1], None)))


def test_a_rejected_strike_waits_before_it_is_quoted_again(make):
    s = make(films=TWO_FILMS)
    intent = by_listing(books(s, 0, listings=BOTH_BOOKS))[B40]
    reject = report(intent, OrderStatus.REJECTED)
    s.on_execution_report(reject)
    seconds = MINUTE // 60
    assert books(s, 0, listings=(YES40,)) == [], "refused: not resent on the next record"
    out = s.on_market_data(_Book(YES40, [(0.45, 900)], [(0.47, 900)], T0 + 4 * seconds))
    assert out == [], "nor inside the backoff"
    again = by_listing(s.on_market_data(_Book(YES40, [(0.45, 900)], [(0.47, 900)], T0 + 6 * seconds)))
    assert set(again) == {B40}, "quoted again once the backoff has passed, though its own book hasn't moved"



# ---- keeping away from fills, in the session ------------------------------------------------------------------------

SECOND = MINUTE // 60


def test_behind_rests_each_side_a_tick_back_from_the_price_it_would_join(make):
    s = make(queue_policy="behind")
    assert sides(by_listing(books(s, 0))[YES60]) == pytest.approx((0.44, 500, 0.48, 500))


def test_inventory_skew_steps_the_bid_back_while_long(make):
    s = make(skew_unit=250, max_skew_ticks=2)
    s._position_view.set(YES60, 300, 0.45)
    assert sides(by_listing(books(s, 0))[YES60]) == pytest.approx((0.44, 200, 0.47, 500))


def test_a_side_near_the_front_is_pulled_and_sent_again_from_the_back(make):
    s = make(queue_policy="requeue", requeue_min_ahead=200, own_orders_in_book=False)
    books(s, 0)
    # 800 of the 900 ahead of our bid trade away: 100 left ahead of it.
    pulled = by_listing(books(s, 1, bids=((0.45, 100),), listings=(YES60,), trade=(0.45, 800)))
    assert sides(pulled[YES60]) == pytest.approx((0, 0, 0.47, 500)), "the bid is pulled, the ask stays"
    again = by_listing(books(s, 1 + 1 / 60, bids=((0.45, 100),), listings=(YES60,)))
    assert sides(again[YES60]) == pytest.approx((0.45, 500, 0.47, 500)), "sent again, behind the 100 there"
    assert s._strike_of[YES60].bid_spot.ahead == 100


def test_a_requeued_side_waits_out_the_cooldown_before_its_next_requeue(make):
    s = make(queue_policy="requeue", requeue_min_ahead=200, requeue_cooldown_seconds=5, own_orders_in_book=False)
    books(s, 0)
    books(s, 1, bids=((0.45, 100),), listings=(YES60,), trade=(0.45, 800))
    books(s, 1 + 1 / 60, bids=((0.45, 100),), listings=(YES60,))
    assert books(s, 1 + 2 / 60, bids=((0.45, 100),), listings=(YES60,)) == [], "near the front, but requeued 1 s ago"
    later = by_listing(books(s, 1 + 7 / 60, bids=((0.45, 100),), listings=(YES60,)))
    assert sides(later[YES60])[:2] == (0, 0)


def test_a_partial_fill_puts_its_side_at_the_front(make):
    s = make(queue_policy="requeue", requeue_min_ahead=1, own_orders_in_book=False)
    intent = by_listing(books(s, 0))[YES60]
    s.on_execution_report(report(intent, OrderStatus.PARTIALLY_FILLED, fill=(0.45, 100), leaves=400))
    assert s._strike_of[YES60].bid_spot.ahead == 0 and s._strike_of[YES60].bid_spot.size == 400
    assert s._strike_of[YES60].ask_spot.ahead == 900, "the fill's price says it was the bid"
    s._position_view.set(YES60, 100, 0.45)
    assert sides(by_listing(books(s, 1, listings=(YES60,)))[YES60]) == pytest.approx((0, 0, 0.47, 500))
