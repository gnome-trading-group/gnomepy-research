"""CS2CrossVenueArb: pair pricing, whole-contract sizing, exposure with shorts, completion, cooldown and backoff.

The registry is patched out and positions, streams and market data are plain objects with the engine's interface, so
nothing here needs a JVM.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gnomepy import OrderStatus, OrderType, Scales, Side
from gnomepy_research.sessions.cs2_cross_venue_arb import strategy as mod
from gnomepy_research.sessions.cs2_cross_venue_arb.decision import (
    BUY,
    SELL,
    Level,
    exposure,
    plan_completion,
    plan_pair,
    team_ladder,
    whole,
)
from tests.sessions.conftest import FakePositions, FakeStreams, stream_of

P, S = Scales.PRICE, Scales.SIZE
SEC = 1_000_000_000
PMI, KAL, PMUS = 4, 5, 6
# listing -> (exchange, security); team A first on every venue.
LISTINGS = {101: (PMI, 1101), 102: (PMI, 1102), 201: (KAL, 1201), 202: (KAL, 1202), 301: (PMUS, 1301), 302: (PMUS, 1302)}
LEGS = [{"venue": "polymarket", "listing_team_a": 101, "listing_team_b": 102},
        {"venue": "kalshi", "listing_team_a": 201, "listing_team_b": 202},
        {"venue": "polymarket_us", "listing_team_a": 301, "listing_team_b": 302}]
RATES = {PMI: (0.05, 0.0), KAL: (0.07, 0.0), PMUS: (0.0695, 0.0)}
MIN_SIZE = {101: 5 * S, 102: 5 * S}


def fee(rate, p):
    return rate * p * (1 - p)


# ---- decision maths ----------------------------------------------------------------------

def test_sell_route_costs_one_minus_the_bid():
    ladder = team_ladder(201, [Level(0.60, 10)], 202, [Level(0.45, 7)], 0.07, sell_rival=True)
    assert [(round(r.cost, 6), r.direction, r.listing_id, r.order_price) for r in ladder] == [
        (0.55, SELL, 202, 0.45), (0.60, BUY, 201, 0.60)]


def test_no_sell_route_where_teams_mirror_one_market():
    ladder = team_ladder(101, [Level(0.60, 10)], 102, [Level(0.45, 7)], 0.05, sell_rival=False)
    assert [r.direction for r in ladder] == [BUY]


def test_pair_walks_only_while_each_pair_clears_the_edge():
    a = team_ladder(101, [Level(0.40, 100), Level(0.45, 100)], 102, [], 0.05, False)
    b = team_ladder(301, [Level(0.55, 50), Level(0.57, 500)], 302, [], 0.0695, False)
    plan = plan_pair(a, b, 0.01, 1000)
    first = 1 - 0.40 - 0.55 - fee(0.05, 0.40) - fee(0.0695, 0.55)
    second = 1 - 0.40 - 0.57 - fee(0.05, 0.40) - fee(0.0695, 0.57)
    assert second < 0.01 < first
    assert plan.contracts == 50
    assert plan.profit == pytest.approx(50 * first)
    assert [(o.listing_id, o.contracts, o.limit) for o in plan.orders_a] == [(101, 50, 0.40)]
    assert [(o.listing_id, o.contracts, o.limit) for o in plan.orders_b] == [(301, 50, 0.55)]


def test_no_plan_without_edge():
    a = team_ladder(101, [Level(0.50, 100)], 102, [], 0.05, False)
    b = team_ladder(301, [Level(0.50, 100)], 302, [], 0.0695, False)
    assert plan_pair(a, b, 0.01, 1000) is None


def test_pair_caps_at_max_contracts():
    a = team_ladder(101, [Level(0.30, 1000)], 102, [], 0.05, False)
    b = team_ladder(301, [Level(0.30, 1000)], 302, [], 0.0695, False)
    assert plan_pair(a, b, 0.01, 120).contracts == 120


def test_whole_drops_fractions_and_respects_minimum_size():
    a = team_ladder(101, [Level(0.40, 7.6)], 102, [], 0.05, False)
    b = team_ladder(301, [Level(0.50, 100)], 302, [], 0.0695, False)
    plan = whole(plan_pair(a, b, 0.01, 1000), {101: 5})
    assert plan.contracts == 7 and plan.orders_a[0].contracts == 7 and plan.orders_b[0].contracts == 7
    small = team_ladder(101, [Level(0.40, 4.5)], 102, [], 0.05, False)
    assert whole(plan_pair(small, b, 0.01, 1000), {101: 5}) is None


def test_exposure_counts_shorts_for_the_rival():
    assert exposure([(0, 10), (1, -4), (1, 3)]) == (14, 3)


def test_completion_stops_at_the_slippage_cap():
    ladder = team_ladder(301, [Level(0.50, 5), Level(0.53, 5), Level(0.60, 50)], 302, [], 0.0695, False)
    orders = plan_completion(ladder, 20, 0.05, {})
    assert [(o.listing_id, o.contracts, o.limit) for o in orders] == [(301, 10, 0.53)]


# ---- the strategy ------------------------------------------------------------------------

@dataclass
class _Spec:
    min_size: int


class _Registry:
    def get_listing(self, *, listing_id, **_):
        return [SimpleNamespace(exchange_id=LISTINGS[listing_id][0], security_id=LISTINGS[listing_id][1])]

    def get_listing_spec(self, *, listing_id, **_):
        return [_Spec(MIN_SIZE.get(listing_id, 0))]


class _Book:
    def __init__(self, listing_id, asks, bids, t, bad=False):
        self.stream_id = stream_of(listing_id)
        self._asks, self._bids, self.event_timestamp, self.is_bad_data = asks, bids, t, bad

    def ask_price(self, i):
        return round(self._asks[i][0] * P) if i < len(self._asks) else 0

    def ask_size(self, i):
        return round(self._asks[i][1] * S) if i < len(self._asks) else 0

    def bid_price(self, i):
        return round(self._bids[i][0] * P) if i < len(self._bids) else 0

    def bid_size(self, i):
        return round(self._bids[i][1] * S) if i < len(self._bids) else 0


@pytest.fixture
def make():
    def _make(**kwargs):
        kwargs.setdefault("taker_delay_seconds", {})
        kwargs.setdefault("legs", LEGS)
        with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()):
            s = mod.CS2CrossVenueArb(**kwargs)
        s._position_view = FakePositions(fee_rates=RATES)
        s._streams = FakeStreams(LISTINGS)
        return s
    return _make


def feed(s, books, t=0):
    out = []
    for listing_id, asks, bids in books:
        out += s.on_market_data(_Book(listing_id, asks, bids, t))
    return out


def report(listing_id, status, t=0):
    eid, sid = LISTINGS[listing_id]
    return SimpleNamespace(exchange_id=eid, security_id=sid, order_status=status, timestamp_recv=t)


# No pair clears the edge until Polymarket US offers team B cheaply.
QUIET = [
    (101, [(0.50, 100)], [(0.48, 100)]),
    (102, [(0.53, 100)], [(0.49, 100)]),
    (201, [(0.50, 100)], [(0.48, 100)]),
    (202, [(0.52, 100)], [(0.49, 100)]),
    (301, [(0.50, 100)], [(0.48, 100)]),
]
CHEAP_B = (302, [(0.45, 30)], [(0.43, 30)])
GONE_B = (302, [(0.60, 30)], [(0.43, 30)])


def test_sends_both_legs_at_once(make):
    s = make()
    intents = feed(s, QUIET + [CHEAP_B])
    sent = {(i.exchange_id, i.security_id): i for i in intents}
    assert set(sent) == {LISTINGS[101], LISTINGS[302]}
    for i in intents:
        assert i.take_side == Side.BID and i.take_order_type == OrderType.LIMIT and i.take_size == 30 * S
    assert sent[LISTINGS[101]].take_limit_price == round(0.50 * P)
    assert sent[LISTINGS[302]].take_limit_price == round(0.45 * P)


def test_quiet_books_send_nothing(make):
    assert feed(make(), QUIET) == []


def test_nothing_while_a_leg_is_in_flight(make):
    s = make()
    feed(s, QUIET + [CHEAP_B])
    assert feed(s, [CHEAP_B], t=60 * SEC) == []


def test_unequal_fills_are_completed_on_the_cheapest_venue(make):
    s = make()
    feed(s, QUIET + [CHEAP_B])
    s.positions.set(LISTINGS[101], 30 * S)
    s.positions.set(LISTINGS[302], 10 * S)
    feed(s, [GONE_B])
    assert s.on_execution_report(report(101, OrderStatus.FILLED)) == []
    intents = s.on_execution_report(report(302, OrderStatus.CANCELED))
    # 20 short on team B, gone from Polymarket US: Kalshi's A bid at 0.48 holds B for 0.52, level with its B ask,
    # and both beat Polymarket International's 0.53; at a tie the sale goes first.
    assert [(i.security_id, i.take_side, i.take_size, i.take_limit_price) for i in intents] == [
        (1201, Side.ASK, 20 * S, round(0.48 * P))]


def test_completion_ignores_the_cooldown(make):
    s = make(cooldown_seconds=3600)
    feed(s, QUIET + [CHEAP_B])
    s.positions.set(LISTINGS[101], 30 * S)
    s.on_execution_report(report(302, OrderStatus.CANCELED))
    assert s.on_execution_report(report(101, OrderStatus.FILLED)) != []


def test_cooldown_then_backoff_until_the_top_moves(make):
    # At this edge only Polymarket International A + Polymarket US B clears, so the backoff is all that stops it.
    s = make(cooldown_seconds=10, min_edge=0.018)
    feed(s, QUIET + [CHEAP_B])
    s.positions.set(LISTINGS[101], 30 * S)
    s.positions.set(LISTINGS[302], 30 * S)
    s.on_execution_report(report(101, OrderStatus.FILLED))
    s.on_execution_report(report(302, OrderStatus.CANCELED))
    assert feed(s, [CHEAP_B], t=5 * SEC) == []
    assert feed(s, [CHEAP_B], t=20 * SEC) == []
    moved = (302, [(0.44, 30)], [(0.43, 30)])
    assert {(i.exchange_id, i.security_id) for i in feed(s, [moved], t=21 * SEC)} == {LISTINGS[101], LISTINGS[302]}


def test_stops_at_the_locked_cap(make):
    s = make(max_locked_usd=30)
    s.positions.set(LISTINGS[101], 30 * S)
    s.positions.set(LISTINGS[302], 30 * S)
    assert feed(s, QUIET + [CHEAP_B]) == []


def test_bad_data_books_are_not_traded(make):
    s = make()
    feed(s, QUIET)
    assert s.on_market_data(_Book(302, [(0.55, 30)], [(0.53, 30)], 0, bad=True)) == []


def test_kalshi_sell_route_takes_a_cheap_team(make):
    s = make()
    # Kalshi's team A bid at 0.58 holds team B for 0.42 as NO; its B ask is 0.60.
    books = QUIET[:2] + [(201, [(0.62, 100)], [(0.58, 40)]), (202, [(0.60, 100)], [(0.36, 100)])] + QUIET[4:]
    intents = feed(s, books)
    assert sorted((i.security_id, i.take_side, i.take_size, i.take_limit_price) for i in intents) == [
        (1101, Side.BID, 40 * S, round(0.50 * P)), (1201, Side.ASK, 40 * S, round(0.58 * P))]


def test_kalshi_sell_route_off_when_not_listed(make):
    s = make(sell_rival_venues=())
    books = QUIET[:2] + [(201, [(0.62, 100)], [(0.58, 40)]), (202, [(0.60, 100)], [(0.36, 100)])] + QUIET[4:]
    assert feed(s, books) == []


def test_slow_venue_leg_goes_first_and_the_other_waits(make):
    s = make(taker_delay_seconds={"polymarket": 1.0})
    first = feed(s, QUIET + [CHEAP_B], t=0)
    assert {(i.exchange_id, i.security_id) for i in first} == {LISTINGS[101]}
    assert feed(s, [CHEAP_B], t=SEC // 2) == []
    later = feed(s, [CHEAP_B], t=SEC)
    assert {(i.exchange_id, i.security_id) for i in later} == {LISTINGS[302]}
    assert later[0].take_limit_price == round(0.45 * P)


def test_no_stagger_without_a_delayed_venue(make):
    s = make(taker_delay_seconds={"polymarket": 1.0})
    books = QUIET[:2] + [(201, [(0.62, 100)], [(0.58, 40)]), (202, [(0.60, 100)], [(0.36, 100)])] + QUIET[4:]
    books[0] = (101, [(0.70, 100)], [(0.68, 100)])
    books += [(302, [(0.70, 100)], [(0.10, 100)]), (301, [(0.50, 100)], [(0.48, 100)])]
    sent = feed(s, books)
    assert {i.exchange_id for i in sent} == {KAL, PMUS}


def test_one_venue_ends_the_session_instead_of_failing(make):
    s = make(legs=LEGS[:1], match_id=2380001)
    exits = []
    s.request_exit = lambda reason, **kw: exits.append(reason) or True
    assert feed(s, [(101, [(0.40, 100)], [(0.38, 100)])]) == []
    assert len(exits) == 1 and "nothing to arb" in exits[0]


def test_a_venue_named_twice_is_a_config_error():
    with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()):
        with pytest.raises(ValueError):
            mod.CS2CrossVenueArb(legs=[LEGS[0], LEGS[0]])


def test_launcher_match_id_is_accepted(make):
    assert make(match_id=2380001).match_id == 2380001
