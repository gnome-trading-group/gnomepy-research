"""CS2CrossVenueArbPMMaker: bid pricing from the hedge, re-quoting, pulling quotes and hedging a fill."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gnomepy import OrderStatus, Scales, Side
from gnomepy_research.sessions.cs2_cross_venue_arb.decision import Level, team_ladder
from gnomepy_research.sessions.cs2_cross_venue_arb__pm_maker import strategy as mod
from gnomepy_research.sessions.cs2_cross_venue_arb__pm_maker.quoting import MoveWindow, maker_bid
from tests.sessions.conftest import FakePositions, FakeStreams
from tests.sessions.test_cs2_cross_venue_arb import LEGS, LISTINGS, RATES, _Book, _Registry

P, S = Scales.PRICE, Scales.SIZE


def test_bid_leaves_min_edge_after_the_hedge():
    hedge = team_ladder(302, [Level(0.40, 50), Level(0.405, 30), Level(0.45, 500)], 301, [], 0.0695, False)
    price, size = maker_bid(hedge, 200, 0.01, 0.0, 0.01, best_ask=0.60, depth_band=0.01)
    worst = 0.405 + 0.0695 * 0.405 * 0.595
    assert size == 80
    assert price == pytest.approx(0.56)
    assert price <= 1 - worst - 0.01


def test_bid_stays_under_the_ask():
    hedge = team_ladder(302, [Level(0.30, 50)], 301, [], 0.0695, False)
    assert maker_bid(hedge, 200, 0.01, 0.0, 0.01, best_ask=0.55, depth_band=0.01)[0] == pytest.approx(0.54)


def test_move_window_forgets_old_prices():
    w = MoveWindow(5)
    w.add(0, 0.60)
    w.add(int(2e9), 0.50)
    assert w.range() == pytest.approx(0.10)
    w.add(int(6e9), 0.52)
    assert w.range() == pytest.approx(0.02)


def test_no_bid_without_a_hedge():
    assert maker_bid([], 200, 0.01, 0.0, 0.01, 0.5, 0.01) is None


@pytest.fixture
def make():
    def _make(**kwargs):
        with patch("gnomepy_research.sessions.cs2_cross_venue_arb.strategy.RegistryClient",
                   lambda *a, **k: _Registry()):
            s = mod.CS2CrossVenueArbPMMaker(legs=LEGS, **kwargs)
        s._position_view = FakePositions(fee_rates=RATES)
        s._streams = FakeStreams(LISTINGS)
        return s
    return _make


BOOKS = [
    (101, [(0.60, 100)], [(0.40, 100)]),
    (102, [(0.62, 100)], [(0.38, 100)]),
    (201, [(0.55, 100)], [(0.40, 100)]),
    (202, [(0.50, 100)], [(0.42, 100)]),
    (301, [(0.56, 100)], [(0.50, 100)]),
    (302, [(0.48, 60)], [(0.44, 100)]),
]


def feed(s, books, t=0):
    out = []
    for listing_id, asks, bids in books:
        out += s.on_market_data(_Book(listing_id, asks, bids, t))
    return out


def quotes(intents):
    return {i.security_id: (i.bid_price / P, i.bid_size / S, i.post_only) for i in intents}


def test_bids_each_team_post_only(make):
    s = make(min_edge=0.01)
    q = quotes(feed(s, BOOKS))
    # Team A's hedge is team B: Polymarket US asks 0.48x60 is the cheapest all-in.
    hedge_b = 0.48 + 0.0695 * 0.48 * 0.52
    assert q[1101][0] == pytest.approx(int((1 - hedge_b - 0.01) * 100) / 100) and q[1101][1] == 60 and q[1101][2]
    assert 1102 in q


def test_unchanged_books_send_nothing(make):
    s = make()
    feed(s, BOOKS)
    assert feed(s, BOOKS) == []


def test_fill_pulls_quotes_and_hedges_elsewhere(make):
    s = make()
    feed(s, BOOKS)
    s.positions.set(LISTINGS[101], 20 * S)
    eid, sid = LISTINGS[101]
    out = s.on_execution_report(SimpleNamespace(exchange_id=eid, security_id=sid, order_status=OrderStatus.PARTIALLY_FILLED,
                                                filled_qty=20 * S, timestamp_recv=0))
    pulled = [i for i in out if i.exchange_id == 4]
    assert {i.security_id for i in pulled} == {1101, 1102} and all(i.bid_size == 0 for i in pulled)
    takes = [i for i in out if i.take_size]
    assert [(i.security_id, i.take_side, i.take_size) for i in takes] == [(1302, Side.BID, 20 * S)]


def test_rejected_bid_is_not_resent_unchanged(make):
    s = make()
    sent = quotes(feed(s, BOOKS))
    eid, sid = LISTINGS[101]
    again = s.on_execution_report(SimpleNamespace(exchange_id=eid, security_id=sid, order_status=OrderStatus.REJECTED,
                                                  filled_qty=0, timestamp_recv=0))
    assert sid not in quotes(again)
    assert 1101 in sent


def test_jumping_hedge_pulls_the_bid_until_it_settles(make):
    s = make(calm_move=0.03, calm_seconds=5)
    assert 1101 in quotes(feed(s, BOOKS))
    moved = [b if b[0] != 302 else (302, [(0.40, 60)], [(0.38, 100)]) for b in BOOKS]
    pulled = quotes(feed(s, moved, t=int(1e9)))
    assert pulled[1101][1] == 0
    assert 1101 in quotes(feed(s, moved, t=int(7e9))) and quotes(feed(s, moved, t=int(7e9))) == {}


def test_without_calm_move_the_bid_follows_the_hedge(make):
    s = make()
    feed(s, BOOKS)
    moved = [b if b[0] != 302 else (302, [(0.40, 60)], [(0.38, 100)]) for b in BOOKS]
    assert quotes(feed(s, moved, t=int(1e9)))[1101][1] > 0
