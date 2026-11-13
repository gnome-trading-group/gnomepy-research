"""rt_maker: the quoting rules, and the session's quotes against fake books, reviews and reward programs."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from gnomepy import OrderStatus, Scales
from gnomepy_research.sessions.rt_maker import strategy as mod
from gnomepy_research.sessions.rt_maker.quotes import NONE, Level, Quote, adds, join_price, lots, quote, reducing
from tests.sessions.conftest import FakeStreams, stream_of

P, S = Scales.PRICE, Scales.SIZE
MINUTE = 60_000_000_000
T0 = pd.Timestamp("2026-11-14T14:00:00Z").value
FILM = "ebenezer"
KALSHI, PM = 5, 4
STRIKES = (40, 60)
LEGS = ([{"venue": "kalshi", "strike": k, "yes_listing": 5000 + k, "no_listing": 5100 + k} for k in STRIKES]
        + [{"venue": "polymarket", "strike": 60, "yes_listing": 4060, "no_listing": 4160}])
LISTINGS = {leg[key]: (KALSHI if leg["venue"] == "kalshi" else PM, leg[key] + 10_000)
            for leg in LEGS for key in ("yes_listing", "no_listing")}
YES40, YES60 = 5040, 5060


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


# ---- the session ---------------------------------------------------------------------------------------------------

@dataclass
class _Listing:
    exchange_id: int
    security_id: int


@dataclass
class _Spec:
    tick_size: int = P // 100
    lot_size: int = S
    min_size: int = 0


class _Registry:
    def get_listing(self, *, listing_id, **_):
        return [_Listing(*LISTINGS[listing_id])]

    def get_listing_spec(self, *, listing_id, **_):
        return [_Spec()]


class _Timeline:
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

    def set(self, listing, contracts, price):
        self.held[LISTINGS[listing]] = _Pos(round(contracts * S), round(price * P))

    def get_position(self, exchange_id, security_id):
        return self.held.get((exchange_id, security_id))


class _Control:
    def isExiting(self):
        return False


class _Book:
    def __init__(self, listing, bids, asks, ts):
        self.stream_id = stream_of(listing)
        self.event_timestamp = ts
        self._bids, self._asks = bids, asks

    def bid_price(self, i):
        return int(round(self._bids[i][0] * P)) if i < len(self._bids) else 0

    def bid_size(self, i):
        return int(self._bids[i][1] * S) if i < len(self._bids) else 0

    def ask_price(self, i):
        return int(round(self._asks[i][0] * P)) if i < len(self._asks) else 0

    def ask_size(self, i):
        return int(self._asks[i][1] * S) if i < len(self._asks) else 0


def _review(url, seen_min):
    ts = pd.Timestamp(T0 + seen_min * MINUTE, tz="UTC")
    return {"url": url, "film_id": FILM, "published_at": ts, "first_seen_at": ts, "is_review": True, "fresh": True}


@pytest.fixture
def make(tmp_path):
    def _make(reviews=(), scores=(), rewarded=(YES40, YES60), **kw):
        path, scores_path = tmp_path / "rt_reviews.parquet", tmp_path / "rt_scores.parquet"
        pd.DataFrame(list(reviews) or [{**_review("old", -24 * 60), "film_id": "other-film"}]).to_parquet(path)
        pd.DataFrame(list(scores), columns=["film_id", "liked", "not_liked", "first_seen_at"]).to_parquet(scores_path)
        timelines = {lid: _Timeline(lid in rewarded) for lid in (YES40, YES60)}
        with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()), \
                patch.object(mod, "load_incentive_timelines", lambda registry, ids: timelines):
            s = mod.RtMaker(film_id=FILM, legs=LEGS, reload_seconds=0, programs_reload_seconds=0,
                            reviews_source=str(path), scores_source=str(scores_path), **kw)
        s._position_view = _Positions()
        s._streams = FakeStreams(LISTINGS)
        s._control = _Control()
        return s
    return _make


def books(s, minute, bids=((0.45, 900),), asks=((0.47, 900),)):
    out = []
    for lid in (YES40, YES60, 4060):
        out += s.on_market_data(_Book(lid, list(bids), list(asks), T0 + minute * MINUTE))
    return out


def by_listing(intents):
    return {next(lid for lid, key in LISTINGS.items() if key == (i.exchange_id, i.security_id)): i for i in intents}


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


def test_an_unchanged_quote_is_not_sent_again_until_one_of_its_orders_ends(make):
    s = make()
    books(s, 0)
    assert books(s, 1) == []
    intent = by_listing(books(s, 2, bids=((0.44, 900),)))[YES40]
    assert sides(intent)[0] == pytest.approx(0.44), "the touch moved"
    s.on_execution_report(SimpleNamespace(exchange_id=intent.exchange_id, security_id=intent.security_id,
                                          order_status=OrderStatus.FILLED))
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


def test_a_side_stops_at_the_position_limit(make):
    s = make()
    books(s, 0)
    s.positions.set(YES40, 500, 0.45)
    s._strike_of[YES40].sent = None
    assert sides(by_listing(books(s, 1))[YES40]) == pytest.approx((0, 0, 0.47, 500))


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
    alone = by_listing(books(live, 1, bids=((0.45, 500), (0.44, 900))))
    assert sides(alone[YES40])[0] == pytest.approx(0.44), "the 500 at 0.45 is ours"
    paper = make(own_orders_in_book=False)
    books(paper, 0)
    assert books(paper, 1, bids=((0.45, 500), (0.44, 900))) == [], "a paper book never shows ours: 500 others"


def test_a_film_without_kalshi_strikes_is_refused(make):
    with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()):
        with pytest.raises(ValueError):
            mod.RtMaker(film_id=FILM, legs=[LEGS[-1]], rewarded_only=False, reload_seconds=0)
