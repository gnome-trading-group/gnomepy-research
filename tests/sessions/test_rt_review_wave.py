"""rt_review_wave: the wave and signal rules, the reviews feed, and the session's orders against fake books."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from gnomepy import OrderStatus, Scales, Side
from gnomepy_research.sessions.rt_review_wave import strategy as mod
from gnomepy_research.models.rt.review_feed import Review, ReviewFeed, reviews_of
from gnomepy_research.sessions.rt_review_wave.decision import NO, YES, Level, signal, take, wave_start
from tests.sessions.conftest import FakeFeeSchedule, FakeStreams, stream_of

P, S = Scales.PRICE, Scales.SIZE
MINUTE = 60_000_000_000
HOUR = 60 * MINUTE
T0 = pd.Timestamp("2026-11-18T14:00:00Z").value
FILM = "the-hunger-games-sunrise-on-the-reaping"
FEE = 0.07
# venue -> exchange id; listing ids: venue digit, strike, side (1 YES, 2 NO).
EXCHANGES = {"polymarket": 4, "kalshi": 5, "polymarket_us": 6}
STRIKES = (60, 80)
LEGS = [{"venue": v, "strike": k, "yes_listing": int(f"{e}{k}1"), "no_listing": int(f"{e}{k}2")}
        for v, e in EXCHANGES.items() for k in STRIKES]
LISTINGS = {leg[key]: (EXCHANGES[leg["venue"]], leg[key] + 10_000) for leg in LEGS for key in ("yes_listing", "no_listing")}


def r(i, seen_min, fresh=True, published_min=None):
    return Review(f"u{i}", T0 + seen_min * MINUTE, T0 + (seen_min if published_min is None else published_min) * MINUTE,
                  fresh)


# ---- the wave and the signal ---------------------------------------------------------------------------------------

def test_the_wave_starts_when_the_third_review_of_a_cluster_is_seen():
    reviews = [r(1, 0), r(2, 1), r(3, 2), r(4, 30)]
    assert wave_start(reviews, 3, 6 * HOUR, 6 * HOUR) == T0 + 2 * MINUTE
    assert wave_start(reviews[:2], 3, 6 * HOUR, 6 * HOUR) is None


def test_reviews_spread_wider_than_the_cluster_are_no_wave():
    assert wave_start([r(1, 0), r(2, 7 * 60), r(3, 14 * 60)], 3, 6 * HOUR, 6 * HOUR) is None


def test_old_reviews_seen_late_never_start_a_wave():
    # A backlog given its verdicts at once (Claude back after an outage, or a restart): published days before.
    backlog = [r(i, 0, published_min=-3 * 24 * 60) for i in range(5)]
    assert wave_start(backlog, 3, 6 * HOUR, 6 * HOUR) is None
    assert wave_start(backlog + [r(9, 1), r(10, 2), r(11, 3)], 3, 6 * HOUR, 6 * HOUR) == T0 + 3 * MINUTE


def test_the_signal_is_the_fresh_share_against_the_baseline():
    fresh = [r(1, 0), r(2, 0), r(3, 0)]
    assert signal(fresh, 0.667, 0.10, 3).side == YES
    rotten = [r(1, 0, False), r(2, 0, False), r(3, 0)]
    assert signal(rotten, 0.667, 0.10, 3).side == NO
    near = [r(1, 0), r(2, 0), r(3, 0, False)]
    assert signal(near, 0.667, 0.10, 3).side == 0, "2/3 is the baseline itself"
    assert signal(fresh[:2], 0.667, 0.10, 3).side == 0, "too few reviews"


def test_a_take_is_bounded_by_the_book_the_contracts_left_and_the_money_left():
    asks = [Level(0.50, 40), Level(0.52, 100), Level(0.60, 500)]
    t = take(asks, 0.53, 1000, 1e9, FEE, 1.0)
    assert (t.contracts, t.limit_price) == (140, 0.53), "nothing above the limit"
    assert take(asks, 0.53, 60, 1e9, FEE, 1.0).contracts == 60
    unit = 0.50 + FEE * 0.25
    assert take(asks, 0.53, 1000, 10 * unit, FEE, 1.0).contracts == 10
    assert take(asks, 0.53, 0.5, 1e9, FEE, 1.0) is None, "less than a lot"


# ---- the reviews feed ----------------------------------------------------------------------------------------------

def _row(url, seen_min, fresh=True, is_review=True, published_min=None):
    return {"url": url, "film_id": FILM, "published_at": pd.Timestamp(T0 + (seen_min if published_min is None
                                                                            else published_min) * MINUTE, tz="UTC"),
            "first_seen_at": pd.Timestamp(T0 + seen_min * MINUTE, tz="UTC"), "is_review": is_review, "fresh": fresh}


def test_the_feed_keeps_reviews_with_a_verdict_as_of_when_their_verdict_was_seen(tmp_path):
    rows = pd.DataFrame([_row("a", 0, fresh=None, is_review=None), _row("a", 5, True), _row("b", 1, False),
                         _row("c", 2, None, None), _row("d", 3, True, is_review=False)])
    rows.loc[len(rows)] = {**_row("x", 0), "film_id": "other-film"}
    path = tmp_path / "rt_reviews.parquet"
    rows.to_parquet(path)
    scores = tmp_path / "rt_scores.parquet"
    pd.DataFrame([{"film_id": FILM, "liked": 3, "not_liked": 1, "first_seen_at": pd.Timestamp(T0 + 4 * MINUTE, tz="UTC")},
                  {"film_id": "other-film", "liked": 90, "not_liked": 9,
                   "first_seen_at": pd.Timestamp(T0, tz="UTC")}]).to_parquet(scores)
    feed = ReviewFeed(FILM, str(path), reload_seconds=0, scores_source=str(scores))
    assert (feed.counted(T0 + 3 * MINUTE), feed.counted(T0 + 4 * MINUTE)) == (0, 4)
    assert [x.url for x in feed.known(T0 + 4 * MINUTE)] == ["b"]
    assert [(x.url, x.fresh) for x in feed.known(T0 + 5 * MINUTE)] == [("b", False), ("a", True)]
    assert reviews_of(pd.DataFrame()) == []


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


@dataclass
class _Pos:
    net_quantity: int
    avg_entry_price: int
    total_fees: float


class _Positions:
    def __init__(self):
        self.filled: dict = {}

    def fill(self, listing, contracts, price):
        key = LISTINGS[listing]
        q, avg, fees = self.filled.get(key, (0, 0, 0.0))
        add = round(contracts * S)
        self.filled[key] = (q + add, int((q * avg + add * int(price * P)) / (q + add)),
                            fees + contracts * FEE * price * (1 - price) * P)

    def get_position(self, exchange_id, security_id):
        f = self.filled.get((exchange_id, security_id))
        return _Pos(*f) if f else None

    def get_effective_quantity(self, exchange_id, security_id):
        return self.filled.get((exchange_id, security_id), (0, 0, 0.0))[0]

    def fee_schedule(self, exchange_id, security_id):
        return FakeFeeSchedule(taker=FEE, maker=0.0)


class _Control:
    def __init__(self):
        self.exits = []

    def requestExit(self, reason, flatten, timeout_millis, notify=False):
        self.exits.append((reason, flatten))
        return len(self.exits) == 1

    def isExiting(self):
        return bool(self.exits)


class _Book:
    def __init__(self, listing, asks, ts):
        self.stream_id = stream_of(listing)
        self.event_timestamp = ts
        self._asks = asks
        self._bids = [(round(asks[0][0] - 0.02, 2), 100)] if asks else []

    def ask_price(self, i):
        return int(self._asks[i][0] * P) if i < len(self._asks) else 0

    def ask_size(self, i):
        return int(self._asks[i][1] * S) if i < len(self._asks) else 0

    def bid_price(self, i):
        return int(self._bids[i][0] * P) if i < len(self._bids) else 0

    def bid_size(self, i):
        return int(self._bids[i][1] * S) if i < len(self._bids) else 0


WAVE = [_row("a", 0), _row("b", 1), _row("c", 2)]  # three fresh: the wave starts at T0 + 2 min


@pytest.fixture
def make(tmp_path):
    def _make(rows=WAVE, scores=(), **kw):
        path, scores_path = tmp_path / "rt_reviews.parquet", tmp_path / "rt_scores.parquet"
        pd.DataFrame(rows).to_parquet(path)
        pd.DataFrame(list(scores), columns=["film_id", "liked", "not_liked", "first_seen_at"]).to_parquet(scores_path)
        with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()):
            s = mod.RtReviewWave(film_id=FILM, legs=kw.pop("legs", LEGS), reload_seconds=0,
                                 reviews_source=str(path), scores_source=str(scores_path), **kw)
        s._position_view = _Positions()
        s._streams = FakeStreams(LISTINGS)
        s._control = _Control()
        return s
    return _make


def yes(venue, strike):
    return next(leg["yes_listing"] for leg in LEGS if leg["venue"] == venue and leg["strike"] == strike)


def no(venue, strike):
    return next(leg["no_listing"] for leg in LEGS if leg["venue"] == venue and leg["strike"] == strike)


def book_all(s, minute, yes_ask=0.40, no_ask=0.62, size=500):
    """Every listing's book at once; returns every intent the updates drew."""
    out = []
    for leg in LEGS:
        out += s.on_market_data(_Book(leg["yes_listing"], [(yes_ask, size)], T0 + minute * MINUTE))
        out += s.on_market_data(_Book(leg["no_listing"], [(no_ask, size)], T0 + minute * MINUTE))
    return out


def listing_of(intent):
    return next(lid for lid, key in LISTINGS.items() if key == (intent.exchange_id, intent.security_id))


def report(s, intent, status=OrderStatus.FILLED, filled=None):
    s.on_execution_report(SimpleNamespace(exchange_id=intent.exchange_id, security_id=intent.security_id,
                                          filled_qty=filled if filled is not None else intent.take_size,
                                          order_status=status))


def test_nothing_is_bought_before_the_wave(make):
    s = make()
    assert book_all(s, 1) == [], "two reviews by T0 + 1 min: no wave yet"


def test_a_fresh_wave_buys_yes_on_every_strike_of_every_venue_at_most_3c_through(make):
    s = make()
    out = book_all(s, 3)
    assert sorted(listing_of(o) for o in out) == sorted(yes(v, k) for v in EXCHANGES for k in STRIKES)
    assert all(o.take_side == Side.BID and o.take_limit_price == int(0.43 * P) for o in out)
    assert all(o.take_size == 100 * S for o in out), "100 contracts a strike"


def test_a_rotten_wave_buys_no(make):
    s = make(rows=[_row("a", 0, False), _row("b", 1, False), _row("c", 2)])
    out = book_all(s, 3)
    assert sorted(listing_of(o) for o in out) == sorted(no(v, k) for v in EXCHANGES for k in STRIKES)


def test_a_share_near_the_baseline_buys_nothing(make):
    s = make(rows=[_row("a", 0), _row("b", 1), _row("c", 2, False)])
    assert book_all(s, 3) == []


def test_strikes_priced_outside_3_to_97c_are_left_alone(make):
    s = make()
    assert book_all(s, 3, yes_ask=0.98) == []
    s2 = make()
    assert book_all(s2, 3, yes_ask=0.02) == []


def test_each_venue_spends_at_most_its_cap(make):
    s = make(max_usd_per_venue=60.0)
    out = book_all(s, 3)
    unit = 0.40 + FEE * 0.40 * 0.60
    by_venue = {}
    for o in out:
        venue = next(v for v, e in EXCHANGES.items() if e == o.exchange_id)
        by_venue[venue] = by_venue.get(venue, 0.0) + o.take_size / S * unit
    assert by_venue and all(usd <= 60.0 + 1e-9 for usd in by_venue.values())


def test_a_held_strike_is_topped_up_only_to_its_cap(make):
    s = make()
    first = book_all(s, 3)
    for o in first:
        s.positions.fill(listing_of(o), o.take_size / S * 0.6, 0.40)
        report(s, o, OrderStatus.FILLED)
    again = book_all(s, 4, yes_ask=0.41)
    assert again and all(o.take_size == 40 * S for o in again)


def test_one_order_in_flight_per_listing_and_none_resent_on_an_unchanged_book(make):
    s = make()
    assert len(book_all(s, 3)) == 6
    assert book_all(s, 3) == [], "orders still in flight"


def test_a_short_order_backs_its_listing_off_until_its_top_of_book_moves(make):
    s = make()
    out = book_all(s, 3)
    target = next(o for o in out if listing_of(o) == yes("kalshi", 60))
    for o in out:
        report(s, o, OrderStatus.CANCELED if o is target else OrderStatus.FILLED, filled=0)
    assert not any(listing_of(o) == yes("kalshi", 60) for o in book_all(s, 4)), "the same top: backed off"
    moved = s.on_market_data(_Book(yes("kalshi", 60), [(0.41, 500)], T0 + 5 * MINUTE))
    assert [listing_of(o) for o in moved] == [yes("kalshi", 60)]


def test_once_yes_is_held_a_signal_turned_rotten_buys_no_no(make):
    rows = WAVE + [_row(f"r{i}", 5 + i, False) for i in range(6)]
    s = make(rows=rows)
    for o in book_all(s, 3):
        s.positions.fill(listing_of(o), o.take_size / S, 0.40)
        report(s, o)
    later = book_all(s, 12, yes_ask=0.30, no_ask=0.72)
    assert later == [], "3 fresh then 6 rotten: the signal says NO, but YES is held"


def test_the_session_ends_itself_after_the_window_once_nothing_is_in_flight(make):
    s = make()
    out = book_all(s, 3)
    book_all(s, 2 + 15)
    assert s._control.exits == [], "orders still in flight"
    for o in out:
        report(s, o)
    book_all(s, 2 + 16)
    assert len(s._control.exits) == 1 and s._control.exits[0][1] is False, "keeps its positions"


def test_a_session_started_after_its_wave_ends_at_once(make):
    s = make()
    assert book_all(s, 60) == [] and len(s._control.exits) == 1


def test_another_stream_wakes_the_session_between_book_updates(make):
    s = make()
    assert book_all(s, 1) == []
    clock = SimpleNamespace(event_timestamp=T0 + 3 * MINUTE)
    out = s.on_stream_event(999, clock)
    assert len(out) == 6, "the wave started at T0 + 2 min; the books from T0 + 1 min still stand"


def test_a_restarted_session_buys_only_what_its_caps_leave(make):
    s = make()
    for leg in LEGS:
        s.positions.fill(leg["yes_listing"], 100, 0.40)
    assert book_all(s, 3) == []


def test_unknown_venues_and_empty_legs_are_refused(make):
    with pytest.raises(ValueError):
        make(legs=[])
    with pytest.raises(ValueError):
        make(legs=[{**LEGS[0], "venue": "betfair"}])


def _score(liked, not_liked, minute):
    return {"film_id": FILM, "liked": liked, "not_liked": not_liked,
            "first_seen_at": pd.Timestamp(T0 + minute * MINUTE, tz="UTC")}


def test_a_film_whose_tomatometer_already_has_a_score_is_not_traded_and_its_session_ends(make):
    # Victorian Psycho on 2026-11-12: RT counted 60 reviews while only two of the outlets' were seen.
    s = make(scores=[_score(47, 13, -60)])
    assert book_all(s, 3) == [] and len(s._control.exits) == 1


def test_a_few_reviews_on_rt_still_trade_but_rt_reaching_the_limit_mid_window_stops_new_orders(make):
    s = make(scores=[_score(2, 0, -60), _score(30, 5, 8)])
    first = book_all(s, 3)
    assert len(first) == 6, "2 reviews counted: the market can't know the score yet"
    for o in first:
        report(s, o, OrderStatus.CANCELED, filled=0)
    assert book_all(s, 9, yes_ask=0.45) == [], "RT counted 35 at T0 + 8 min"
    assert len(s._control.exits) == 1
