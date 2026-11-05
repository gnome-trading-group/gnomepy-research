"""CS2SeriesMM: resting bids around a fair price anchored to the liquid venues. Quote placement (join a liquid best bid
while it leaves the edge, set the price inside a wide book, never cross), one- and two-sided quoting, wide-only, sizing,
sharing and unwinding, keeping queue position, the reference mid, the pull rules, and only changes being sent. The
stand-ins are test_cs2_series's: a model pricing team A at 0.40 and a blend that keeps only the model, so team B's fair
price is 0.60."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from gnomepy import Scales
from gnomepy_research.models.cs2.e12.blend import MarketBlend
from gnomepy_research.sessions.cs2_series import strategy as base
from gnomepy_research.sessions.cs2_series.decision import kelly_shares
from gnomepy_research.sessions.cs2_series__mm.strategy import CS2SeriesMM
from tests.sessions.conftest import FakeStreams
from tests.sessions.test_cs2_series import (EARLY, LEGS, LISTINGS, STANDIN_B, _Book, _Control, _feature, _models,
                                            _Positions, _Registry, _Spec, _before, _states, _write, K_NS, MINUTE)

P, S = Scales.PRICE, Scales.SIZE
MARKET_ONLY = MarketBlend(market=1.0, model=0.0, intercept=0.0)


@pytest.fixture
def make(tmp_path):
    def _make(features=None, blend=None, **kw):
        f, m = _write(tmp_path, features or [_feature(EARLY)], _states())
        with patch.object(base, "RegistryClient", lambda *a, **k: _Registry(_Spec())), \
                patch.object(base, "load_models", lambda *paths: _models(**({"blend": blend} if blend else {}))):
            kw.setdefault("bankroll_usd", 10_000.0)
            s = CS2SeriesMM(match_id=7, legs=LEGS, reload_minutes=0, features_source=f, matches_source=m, **kw)
        s._position_view = _Positions()
        s._streams = FakeStreams(LISTINGS)
        s._control = _Control()
        return s
    return _make


def _tick(s, listing, bids, asks, minutes_before):
    return s.on_market_data(_Book(listing, asks, K_NS - int(minutes_before * MINUTE), bids))


def _quotes(intents):
    """{(exchange, security): (bid price, bid shares)}, (0, 0) for a cancel."""
    return {(i.exchange_id, i.security_id): (i.bid_price / P, i.bid_size / S) for i in intents}


def _market(s, minutes_before=30, a=((0.46, 500),), a_asks=((0.48, 500),), b=((0.51, 500),), b_asks=((0.53, 500),)):
    """Polymarket's books, team A's then team B's: team A mid 0.47, so the model (team A at 0.40) favours team B. Returns
    what both updates sent, in order."""
    return _tick(s, 101, list(a), list(a_asks), minutes_before + 0.2) + _tick(s, 102, list(b), list(b_asks), minutes_before)


def test_one_sided_joins_the_best_bid_of_the_favoured_team_only(make):
    q = _quotes(_market(make()))
    assert set(q) == {LISTINGS[102]}, "team B is underpriced; team A is not quoted"
    price, shares = q[LISTINGS[102]]
    assert price == pytest.approx(0.51) and shares > 0, "0.51 leaves far more than 2c below 0.60: join the best bid"


def test_a_bid_rests_below_the_best_bid_when_joining_would_leave_less_than_the_edge(make):
    q = _quotes(_market(make(), b=((0.59, 500),), b_asks=((0.62, 500),)))
    assert q[LISTINGS[102]][0] == pytest.approx(0.58), "fair 0.60 less the 2c edge, no maker fee on Polymarket"


def test_quotes_never_reach_the_ask(make):
    for intent in _market(make(), b=((0.45, 500),), b_asks=((0.47, 500),)):
        assert intent.bid_price / P < 0.47


def test_a_team_is_bid_on_every_qualifying_venue_with_its_size_shared_equally(make):
    s = make(kelly_fraction=0.02)
    q = _quotes(_tick(s, 202, [(0.54, 500)], [(0.56, 500)], 30.5) + _market(s))
    assert q[LISTINGS[202]][0] == pytest.approx(0.54) and q[LISTINGS[102]][0] == pytest.approx(0.51)
    kelly = kelly_shares(0.60, 0.54, 0.02 * 10_000.0, 0.0, 0.0)
    assert q[LISTINGS[202]][1] + q[LISTINGS[102]][1] <= kelly + 1e-6, "together within the Kelly size at the best price"
    assert q[LISTINGS[202]][1] == pytest.approx(q[LISTINGS[102]][1], abs=0.02), "an equal share each"


def test_shared_bids_stay_within_the_stake_cap(make):
    s = make(max_stake_usd=100.0)
    q = _quotes(_tick(s, 202, [(0.54, 500)], [(0.56, 500)], 30.5) + _market(s))
    assert len(q) == 2 and sum(p * n for p, n in q.values()) <= 100.0 + 1e-6


def test_two_sided_also_quotes_the_other_team_below_its_fair_price(make):
    q = _quotes(_market(make(two_sided=True)))
    assert q[LISTINGS[101]][0] == pytest.approx(0.38) and q[LISTINGS[102]][0] == pytest.approx(0.51)


def test_size_stays_within_the_stake_cap_counting_resting_quotes(make):
    q = _quotes(_market(make(max_stake_usd=50.0, two_sided=True)))
    assert sum(p * n for p, n in q.values()) <= 50.0 + 1e-6


def test_a_fill_on_the_other_team_is_quoted_to_unwind(make):
    s = make(two_sided=True, kelly_fraction=0.01)
    s.positions.fill(101, shares=200, price=0.38)
    q = _quotes(_market(s))
    assert q[LISTINGS[102]][1] >= 200, "buying 200 team B shares below fair locks the team A position in"


def test_only_changed_quotes_are_sent(make):
    s = make()
    assert _market(s) != []
    assert _tick(s, 102, [(0.51, 500)], [(0.53, 500)], 29) == [], "nothing the quote depends on moved"


def test_a_new_model_price_pulls_every_quote(make):
    s = make(features=[_feature(EARLY), _feature(_before(20), lineup_b=STANDIN_B)])
    assert _market(s, 30)
    pulled = _tick(s, 101, [(0.46, 500)], [(0.48, 500)], 19)
    assert _quotes(pulled)[LISTINGS[102]] == (0.0, 0.0), "the lineup changed: cancel before repricing"
    assert _tick(s, 102, [(0.52, 500)], [(0.53, 500)], 18.9), "then quote again on the new price"


def test_quotes_are_pulled_before_kickoff_and_the_session_ends_once_none_rest(make):
    s = make(pull_before_kickoff_min=5)
    _market(s, 30)
    assert _quotes(_tick(s, 102, [(0.51, 500)], [(0.53, 500)], 4))[LISTINGS[102]] == (0.0, 0.0)
    _tick(s, 102, [(0.51, 500)], [(0.53, 500)], -0.1)
    assert s._control.exits == [("match 7 reached its entry cutoff", False)]


# ---- keeping queue position -----------------------------------------------------------------------------------------

def test_a_resting_quote_stays_put_while_the_fair_price_moves_a_little(make):
    s = make()
    _market(s)
    assert _tick(s, 101, [(0.47, 500)], [(0.49, 500)], 29) == [], \
        "the mid moved a cent; the bid at 0.51 is still at the top and within the edge: modifying would lose its place"


def test_a_quote_moves_up_to_rejoin_a_best_bid_that_rose_past_it(make):
    s = make()
    _market(s)
    q = _quotes(_tick(s, 102, [(0.53, 500)], [(0.55, 500)], 29))
    assert q[LISTINGS[102]][0] == pytest.approx(0.53)


def test_a_quote_moves_down_when_the_edge_requires(make):
    s = make(two_sided=True, blend=MARKET_ONLY)
    assert _quotes(_market(s))[LISTINGS[102]][0] == pytest.approx(0.51), "fair = the mid: team B at 0.53, join at 0.51"
    q = _quotes(_tick(s, 101, [(0.48, 500)], [(0.50, 500)], 29))
    assert q[LISTINGS[102]][0] == pytest.approx(0.49), "team A's mid rose to 0.49: team B's fair is 0.51, so 0.49 at most"


def test_the_favoured_team_does_not_flip_on_a_small_lean(make):
    s = make()
    _market(s, a=((0.41, 500),), a_asks=((0.42, 500),))
    q = _quotes(_tick(s, 101, [(0.40, 500)], [(0.41, 500)], 29))
    assert LISTINGS[101] not in q, "the model leans the other way by less than a cent: keep quoting team B"


def test_the_reference_is_polymarket_while_liquid_else_the_tightest_liquid_venue(make):
    s = make()
    _tick(s, 302, [(0.70, 500)], [(0.72, 500)], 31)
    _market(s)
    assert s._reference_mid(K_NS - 29 * MINUTE) == pytest.approx(0.47)
    _tick(s, 101, [(0.10, 500)], [(0.88, 500)], 29.5)
    _tick(s, 102, [(0.12, 500)], [(0.90, 500)], 29.4)
    assert s._reference_mid(K_NS - 29 * MINUTE) == pytest.approx(0.29), "Polymarket 78c wide: Polymarket US, 1 - 0.71"


def test_the_last_reference_holds_for_a_minute_when_no_venue_is_liquid(make):
    s = make()
    _market(s)
    _tick(s, 101, [(0.10, 500)], [(0.88, 500)], 29.5)
    _tick(s, 102, [(0.12, 500)], [(0.90, 500)], 29.4)
    assert s._reference_mid(K_NS - int(29.3 * MINUTE)) == pytest.approx(0.47)
    assert s._reference_mid(K_NS - 27 * MINUTE) is None, "past a minute: no reference, the model alone"


def test_a_resting_quote_keeps_its_place_inside_the_edge_slack(make):
    s = make(blend=MARKET_ONLY)
    q = _quotes(_market(s, a=((0.41, 500),), a_asks=((0.43, 500),), b=((0.56, 500),), b_asks=((0.60, 500),)))
    assert q[LISTINGS[102]][0] == pytest.approx(0.56), "fair 0.58, so 0.56 leaves exactly the 2c edge"
    assert _tick(s, 101, [(0.412, 500)], [(0.432, 500)], 29) == [], "1.8c of edge left: within the slack, stay"
    assert _quotes(_tick(s, 101, [(0.43, 500)], [(0.45, 500)], 28))[LISTINGS[102]][0] < 0.56, "past the slack: move down"


# ---- setting the price on a wide book -------------------------------------------------------------------------------

WIDE_A, WIDE_A_ASK, WIDE_B, WIDE_B_ASK = [(0.10, 500)], [(0.88, 500)], [(0.12, 500)], [(0.90, 500)]


def _wide_market(s):
    """Polymarket US liquid (team A 0.32/0.33), Polymarket International wide on both teams."""
    out = _tick(s, 301, [(0.32, 500)], [(0.33, 500)], 30.5)
    return out + _tick(s, 101, WIDE_A, WIDE_A_ASK, 30.2) + _tick(s, 102, WIDE_B, WIDE_B_ASK, 30)


def test_a_wide_book_gets_both_teams_bid_inside_the_spread_around_the_liquid_venues_price(make):
    q = _quotes(_wide_market(make(half_spread=0.03, blend=MARKET_ONLY)))
    assert q[LISTINGS[101]][0] == pytest.approx(0.29), "fair 0.325 (Polymarket US mid) less 3c, floored to the tick"
    assert q[LISTINGS[102]][0] == pytest.approx(0.64), "team B: fair 0.675 less 3c"
    assert 0.10 < q[LISTINGS[101]][0] < 0.88 and 0.12 < q[LISTINGS[102]][0] < 0.90, "inside the wide spread"


def test_with_no_liquid_venue_the_wide_book_is_quoted_around_the_model(make):
    s = make(half_spread=0.03)
    q = _quotes(_tick(s, 101, WIDE_A, WIDE_A_ASK, 30.2) + _tick(s, 102, WIDE_B, WIDE_B_ASK, 30))
    assert q[LISTINGS[101]][0] == pytest.approx(0.37) and q[LISTINGS[102]][0] == pytest.approx(0.57)


def test_a_wide_book_bid_stays_below_a_liquid_venues_ask_for_the_team(make):
    q = _quotes(_wide_market(make(half_spread=0.03)))
    assert q[LISTINGS[101]][0] < 0.33, "the model says 0.40 but Polymarket US sells team A at 0.33: never bid above it"


def test_wide_only_never_joins_a_liquid_book(make):
    q = _quotes(_wide_market(make(half_spread=0.03, wide_only=True, two_sided=True)))
    assert set(q) == {LISTINGS[101], LISTINGS[102]}, "only Polymarket International, the wide book; Polymarket US is left alone"


def test_wide_only_with_every_book_liquid_quotes_nothing(make):
    assert _market(make(wide_only=True, two_sided=True)) == []


def test_both_teams_of_a_wide_book_share_the_budget(make):
    q = _quotes(_wide_market(make(half_spread=0.03, wide_only=True, max_stake_usd=60.0)))
    a, b = q[LISTINGS[101]], q[LISTINGS[102]]
    assert a[1] > 0 and b[1] > 0, "the first team quoted doesn't take the whole stake"
    assert a[0] * a[1] + b[0] * b[1] <= 60.0 + 1e-6
