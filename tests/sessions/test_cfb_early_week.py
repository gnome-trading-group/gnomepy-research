"""
CFBEarlyWeek: the decision at 72 hours out, resting entries and their re-pegs, the entry close, resting exits, the
taker exit before kickoff, restarts, the session's end, and its metrics.

The registry, books, session control and metrics are the in-memory stand-ins of test_cfb_pregame_strategy; positions
are a plain table of held shares. Intents are recorded by a stand-in with Intent's fields, since a local engine JAR may
predate post-only orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd
import pytest

from gnomepy import OrderStatus, OrderType, Scales, Side
from gnomepy_research.sessions.cfb_pregame import early_week as mod
from tests.sessions.test_cfb_pregame_strategy import LISTINGS, _Book, _Control, _Metrics, _Registry, _Report

P, S = Scales.PRICE, Scales.SIZE
HOUR = 3_600_000_000_000
KICKOFF = pd.Timestamp("2026-11-07T17:00:00Z")
# game_id -> (home listing, away listing, kickoff)
GAMES = {1: (101, 102, KICKOFF), 2: (201, 202, KICKOFF + pd.Timedelta(hours=3))}
# Home .48/.50 and away .50/.52: the market has the home team at .49.
HOME_BOOK = ([(0.48, 500)], [(0.50, 500)])
AWAY_BOOK = ([(0.50, 500)], [(0.52, 500)])


@dataclass
class _Intent:
    exchange_id: int
    security_id: int
    strategy_id: int = 0
    bid_price: int = 0
    bid_size: int = 0
    ask_price: int = 0
    ask_size: int = 0
    take_side: Side | None = None
    take_size: int = 0
    take_order_type: OrderType | None = None
    take_limit_price: int = 0
    post_only: bool = False


@dataclass
class _Pos:
    net_quantity: int
    avg_entry_price: int = 0
    total_fees: float = 0.0


class _Held:
    def __init__(self):
        self.shares: dict = {}

    def set(self, listing, shares):
        self.shares[LISTINGS[listing]] = shares

    def get_position(self, exchange_id, security_id):
        q = self.shares.get((exchange_id, security_id))
        return _Pos(int(q * S)) if q else None


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(mod, "Intent", _Intent)

    def _make(p_home=(0.70, 0.50), games=(1, 2), missing=(), off_protocol=(), **kw):
        args = [{"game_id": g, "listing_home": GAMES[g][0], "listing_away": GAMES[g][1],
                 "kickoff": GAMES[g][2].isoformat(), "p_home": p, "off_protocol": g in off_protocol}
                for g, p in zip(games, p_home)]
        with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry(missing)):
            strat = mod.CFBEarlyWeek(games=args, model="e11", **kw)
        strat._position_view = _Held()
        strat._control = _Control()
        strat._metric_recorder = _Metrics()
        strat.register_metrics()
        return strat
    return _make


def _at(hours_before, game=1):
    return GAMES[game][2].value - int(hours_before * HOUR)


def _tick(strat, listing, book, hours_before, game=1):
    bids, asks = book
    return strat.on_market_data(_Book(LISTINGS[listing], bids, asks, _at(hours_before, game)))


def _both(strat, hours_before, game=1, home=HOME_BOOK, away=AWAY_BOOK):
    """The away leg's book, then the home leg's; returns every intent either update produced."""
    h, a = GAMES[game][:2]
    return _tick(strat, a, away, hours_before, game) + _tick(strat, h, home, hours_before, game)


def _for(intents, listing):
    return [i for i in intents if (i.exchange_id, i.security_id) == LISTINGS[listing]]


def test_early_mid_averages_the_usable_legs_in_home_terms():
    assert mod.early_mid(0.48, 0.50, 0.50, 0.52, 0.06) == pytest.approx(0.49)
    assert mod.early_mid(0.48, 0.50, 0.30, 0.60, 0.06) == pytest.approx(0.49), "a leg wider than 6c is left out"
    assert mod.early_mid(float("nan"), 0.50, 0.50, 0.52, 0.06) == pytest.approx(0.49), "a one-sided leg is left out"
    assert pd.isna(mod.early_mid(0.40, 0.60, 0.30, 0.60, 0.06))


def test_nothing_happens_before_the_decision_time(make):
    s = make()
    assert _both(s, 73) == []
    assert s.metrics.streams["cfb_early_week_decisions"].rows == []


def test_at_72_hours_it_rests_a_post_only_bid_on_the_side_the_model_prefers(make):
    out = _for(_both(make(), 72), 101)
    assert len(out) == 1
    o = out[0]
    assert o.bid_price == round(0.48 * P) and o.bid_size == S, "one contract at the home team's best bid"
    assert o.ask_size == 0 and o.take_size == 0 and o.post_only


def test_the_away_side_when_the_model_is_below_the_market(make):
    out = _both(make(p_home=(0.40, 0.50)), 72)
    assert _for(out, 101) == []
    [o] = _for(out, 102)
    assert o.bid_price == round(0.50 * P), "the away team's best bid"


def test_a_gap_under_five_points_is_left_alone_for_good(make):
    s = make(p_home=(0.53, 0.50))
    assert _both(s, 72) == []
    assert _both(s, 66, home=([(0.40, 500)], [(0.42, 500)]), away=([(0.58, 500)], [(0.60, 500)])) == [], \
        "the decision is made once, at the decision time"
    [d] = [r for r in s.metrics.streams["cfb_early_week_decisions"].rows if r["game_id"] == 1]
    assert d["side"] == mod.NO_SIDE and d["market_mid"] == pytest.approx(0.49) and d["gap"] == pytest.approx(0.04)


def test_it_waits_for_a_usable_book_but_not_past_the_late_limit(make):
    wide = ([(0.40, 500)], [(0.60, 500)])
    s = make()
    assert _both(s, 72, home=wide, away=wide) == [], "no usable mid yet"
    assert len(_for(_both(s, 68), 101)) == 1, "a usable book four hours later still counts"
    late = make()
    assert _both(late, 72, home=wide, away=wide) == []
    assert _both(late, 65) == [], "seven hours late: left alone"
    [d] = [r for r in late.metrics.streams["cfb_early_week_decisions"].rows if r["game_id"] == 1]
    assert d["late"] == 1 and d["side"] == mod.NO_SIDE


def test_an_off_protocol_game_decides_at_its_first_look_however_late_and_is_flagged(make):
    s = make(off_protocol=(1,))
    [o] = _for(_both(s, 30), 101)
    assert o.bid_price == round(0.48 * P) and o.bid_size == S
    [d] = [r for r in s.metrics.streams["cfb_early_week_decisions"].rows if r["game_id"] == 1]
    assert d["off_protocol"] == 1 and d["late"] == 0 and d["hours_before"] == pytest.approx(30)


def test_an_off_protocol_game_still_needs_its_entry_window(make):
    s = make(off_protocol=(1,))
    assert _for(_both(s, 5), 101) == []
    [d] = [r for r in s.metrics.streams["cfb_early_week_decisions"].rows if r["game_id"] == 1]
    assert d["late"] == 1 and d["off_protocol"] == 1


def test_a_protocol_game_is_flagged_as_such(make):
    s = make()
    _both(s, 72)
    [d] = [r for r in s.metrics.streams["cfb_early_week_decisions"].rows if r["game_id"] == 1]
    assert d["off_protocol"] == 0 and d["hours_before"] == pytest.approx(72)


def test_the_bid_moves_to_the_best_bid_every_six_hours_and_not_between(make):
    s = make()
    _both(s, 72)
    higher = ([(0.52, 500)], [(0.54, 500)])
    assert _for(_both(s, 67, home=higher), 101) == [], "five hours on: the bid stays put"
    [o] = _for(_both(s, 66, home=higher), 101)
    assert o.bid_price == round(0.52 * P) and o.bid_size == S


def test_an_unfilled_bid_is_cancelled_six_hours_out_and_the_game_ends(make):
    s = make()
    _both(s, 72)
    [o] = _for(_both(s, 6), 101)
    assert o.bid_size == 0 and o.ask_size == 0 and o.take_size == 0, "a bare intent cancels the resting bid"
    assert _for(_both(s, 1), 101) == []


def test_a_filled_entry_is_offered_at_the_best_ask_from_six_hours_out_and_repegged_hourly(make):
    s = make()
    _both(s, 72)
    s.positions.set(101, 1.0)
    assert _for(_both(s, 30), 101) == [], "filled: no more bids"
    [o] = _for(_both(s, 6), 101)
    assert o.ask_price == round(0.50 * P) and o.ask_size == S and o.bid_size == 0 and o.post_only
    assert _for(_both(s, 5.5), 101) == []
    [o] = _for(_tick(s, 101, ([(0.55, 500)], [(0.57, 500)]), 5), 101)
    assert o.ask_price == round(0.57 * P), "an hour on, the ask moves to the best ask"


def test_whatever_is_held_five_minutes_out_is_sold_into_the_bid(make):
    s = make()
    _both(s, 72)
    s.positions.set(101, 1.0)
    _both(s, 6)
    [o] = _for(_both(s, 5 / 60), 101)
    assert o.take_side == Side.ASK and o.take_size == S and o.take_order_type == OrderType.LIMIT
    assert o.take_limit_price == round(0.48 * P) and o.ask_size == 0
    assert _for(_both(s, 1 / 60), 101) == [], "one exit only"


def test_a_sold_position_ends_the_game_without_a_taker_exit(make):
    s = make()
    _both(s, 72)
    s.positions.set(101, 1.0)
    _both(s, 6)
    s.positions.set(101, 0.0)
    assert _for(_both(s, 5 / 60), 101) == []


def test_a_restart_with_a_position_picks_up_the_held_side(make):
    s = make(p_home=(0.50, 0.50))
    s.positions.set(102, 1.0)
    [o] = _for(_both(s, 6), 102)
    assert o.ask_price == round(0.52 * P) and o.ask_size == S, "offered on the held away leg, whatever the model said"


def test_the_session_ends_once_every_game_is_done(make):
    s = make(p_home=(0.70, 0.70))
    _both(s, 72)
    _both(s, 72, game=2)
    _both(s, 6)
    assert not s.is_exiting, "game 2 kicks off three hours later and its bid still rests"
    _both(s, 6, game=2)
    assert s._control.exits == [("every game is done", False)]


def test_the_session_waits_for_the_taker_exit_to_finish(make):
    s = make(games=(1,), p_home=(0.70,))
    _both(s, 72)
    s.positions.set(101, 1.0)
    _both(s, 6)
    _both(s, 5 / 60)
    assert not s.is_exiting
    s.on_execution_report(_Report(*LISTINGS[101], OrderStatus.FILLED))
    s.positions.set(101, 0.0)
    _both(s, 4 / 60)
    assert s._control.exits == [("every game is done", False)]


def test_a_rejected_resting_order_is_resent_on_the_next_update(make):
    s = make()
    _both(s, 72)
    s.on_execution_report(_Report(*LISTINGS[101], OrderStatus.REJECTED))
    assert len(_for(_both(s, 71.9), 101)) == 1


def test_orders_are_recorded(make):
    s = make()
    _both(s, 72)
    rows = s.metrics.streams["cfb_early_week_orders"].rows
    assert rows == [{"timestamp": _at(72), "game_id": 1, "side": mod.HOME, "action": mod.BID, "price": 0.48,
                     "shares": 1.0, "held": 0.0, "best_bid": 0.48, "best_ask": 0.50}]
    [d] = [r for r in s.metrics.streams["cfb_early_week_decisions"].rows if r["game_id"] == 1]
    assert d["p_model"] == 0.70 and d["side"] == mod.HOME and d["late"] == 0


def test_a_game_missing_a_listing_is_dropped_and_an_empty_slate_refused(make):
    assert list(make(missing=(201,))._games) == [1]
    with pytest.raises(ValueError):
        make(missing=(101, 201))


@pytest.mark.parametrize("kw", [{"contracts": 0}, {"min_gap": 0}, {"decide_hours": 5}, {"taker_exit_minutes": 400}])
def test_bad_parameters_are_refused(make, kw):
    with pytest.raises(ValueError):
        make(**kw)
