"""
CFBPregame: entries against the anchored and raw prices, cutoffs and the slate's exit, skip rules, stake caps,
restart safety, games it cannot trade, and its metrics.

The strategy resolves listings through RegistryClient in __init__, so the registry is replaced with an in-memory
stand-in; the prediction feed reads real parquet extracts; positions, market data, the session control and metrics
are plain objects with the engine's interface, as in test_cs2_series.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd
import pytest

from gnomepy import OrderStatus, OrderType, Scales, Side
from gnomepy_research.sessions.cfb_pregame import strategy as mod
from tests.sessions.conftest import FakeFeeSchedule

P, S = Scales.PRICE, Scales.SIZE
MINUTE = 60_000_000_000
KICKOFF = pd.Timestamp("2026-10-24T19:30:00Z")
EARLY = KICKOFF - pd.Timedelta(hours=6)
# Predictions for the trading tests are written here, inside max_prediction_age of every tick they make.
T0 = KICKOFF - pd.Timedelta(hours=3)
# game_id -> (home listing, away listing, kickoff); listing -> (exchange_id, security_id)
GAMES = {1: (101, 102, KICKOFF), 2: (201, 202, KICKOFF + pd.Timedelta(hours=1))}
LISTINGS = {101: (5, 1101), 102: (5, 1102), 201: (5, 1201), 202: (5, 1202)}
# Two-sided books on both legs, consistent with each other: home .48/.50, away .50/.52.
HOME_BOOK = ([(0.48, 1000)], [(0.50, 1000)])
AWAY_BOOK = ([(0.50, 1000)], [(0.52, 1000)])


def _row(at, game_id=1, p_home=0.70, week=10, home_qb_p=0.8, away_qb_p=0.8, home_rank=20, away_rank=40,
         home_newcomer=False, away_newcomer=False):
    return {"game_id": game_id, "season": 2026, "week": week, "season_type": "regular", "kickoff": GAMES[game_id][2],
            "home_id": 10 * game_id, "away_id": 10 * game_id + 1, "home_team": "Home", "away_team": "Away",
            "neutral_site": False, "p_home": p_home, "p_home_lo90": p_home - 0.1, "p_home_hi90": p_home + 0.1,
            "home_qb_id": "1", "away_qb_id": "2", "home_qb_p": home_qb_p, "away_qb_p": away_qb_p,
            "home_qb_delta": 0.0, "away_qb_delta": 0.0, "home_rank": home_rank, "away_rank": away_rank,
            "fbs_teams": 136, "home_newcomer": home_newcomer, "away_newcomer": away_newcomer, "model": "B0",
            "model_version": "abc", "qb_status_at": at, "generated_at": at}


@dataclass
class _Listing:
    exchange_id: int
    security_id: int


@dataclass
class _Spec:
    tick_size: int = P // 100
    lot_size: int = S // 100
    min_size: int = 0


class _Registry:
    def __init__(self, missing=()):
        self.missing = set(missing)

    def get_listing(self, *, listing_id, **_):
        return [] if listing_id in self.missing else [_Listing(*LISTINGS[listing_id])]

    def get_listing_spec(self, *, listing_id, **_):
        return [_Spec()]


@dataclass
class _Pos:
    net_quantity: int
    avg_entry_price: int
    total_fees: float


class _Positions:
    def __init__(self, no_fees=()):
        self.filled: dict = {}
        self.no_fees = set(no_fees)

    def fill(self, listing, shares, price, fee=0.0):
        key = LISTINGS[listing]
        q, avg, fees = self.filled.get(key, (0, 0, 0.0))
        nq = q + int(shares * S)
        self.filled[key] = (nq, int((q * avg + int(shares * S) * int(price * P)) / nq), fees + fee * P)

    def get_position(self, exchange_id, security_id):
        f = self.filled.get((exchange_id, security_id))
        return _Pos(*f) if f else None

    def get_effective_quantity(self, exchange_id, security_id):
        return self.filled.get((exchange_id, security_id), (0, 0, 0.0))[0]

    def fee_schedule(self, exchange_id, security_id):
        return None if (exchange_id, security_id) in self.no_fees else FakeFeeSchedule(taker=0.07, maker=0.0)


class _Book:
    def __init__(self, key, bids, asks, ts):
        self.exchange_id, self.security_id = key
        self.event_timestamp = ts
        self._bids, self._asks = bids, asks

    def bid_price(self, i):
        return int(self._bids[i][0] * P) if i < len(self._bids) else 0

    def bid_size(self, i):
        return int(self._bids[i][1] * S) if i < len(self._bids) else 0

    def ask_price(self, i):
        return int(self._asks[i][0] * P) if i < len(self._asks) else 0

    def ask_size(self, i):
        return int(self._asks[i][1] * S) if i < len(self._asks) else 0


@dataclass
class _Report:
    exchange_id: int
    security_id: int
    order_status: OrderStatus


class _Control:
    def __init__(self):
        self.exits = []

    def requestExit(self, reason, flatten, timeout_millis):
        self.exits.append((reason, flatten))
        return len(self.exits) == 1

    def isExiting(self):
        return bool(self.exits)


class _MetricStream:
    def __init__(self):
        self.columns, self.rows, self._row, self._frozen = [], [], None, False

    def add_long_column(self, name):
        assert not self._frozen
        self.columns.append(name)
        return len(self.columns) - 1

    add_double_column = add_long_column

    def freeze(self):
        self._frozen = True

    def begin(self):
        assert self._frozen and self._row is None
        self._row = {}

    def set_long(self, column, value):
        self._row[self.columns[column]] = value

    set_double = set_long

    def commit(self):
        self.rows.append(self._row)
        self._row = None


class _Metrics:
    def __init__(self):
        self.streams = {}

    def create_stream(self, name):
        self.streams[name] = _MetricStream()
        return self.streams[name]


@pytest.fixture
def make(tmp_path):
    def _make(rows=None, games=(1, 2), missing=(), no_fees=(), **kw):
        path = tmp_path / "predictions.parquet"
        pd.DataFrame(rows if rows is not None else [_row(T0, 1), _row(T0, 2)]).to_parquet(path)
        args = [{"game_id": g, "listing_home": GAMES[g][0], "listing_away": GAMES[g][1],
                 "kickoff": GAMES[g][2].isoformat()} for g in games]
        with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry(missing)):
            kw.setdefault("bankroll_usd", 10_000.0)
            strat = mod.CFBPregame(games=args, reload_minutes=0, predictions_source=str(path), **kw)
        strat._position_view = _Positions(no_fees=[LISTINGS[x] for x in no_fees])
        strat._control = _Control()
        strat._metric_recorder = _Metrics()
        strat.register_metrics()
        return strat
    return _make


def _tick(strat, listing, book, minutes_before, game=1):
    bids, asks = book
    ts = GAMES[game][2].value - int(minutes_before * MINUTE)
    return strat.on_market_data(_Book(LISTINGS[listing], bids, asks, ts))


def _both(strat, minutes_before, game=1, home=HOME_BOOK, away=AWAY_BOOK):
    """The away leg's book, then the home leg's: the home update is the one that can trade."""
    h, a = GAMES[game][:2]
    _tick(strat, a, away, minutes_before, game)
    return _tick(strat, h, home, minutes_before, game)


def _cancel(strat, listing):
    strat.on_execution_report(_Report(*LISTINGS[listing], OrderStatus.CANCELED))


def test_anchor_buys_the_side_the_model_prefers_when_the_edge_clears_fees(make):
    out = _both(make(), 120)
    assert len(out) == 1
    o = out[0]
    assert (o.exchange_id, o.security_id) == LISTINGS[101], "the model likes the home team"
    assert o.take_side == Side.BID and o.take_order_type == OrderType.LIMIT
    assert 0.50 * P <= o.take_limit_price <= 0.52 * P


def test_anchor_stays_out_when_the_model_agrees_with_the_market(make):
    assert _both(make(rows=[_row(T0, 1, p_home=0.51), _row(T0, 2)]), 120) == []


def test_anchor_needs_a_big_disagreement_that_raw_would_take(make):
    rows = [_row(T0, 1, p_home=0.56), _row(T0, 2)]
    assert _both(make(rows=rows), 120) == [], "w 0.25 moves the fair price only a quarter of the way"
    assert len(_both(make(rows=rows, anchor_weight=1.0, min_edge=0.03), 120)) == 1, "the raw price clears 3c"


def test_the_buy_goes_to_the_away_listing_when_the_model_prefers_the_away_team(make):
    out = _both(make(rows=[_row(T0, 1, p_home=0.30), _row(T0, 2)]), 120, home=AWAY_BOOK, away=HOME_BOOK)
    # Away leg .48/.50 now: the away update came first, so the trade comes on the home update, for the away listing.
    assert len(out) == 1 and (out[0].exchange_id, out[0].security_id) == LISTINGS[102]


def test_no_entry_past_a_games_cutoff_and_the_session_exits_after_the_last(make):
    s = make(rows=[_row(T0, 1), _row(KICKOFF - pd.Timedelta(hours=2), 2)])
    assert _both(s, 29, game=1) == [], "game 1's entry closed 30 minutes before kickoff"
    assert not s.is_exiting
    assert len(_both(s, 89, game=2)) == 1, "game 2 kicks off an hour later and is still open"
    _cancel(s, 201)
    assert _both(s, 29, game=2) == []
    assert s._control.exits == [("slate reached its last entry cutoff", False)], "positions are kept"


def test_the_session_waits_for_an_order_in_flight_before_ending(make):
    s = make(games=(1,))
    assert len(_both(s, 31)) == 1
    _both(s, 29)
    assert not s.is_exiting, "an IOC still working"
    _cancel(s, 101)
    _both(s, 28)
    assert s._control.exits == [("slate reached its last entry cutoff", False)]


def test_no_entry_on_a_stale_prediction_but_the_exit_still_fires(make):
    s = make(rows=[_row(EARLY - pd.Timedelta(hours=2), 1)], games=(1,))
    assert _both(s, 200) == [], "the prediction is 160 minutes old"
    _both(s, 29)
    assert s._control.exits == [("slate reached its last entry cutoff", False)]


def test_the_exit_fires_on_the_launch_kickoffs_when_no_prediction_ever_loads(make):
    s = make(rows=[_row(T0, 2)], games=(1,))
    assert _both(s, 120) == []
    _both(s, 29)
    assert s._control.exits == [("slate reached its last entry cutoff", False)]


@pytest.mark.parametrize("override", [{"away_qb_p": 0.5}, {"week": 8, "home_rank": 120}, {"home_newcomer": True}])
def test_skip_rules_block_entry(make, override):
    assert _both(make(rows=[_row(T0, 1, **override), _row(T0, 2)]), 120) == []


def test_a_wide_spread_blocks_entry(make):
    assert _both(make(), 120, home=([(0.42, 1000)], [(0.50, 1000)])) == []


def test_a_one_sided_book_blocks_entry(make):
    assert _both(make(), 120, away=([], [(0.52, 1000)])) == []


def _cost(strat) -> float:
    return strat.metrics.streams["cfb_pregame_orders"].rows[-1]["cost"]


def test_the_per_game_cap_counts_what_is_already_held(make):
    s = make(max_game_stake_usd=100.0)
    s.positions.fill(101, shares=190, price=0.50)
    assert len(_both(s, 120)) == 1
    assert _cost(s) <= 5.0 + 1e-9, "95 of the game's 100 already committed"


def test_the_total_cap_counts_every_game(make):
    s = make(max_total_stake_usd=120.0)
    s.positions.fill(101, shares=200, price=0.50)
    assert len(_both(s, 150, game=2)) == 1
    assert _cost(s) <= 20.0 + 1e-9


def test_a_restarted_session_at_the_cap_does_not_buy_again(make):
    s = make(max_game_stake_usd=100.0)
    s.positions.fill(101, shares=200, price=0.50)
    assert _both(s, 120) == []


def test_holding_one_side_never_buys_the_other(make):
    s = make(rows=[_row(T0, 1, p_home=0.30), _row(T0, 2)])
    s.positions.fill(101, shares=10, price=0.50)
    assert _both(s, 120, home=AWAY_BOOK, away=HOME_BOOK) == []


def test_a_game_without_a_fee_schedule_is_left_out_and_the_rest_trade(make, caplog):
    s = make(no_fees=(201,))
    with caplog.at_level(logging.WARNING):
        assert _both(s, 150, game=2) == []
    assert "2" in caplog.text and "fee schedule" in caplog.text
    assert len(_both(s, 120, game=1)) == 1


def test_a_game_missing_from_the_registry_is_dropped_at_init(make):
    s = make(missing=(202,))
    assert _both(s, 150, game=2) == [] and len(_both(s, 120, game=1)) == 1


def test_no_tradeable_game_is_an_error(make):
    with pytest.raises(ValueError):
        make(missing=(102, 202))


def test_a_mark_is_recorded_once_per_prediction(make):
    rows = [_row(EARLY, 1, p_home=0.51), _row(EARLY + pd.Timedelta(hours=1), 1, p_home=0.52), _row(EARLY, 2)]
    s = make(rows=rows)
    _both(s, 340)
    _both(s, 330)
    _both(s, 290)
    marks = s.metrics.streams["cfb_pregame_marks"].rows
    assert [m["p_model"] for m in marks] == pytest.approx([0.51, 0.52])
    assert marks[0]["market_mid"] == pytest.approx(0.49) and marks[0]["skipped"] == 0


def test_an_order_records_what_it_was_priced_on(make):
    s = make()
    _both(s, 120)
    row = s.metrics.streams["cfb_pregame_orders"].rows[-1]
    assert row["game_id"] == 1 and row["side"] == 0 and row["p_model"] == pytest.approx(0.70)
    assert row["market_mid"] == pytest.approx(0.49) and row["fair"] > row["market_mid"]
    assert row["best_ask"] == pytest.approx(0.50) and row["shares"] > 0


@pytest.mark.parametrize("kw", [{"bankroll_usd": 0.0}, {"anchor_weight": 1.5}, {"min_price": 0.9, "max_price": 0.9}])
def test_bad_settings_are_refused(make, kw):
    with pytest.raises(ValueError):
        make(**kw)


def test_no_adding_after_the_price_falls_adverse_move_below_the_average_entry(make):
    """A big pre-kickoff move is usually news the model cannot see (a QB ruled out after the last refresh)."""
    s = make(rows=[_row(T0, 1, p_home=0.80), _row(T0, 2)])
    s.positions.fill(101, shares=10, price=0.60)
    assert _both(s, 120, home=([(0.50, 1000)], [(0.52, 1000)]), away=([(0.48, 1000)], [(0.50, 1000)])) == [], \
        "0.52 is 8c under the 0.60 entry"
    assert len(_both(s, 110, home=([(0.53, 1000)], [(0.55, 1000)]), away=([(0.45, 1000)], [(0.47, 1000)]))) == 1


# ---- take profit ----------------------------------------------------------------------

TP = dict(anchor_weight=1.0, min_edge=0.03, take_profit=True)
RICH_HOME = ([(0.62, 1000)], [(0.64, 1000)])      # the market now pays more than C2's 0.60 for home
RICH_AWAY = ([(0.36, 1000)], [(0.38, 1000)])
FAIR_HOME = ([(0.61, 1000)], [(0.63, 1000)])      # 0.61 less the fee is still under 0.60
FAIR_AWAY = ([(0.37, 1000)], [(0.39, 1000)])


def _tp(make, **kw):
    s = make(rows=[_row(T0, 1, p_home=0.60), _row(T0, 2)], games=(1,), **{**TP, **kw})
    s.positions.fill(101, shares=100, price=0.50)
    return s


def test_take_profit_sells_once_the_bid_after_fees_pays_the_fair_price(make):
    out = _both(_tp(make), 60, home=RICH_HOME, away=RICH_AWAY)
    assert len(out) == 1
    o = out[0]
    assert (o.exchange_id, o.security_id) == LISTINGS[101] and o.take_side == Side.ASK
    assert o.take_size == 100 * S and o.take_limit_price == int(0.62 * P)


def test_take_profit_holds_while_the_bid_is_under_the_fair_price(make):
    assert _both(_tp(make), 60, home=FAIR_HOME, away=FAIR_AWAY) == []


def test_hold_to_settlement_never_sells(make):
    s = make(rows=[_row(T0, 1, p_home=0.60), _row(T0, 2)], games=(1,), anchor_weight=1.0, min_edge=0.03)
    s.positions.fill(101, shares=100, price=0.50)
    assert _both(s, 60, home=RICH_HOME, away=RICH_AWAY) == []


def test_after_kickoff_the_last_pregame_fair_price_still_decides_and_the_session_stays_open(make):
    s = _tp(make)
    _both(s, 60, home=FAIR_HOME, away=FAIR_AWAY)
    assert _both(s, -20, home=([(0.55, 1000)], [(0.57, 1000)]), away=([(0.43, 1000)], [(0.45, 1000)])) == []
    assert not s.is_exiting, "it still holds the home side during the game"
    out = _both(s, -45, home=([(0.70, 1000)], [(0.72, 1000)]), away=([(0.28, 1000)], [(0.30, 1000)]))
    assert len(out) == 1 and out[0].take_side == Side.ASK, "the home side rallied past 0.60 in the game"


def test_the_session_ends_once_past_kickoff_with_nothing_held(make):
    s = _tp(make)
    _both(s, 60, home=RICH_HOME, away=RICH_AWAY)
    s.positions.filled.pop(LISTINGS[101])
    s.on_execution_report(_Report(*LISTINGS[101], OrderStatus.FILLED))
    _both(s, -5, home=RICH_HOME, away=RICH_AWAY)
    assert s._control.exits == [("slate is past kickoff with nothing held", False)]


def test_the_session_ends_when_the_exit_window_closes_whatever_it_holds(make):
    s = _tp(make, manage_hours=5.0)
    _both(s, 60, home=FAIR_HOME, away=FAIR_AWAY)
    _both(s, -301, home=FAIR_HOME, away=FAIR_AWAY)
    assert s._control.exits == [("exit window closed", False)]


def test_a_game_it_took_profit_on_is_not_bought_again(make):
    s = _tp(make)
    assert len(_both(s, 90, home=RICH_HOME, away=RICH_AWAY)) == 1
    s.positions.filled.pop(LISTINGS[101])
    s.on_execution_report(_Report(*LISTINGS[101], OrderStatus.FILLED))
    assert _both(s, 80, home=([(0.48, 1000)], [(0.50, 1000)]), away=([(0.50, 1000)], [(0.52, 1000)])) == []


def test_an_exit_is_recorded(make):
    s = _tp(make)
    _both(s, 60, home=RICH_HOME, away=RICH_AWAY)
    row = s.metrics.streams["cfb_pregame_exits"].rows[-1]
    assert row["game_id"] == 1 and row["side"] == 0 and row["fair"] == pytest.approx(0.60)
    assert row["best_bid"] == pytest.approx(0.62) and row["shares"] == pytest.approx(100) and row["avg_entry"] == pytest.approx(0.50)
