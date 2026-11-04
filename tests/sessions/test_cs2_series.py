"""
CS2Series: entry and sale maths, the match feed's view of time, windows, lineups, venues, budgets, rebalancing on a
model change, restart safety.

The strategy resolves listings through RegistryClient and loads its model and blend in __init__, so the registry and
the loader are replaced with in-memory stand-ins (a model that prices team A at 0.40, lower with a stand-in, and a
blend that keeps only the model's price, unless a test says otherwise); the match feed reads real parquet extracts;
positions and market data are plain objects with the engine's interface. Nothing here needs a JVM.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gnomepy import OrderStatus, OrderType, Scales, Side
from gnomepy.data import ArtifactRef
from gnomepy_research.models.cs2.e12.blend import MarketBlend
from gnomepy_research.models.cs2.e12.model import E12Model
from gnomepy_research.models.cs2.e12.skill import SkillFit
from gnomepy_research.sessions.cs2_series import strategy as mod
from gnomepy_research.sessions.cs2_series.decision import (
    Level,
    cost_per_share,
    evaluate_side,
    kelly_shares,
    kelly_take,
    max_entry_price,
    proceeds_per_share,
    sell_down,
    taker_fee,
)
from gnomepy_research.sessions.cs2_series.match_feed import MatchFeed
from tests.sessions.conftest import FakeFeeSchedule, FakeStreams, stream_of

P, S = Scales.PRICE, Scales.SIZE
MINUTE = 60_000_000_000
HOUR = 60 * MINUTE
KICKOFF = pd.Timestamp("2026-10-05T18:00:00Z")
K_NS = KICKOFF.value
VENUES = {"polymarket": (101, 102), "kalshi": (201, 202), "polymarket_us": (301, 302)}
LISTINGS = {101: (4, 1101), 102: (4, 1102), 201: (5, 1201), 202: (5, 1202), 301: (6, 1301), 302: (6, 1302)}
LEGS = [{"venue": v, "listing_team_a": a, "listing_team_b": b} for v, (a, b) in VENUES.items()]
LINEUP_A, LINEUP_B = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
# A stand-in on team A makes team B the favourite; one on team B makes team A it.
STANDIN_A, STANDIN_B = [11, 2, 3, 4, 5], [12, 7, 8, 9, 10]
FEE = 0.07


# ---- decision maths ------------------------------------------------------------------

def test_taker_fee_is_the_parametric_formula():
    assert taker_fee(0.5, FEE) == pytest.approx(0.0175)
    assert taker_fee(0.9, FEE) == pytest.approx(0.0063)


def test_max_entry_price_leaves_exactly_the_threshold():
    for fair, edge in ((0.6, 0.05), (0.75, 0.10), (0.35, 0.03)):
        cap = max_entry_price(fair, edge, FEE)
        assert fair - cap - taker_fee(cap, FEE) == pytest.approx(edge, abs=1e-9)


def test_max_entry_price_without_fee_is_fair_minus_edge():
    assert max_entry_price(0.6, 0.05, 0.0) == pytest.approx(0.55)


def _side(fair, asks, fee_rate=FEE):
    return evaluate_side(fair, asks, min_edge=0.05, fee_rate=fee_rate, tick=0.01)


def test_no_entry_when_best_ask_does_not_clear_the_threshold():
    assert _side(0.6, [Level(0.58, 100)]) is None


def test_the_entry_reaches_every_level_that_still_clears_the_threshold():
    # fair 0.70, 5c edge, 7% fee: the cap is the price p with 0.70 - p - 0.07 p (1 - p) = 0.05, about 0.6339.
    asks = [Level(0.40, 50), Level(0.50, 50), Level(0.63, 50), Level(0.64, 50)]
    e = _side(0.70, asks)
    assert e.limit_price == pytest.approx(0.63), "floored to the tick below the cap, however far above the best ask"
    assert [lv.price for lv in e.levels] == [0.40, 0.50, 0.63] and e.shares_available == pytest.approx(150)
    for lv in e.levels:
        assert 0.70 - lv.price - taker_fee(lv.price, FEE) >= 0.05 - 1e-12


def test_the_threshold_price_cuts_the_book():
    e = _side(0.58, [Level(0.50, 50), Level(0.53, 50)])
    assert e.limit_price <= max_entry_price(0.58, 0.05, FEE)
    assert e.shares_available == pytest.approx(50), "the 0.53 level is above the cap"


def test_long_shots_and_favourites_are_bought_on_the_same_rule():
    assert _side(0.20, [Level(0.10, 100)]) is not None
    assert _side(0.99, [Level(0.93, 100)]) is not None


def test_from_nothing_kelly_spends_the_fraction_for_a_contract_paying_one():
    u = cost_per_share(0.40, FEE)
    spent = kelly_shares(0.60, u, 1000.0, 0.0, 0.0) * u
    assert spent == pytest.approx(1000.0 * (0.60 - u) / (1 - u))


def test_kelly_tops_up_to_the_same_holding_whatever_was_bought_on_the_way():
    u = cost_per_share(0.40, FEE)
    full = kelly_shares(0.60, u, 1000.0, 0.0, 0.0)
    part = full / 3
    assert kelly_shares(0.60, u, 1000.0, part, part * u) == pytest.approx(full - part)
    assert kelly_shares(0.60, u, 1000.0, full, full * u) == pytest.approx(0.0, abs=1e-9)
    assert kelly_shares(0.60, u, 1000.0, 2 * full, 2 * full * u) < 0, "past the Kelly size: nothing more to buy"


def test_no_kelly_size_without_an_edge():
    assert kelly_shares(0.40, cost_per_share(0.40, FEE), 1000.0, 0.0, 0.0) < 0


def test_a_take_stops_at_the_kelly_size_in_a_deep_book():
    e = _side(0.80, [Level(0.40, 10_000), Level(0.45, 10_000)])
    take = kelly_take(e, 100.0, 0.0, 0.0, max_cost=1e9, fee_rate=FEE, lot=1.0)
    u = cost_per_share(0.40, FEE)
    assert take.limit_price == pytest.approx(0.40), "the best level alone holds the Kelly size"
    assert take.shares == pytest.approx(math.floor(kelly_shares(0.80, u, 100.0, 0.0, 0.0)))


def test_a_take_walks_levels_each_at_its_own_price_and_fee_until_the_kelly_size():
    e = _side(0.80, [Level(0.40, 20), Level(0.45, 30), Level(0.50, 10_000)])
    take = kelly_take(e, 100.0, 0.0, 0.0, max_cost=1e9, fee_rate=FEE, lot=1.0)
    by_hand = 20 * cost_per_share(0.40, FEE) + 30 * cost_per_share(0.45, FEE)
    u = cost_per_share(0.50, FEE)
    rest = math.floor(kelly_shares(0.80, u, 100.0, 50.0, by_hand))
    assert take.shares == pytest.approx(50 + rest)
    assert take.cost == pytest.approx(by_hand + rest * u) and take.limit_price == pytest.approx(0.50)


def test_a_take_stops_at_the_cap_before_the_kelly_size():
    e = _side(0.80, [Level(0.40, 10_000)])
    take = kelly_take(e, 10_000.0, 0.0, 0.0, max_cost=50.0, fee_rate=FEE, lot=1.0)
    assert 50.0 - cost_per_share(0.40, FEE) < take.cost <= 50.0


def test_a_take_counts_what_is_already_held_against_the_cap_and_the_kelly_size():
    e = _side(0.80, [Level(0.40, 10_000)])
    u = cost_per_share(0.40, FEE)
    assert kelly_take(e, 10_000.0, 100.0, 45.0, max_cost=50.0, fee_rate=FEE, lot=1.0).cost <= 5.0
    full = kelly_shares(0.80, u, 100.0, 0.0, 0.0)
    assert kelly_take(e, 100.0, full, full * u, max_cost=1e9, fee_rate=FEE, lot=1.0).shares == 0.0


def test_a_take_is_whole_lots_and_stops_with_the_book():
    e = _side(0.80, [Level(0.40, 7.5)])
    take = kelly_take(e, 10_000.0, 0.0, 0.0, max_cost=50.0, fee_rate=FEE, lot=2.0)
    assert take.shares == pytest.approx(6.0) and take.cost == pytest.approx(6 * cost_per_share(0.40, FEE))


def _sale(fair, bids, held, lot=1.0):
    return sell_down(fair, bids, held, min_edge=0.02, fee_rate=FEE, lot=lot)


def test_no_sale_unless_the_best_bid_beats_holding_by_the_edge():
    assert _sale(0.40, [Level(0.41, 100)], 50) is None, "0.41 less its fee is under 0.40 + 2c"
    assert _sale(0.40, [Level(0.45, 100)], 50) is not None
    assert _sale(0.40, [Level(0.45, 100)], 0) is None, "nothing held"
    assert _sale(0.40, [], 50) is None


def test_a_sale_takes_every_bid_that_beats_holding_and_no_more_than_is_held():
    bids = [Level(0.47, 20), Level(0.45, 20), Level(0.43, 1000)]
    s = _sale(0.40, bids, 100)
    assert s.shares == pytest.approx(40) and s.limit_price == pytest.approx(0.45), "0.43 doesn't clear 2c after fee"
    assert s.proceeds == pytest.approx(20 * proceeds_per_share(0.47, FEE) + 20 * proceeds_per_share(0.45, FEE))
    assert s.edge_at_best == pytest.approx(proceeds_per_share(0.47, FEE) - 0.40)
    held = _sale(0.40, bids, 25)
    assert held.shares == pytest.approx(25) and held.limit_price == pytest.approx(0.45)


def test_a_sale_is_whole_lots():
    assert _sale(0.40, [Level(0.45, 100)], 7.5, lot=2.0).shares == pytest.approx(6.0)
    assert _sale(0.40, [Level(0.45, 100)], 1.5, lot=2.0) is None


# ---- the match's rows ----------------------------------------------------------------

class _Model:
    """Prices team A at 0.40 with its usual five, 0.25 with a stand-in, 0.60 when team B has one."""

    def __init__(self, cutoff=pd.Timestamp("2026-10-05") - pd.Timedelta(days=3)):
        self.cutoff = cutoff
        self.priced = []

    def price(self, team_a, five_a, team_b, five_b):
        self.priced.append((team_a, tuple(five_a), team_b, tuple(five_b)))
        if 11 in five_a:
            return 0.25
        return 0.60 if 12 in five_b else 0.40


MODEL_ONLY = MarketBlend(market=0.0, model=1.0, intercept=0.0)


def _models(model=None, blend=MODEL_ONLY):
    return mod.Models(model or _Model(), blend, {"model": "m", "blend": "b"})


def _feature(at, lineup_a=LINEUP_A, lineup_b=LINEUP_B, match_id=7, team_a_id=1):
    return {"match_id": match_id, "map_name": "", "generated_at": at, "match_date": KICKOFF.normalize(),
            "team_a_id": team_a_id, "team_b_id": 2, "team_a_lineup_ids": lineup_a, "team_b_lineup_ids": lineup_b}


def _state(at, kickoff=KICKOFF, status="upcoming"):
    markets = [{"venue": v, "market": "series", "listing_team_a": a, "listing_team_b": b} for v, (a, b) in VENUES.items()]
    return {"match_id": 7, "generated_at": at, "kickoff": kickoff, "status": status, "markets": json.dumps(markets)}


EARLY = KICKOFF - pd.Timedelta(hours=24)
RUN_EVERY = pd.Timedelta(minutes=15)


def _states(*changes, until=KICKOFF + pd.Timedelta(hours=6)):
    """A matches row every pipeline run from EARLY, as cs2_upcoming writes them; changes are
    (from_time, {"kickoff": ..., "status": ...}) applied from that run on."""
    rows, current = [], {"kickoff": KICKOFF, "status": "upcoming"}
    pending = sorted(changes, key=lambda c: c[0])
    at = EARLY
    while at <= until:
        while pending and pending[0][0] <= at:
            current = {**current, **pending.pop(0)[1]}
        rows.append(_state(at, **current))
        at += RUN_EVERY
    return rows


def _write(tmp_path, features, states):
    f, m = tmp_path / "features.parquet", tmp_path / "matches.parquet"
    pd.DataFrame(features).to_parquet(f)
    pd.DataFrame(states).to_parquet(m)
    return str(f), str(m)


def _pricer(calls):
    model = _Model()

    def pricer(rows):
        calls.append(len(rows))
        return [model.price(r.team_a_id, list(r.team_a_lineup_ids), r.team_b_id, list(r.team_b_lineup_ids))
                for r in rows.itertuples()]
    return pricer


def _before(minutes):
    return KICKOFF - pd.Timedelta(minutes=minutes)


def test_the_feed_is_point_in_time(tmp_path):
    t0 = KICKOFF - pd.Timedelta(hours=10)
    f, m = _write(tmp_path, [_feature(t0), _feature(t0 + pd.Timedelta(hours=1), lineup_a=STANDIN_A)],
                  [_state(t0), _state(t0 + pd.Timedelta(hours=2), kickoff=KICKOFF + pd.Timedelta(hours=2))])
    feed = MatchFeed(7, _pricer([]), f, m, reload_seconds=0)
    assert feed.quote(t0.value - 1) is None and feed.state(t0.value - 1) is None, "nothing before it was written"
    assert feed.quote(t0.value).q == pytest.approx(0.40)
    assert feed.quote((t0 + pd.Timedelta(hours=5)).value).q == pytest.approx(0.25)
    assert feed.state((t0 + pd.Timedelta(hours=1)).value).kickoff_ns == K_NS
    assert feed.state((t0 + pd.Timedelta(hours=3)).value).kickoff_ns == (KICKOFF + pd.Timedelta(hours=2)).value


def test_the_feed_reads_only_its_match_and_carries_lineups(tmp_path):
    f, m = _write(tmp_path, [_feature(EARLY, lineup_a=[]), _feature(EARLY, lineup_a=STANDIN_A, match_id=8)],
                  [_state(EARLY)])
    q = MatchFeed(7, _pricer([]), f, m, reload_seconds=0).quote(K_NS)
    assert q.q == pytest.approx(0.40) and not q.lineup_known
    assert q.lineups == ((), tuple(LINEUP_B))


def test_reload_prices_only_new_rows_and_survives_a_bad_read(tmp_path):
    calls = []
    f, m = _write(tmp_path, [_feature(EARLY)], [_state(EARLY)])
    feed = MatchFeed(7, _pricer(calls), f, m, reload_seconds=0.05)
    later = EARLY + pd.Timedelta(hours=1)
    _write(tmp_path, [_feature(EARLY), _feature(later, lineup_a=STANDIN_A)], [_state(EARLY)])
    deadline = time.monotonic() + 5
    while feed.quote(K_NS).at_ns != later.value and time.monotonic() < deadline:
        time.sleep(0.02)
    assert feed.quote(K_NS).q == pytest.approx(0.25)
    assert calls == [1, 1], "the first row is not priced again"
    (tmp_path / "features.parquet").write_bytes(b"not a parquet file")
    time.sleep(0.2)
    assert feed.quote(K_NS).at_ns == later.value, "a failed reload keeps the rows already held"
    # The reload thread outlives the test (it is a daemon, as in production); give it a readable file.
    _write(tmp_path, [_feature(EARLY)], [_state(EARLY)])


def test_the_feed_rereads_faster_near_kickoff(tmp_path):
    f, m = _write(tmp_path, [_feature(EARLY)], _states())
    feed = MatchFeed(7, _pricer([]), f, m, reload_seconds=300, near_kickoff_reload_seconds=60, near_kickoff_seconds=3600)
    assert feed.reload_interval(K_NS - 2 * HOUR) == 300
    assert feed.reload_interval(K_NS - HOUR // 2) == 60
    assert feed.reload_interval(K_NS + HOUR) == 60, "and after kickoff, until HLTV shows it live"


# ---- strategy ------------------------------------------------------------------------

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
    def __init__(self, spec):
        self.spec = spec

    def get_listing(self, *, listing_id, **_):
        return [_Listing(*LISTINGS[listing_id])]

    def get_listing_spec(self, *, listing_id, **_):
        return [self.spec]


@dataclass
class _Pos:
    net_quantity: int
    avg_entry_price: int
    total_fees: float


class _Positions:
    """Engine position view: filled quantity, average entry, fees in price units as the OMS keeps them, plus pending
    orders. A sale (negative shares) leaves the average entry where it was."""

    def __init__(self, rates=None):
        self.filled: dict = {}
        self.pending: dict = {}
        self.rates = rates or {}

    def fill(self, listing, shares, price, fee=0.0):
        key = LISTINGS[listing]
        q, avg, fees = self.filled.get(key, (0, 0, 0.0))
        add = round(shares * S)
        nq = q + add
        if add > 0:
            avg = int((q * avg + add * int(price * P)) / nq)
        self.filled[key] = (nq, avg if nq else 0, fees + fee * P)

    def get_position(self, exchange_id, security_id):
        f = self.filled.get((exchange_id, security_id))
        return _Pos(*f) if f else None

    def get_effective_quantity(self, exchange_id, security_id):
        key = (exchange_id, security_id)
        return self.filled.get(key, (0, 0, 0.0))[0] + self.pending.get(key, 0)

    def fee_schedule(self, exchange_id, security_id):
        return FakeFeeSchedule(taker=self.rates.get(exchange_id, FEE), maker=0.0)


class _Book:
    def __init__(self, listing, asks, ts, bids):
        self.stream_id = stream_of(listing)
        self.event_timestamp = ts
        self._asks = asks
        self._bids = bids

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
    """Stands in for the session's StrategyControl, recording exit requests."""

    def __init__(self):
        self.exits = []

    def requestExit(self, reason, flatten, timeout_millis):
        self.exits.append((reason, flatten))
        return len(self.exits) == 1

    def isExiting(self):
        return bool(self.exits)


class _MetricStream:
    """Keeps committed rows, holding the strategy to the declare, freeze, begin, set, commit order sessions enforce."""

    def __init__(self):
        self.columns = []
        self.rows = []
        self._frozen = False
        self._row = None

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
    def _make(features=None, states=None, spec=None, models=None, rates=None, **kw):
        f, m = _write(tmp_path, features or [_feature(EARLY)], states or _states())
        with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry(spec or _Spec())), \
                patch.object(mod, "load_models", lambda *paths: models or _models()):
            kw.setdefault("bankroll_usd", 10_000.0)
            kw.setdefault("legs", LEGS)
            strat = mod.CS2Series(match_id=7, reload_minutes=0, features_source=f, matches_source=m, **kw)
        strat._position_view = _Positions(rates)
        strat._streams = FakeStreams(LISTINGS)
        strat._control = _Control()
        return strat
    return _make


def _tick(strat, listing, asks, minutes_before, bids=None):
    """A book update; unless a test gives bids, the best bid sits 2c under the best ask, so the book has a mid."""
    if bids is None:
        bids = [(round(asks[0][0] - 0.02, 2), 1000)] if asks else []
    return strat.on_market_data(_Book(listing, asks, K_NS - int(minutes_before * MINUTE), bids))


def _done(strat, listing, status=OrderStatus.FILLED):
    strat.on_execution_report(_Report(*LISTINGS[listing], status))


def _key(order):
    return order.exchange_id, order.security_id


# Team A at 0.40 makes team B worth 0.60: its 0.50 ask clears 2c after the fee by about 8c.
B_ASK = [(0.50, 1000)]


def test_buys_the_underpriced_team_with_a_limit_at_the_deepest_level_it_costed(make):
    out = _tick(make(), 102, B_ASK, 30)
    assert len(out) == 1
    o = out[0]
    assert _key(o) == LISTINGS[102], "team B is underpriced, so buy team B's listing"
    assert o.take_side == Side.BID and o.take_order_type == OrderType.LIMIT
    assert o.take_limit_price == round(0.50 * P)


def test_buys_on_the_venue_with_the_biggest_edge_after_its_own_fee(make):
    s = make()
    assert _tick(s, 202, [(0.48, 1000)], 90) == [], "outside the window: the book is only kept"
    assert _key(_tick(s, 102, B_ASK, 30)[0]) == LISTINGS[202], "Kalshi's 0.48 beats Polymarket's 0.50"
    expensive = make(rates={5: 0.5})
    _tick(expensive, 202, [(0.48, 1000)], 90)
    assert _key(_tick(expensive, 102, B_ASK, 30)[0]) == LISTINGS[102], "but not at a 0.5 fee rate"


def test_the_stake_cap_counts_every_venue(make):
    s = make(max_stake_usd=250.0)
    s.positions.fill(102, shares=380, price=0.50, fee=380 * taker_fee(0.50, FEE))
    o = _tick(s, 202, B_ASK, 30)[0]
    assert _key(o) == LISTINGS[202]
    committed = 380 * cost_per_share(0.50, FEE)
    assert o.take_size / S * cost_per_share(0.50, FEE) <= 250.0 - committed + 1e-6
    shares = o.take_size / S
    s.positions.fill(202, shares=shares, price=0.50, fee=shares * taker_fee(0.50, FEE))
    _done(s, 202)
    assert _tick(s, 302, B_ASK, 29) == [], "the match's cap is spent, whichever venue is cheap"


def test_never_buys_the_other_team_while_holding_one(make):
    s = make()
    s.positions.fill(102, shares=10, price=0.50)
    assert _tick(s, 101, [(0.30, 1000)], 30) == [], "holding team B; team A now looks cheap - no hedge"


def test_nothing_outside_the_entry_window(make):
    s = make()
    assert _tick(s, 102, B_ASK, 61) == [], "the window opens an hour out"
    assert _tick(s, 102, B_ASK, 59) != []
    _done(s, 102, OrderStatus.CANCELED)
    assert _tick(s, 102, B_ASK, 0) == [], "and closes at kickoff"
    assert s._control.exits == [("match 7 reached its entry cutoff", False)]


def test_the_window_follows_the_latest_kickoff(make):
    s = make(states=_states((EARLY + pd.Timedelta(hours=1), {"kickoff": KICKOFF + pd.Timedelta(hours=1)})))
    assert _tick(s, 102, B_ASK, 30) == [], "90 minutes before the new kickoff"
    assert _tick(s, 102, B_ASK, -30) != [], "30 minutes before it"
    assert not s.is_exiting, "a delayed kickoff moves the cutoff"
    _done(s, 102, OrderStatus.CANCELED)
    assert _tick(s, 102, B_ASK, -61) == []
    assert s._control.exits == [("match 7 reached its entry cutoff", False)]


@pytest.mark.parametrize("status", ["live", "missing"])
def test_the_session_ends_once_hltv_says_live_or_drops_the_match(make, status):
    s = make(states=_states((_before(20), {"status": status})))
    assert _tick(s, 102, B_ASK, 30) != []
    _done(s, 102, OrderStatus.CANCELED)
    assert _tick(s, 102, B_ASK, 10) == []
    assert s._control.exits == [(f"match 7 is {status}", False)]


@pytest.mark.parametrize("holding", [False, True])
def test_the_session_ends_at_kickoff_keeping_whatever_it_holds(make, holding):
    s = make()
    if holding:
        s.positions.fill(102, shares=10, price=0.50)
    _tick(s, 102, [(0.95, 1000)], 1)
    assert not s.is_exiting
    assert _tick(s, 102, B_ASK, -0.1) == []
    assert s._control.exits == [("match 7 reached its entry cutoff", False)], "positions are kept, not flattened"


def test_the_session_waits_for_an_order_in_flight_before_ending(make):
    s = make()
    assert _tick(s, 102, B_ASK, 0.1) != []
    _tick(s, 102, B_ASK, -0.1)
    assert not s.is_exiting, "an order still working"
    _done(s, 102)
    _tick(s, 102, B_ASK, -0.2)
    assert s._control.exits == [("match 7 reached its entry cutoff", False)]


def test_the_session_ends_at_kickoff_even_on_stale_state(make):
    s = make(states=_states(until=KICKOFF - pd.Timedelta(hours=3)))
    _tick(s, 102, B_ASK, 30)
    assert not s.is_exiting, "stale state stops trading"
    _tick(s, 102, B_ASK, 0)
    assert s._control.exits == [("match 7 reached its entry cutoff", False)]


@pytest.mark.parametrize("kw", [
    {"entry_cutoff_hours": 1.0},
    {"bankroll_usd": 0.0},
    {"legs": []},
    {"legs": LEGS + LEGS[:1]},
])
def test_a_session_needs_a_window_a_bankroll_and_each_venue_once(make, kw):
    with pytest.raises(ValueError):
        make(**kw)


def test_no_trading_on_stale_match_state(make):
    s = make(states=_states(until=_before(75)))
    assert _tick(s, 102, B_ASK, 50) != [], "the newest state is 25 minutes old"
    _done(s, 102, OrderStatus.CANCELED)
    assert _tick(s, 102, B_ASK, 40) == [], "35 minutes old: the pipeline has stalled"


def test_no_trade_before_the_first_features_row(make):
    s = make(features=[_feature(_before(20))])
    assert _tick(s, 102, B_ASK, 30) == [], "a replay must not trade on a future row"
    assert _tick(s, 102, B_ASK, 19) != []


def test_no_trade_until_both_lineups_are_announced(make):
    s = make(features=[_feature(EARLY, lineup_b=[]), _feature(_before(20))])
    assert _tick(s, 102, B_ASK, 30) == []
    assert _tick(s, 102, B_ASK, 19) != []


def test_no_trade_without_a_team_id_to_price(make):
    assert _tick(make(features=[_feature(EARLY, team_a_id=None)]), 102, B_ASK, 30) == []


def test_the_model_prices_the_announced_fives_of_the_hltv_teams(make):
    model = _Model()
    make(models=_models(model), features=[_feature(EARLY, lineup_b=STANDIN_B)])
    assert model.priced == [(1, tuple(LINEUP_A), 2, tuple(STANDIN_B))]


def test_no_trade_without_a_two_sided_book_for_the_mid(make):
    s = make()
    assert _tick(s, 102, B_ASK, 30, bids=[]) == [], "no venue has a mid to blend with"
    assert _tick(s, 102, B_ASK, 29) != []


def test_the_mid_is_polymarket_internationals_else_the_tightest_elsewhere(make):
    s = make()
    _tick(s, 201, [(0.61, 10)], 90, bids=[(0.60, 10)])
    _tick(s, 302, [(0.42, 10)], 90, bids=[(0.38, 10)])
    assert s._mid_team_a() == pytest.approx(0.605), "Kalshi's team A book is the tightest"
    _tick(s, 101, [(0.53, 10)], 90, bids=[(0.47, 10)])
    assert s._mid_team_a() == pytest.approx(0.50), "Polymarket's, though wider: the blend was fitted on it"
    _tick(s, 101, [(0.53, 10)], 90, bids=[])
    _tick(s, 102, [(0.51, 10)], 90, bids=[(0.49, 10)])
    assert s._mid_team_a() == pytest.approx(0.50), "Polymarket's team B book, where team A's is one-sided"
    _tick(s, 102, [(0.51, 10)], 90, bids=[])
    _tick(s, 201, [(0.61, 10)], 90, bids=[])
    assert s._mid_team_a() == pytest.approx(0.60), "Polymarket US team B's, the only two-sided book left"


def test_the_market_mid_pulls_the_fair_price(make):
    blend = MarketBlend(0.6123418580625587, 0.5130722002876456, 0.008852116804822632)
    s = make(models=_models(blend=blend))
    _tick(s, 101, [(0.51, 10)], 90, bids=[(0.49, 10)])
    assert _tick(s, 102, B_ASK, 30, bids=[(0.48, 10)]) != [], "mid 0.50 and model 0.40: team B is worth about 0.55"
    _done(s, 102, OrderStatus.CANCELED)
    _tick(s, 101, [(0.66, 10)], 29, bids=[(0.64, 10)])
    assert _tick(s, 102, B_ASK, 28, bids=[(0.48, 10)]) == [], "the market says team A at 0.65: team B is no bargain"


@pytest.mark.parametrize("cutoff, trades", [
    (pd.Timestamp("2026-09-28"), True),
    (pd.Timestamp("2026-09-26"), False),
    (pd.Timestamp("2026-10-06"), False),
])
def test_no_trading_on_a_stale_or_future_model(make, cutoff, trades):
    s = make(models=_models(_Model(cutoff=cutoff)))
    assert (_tick(s, 102, B_ASK, 30) != []) is trades


def test_a_bankroll_too_small_for_the_cap_buys_the_kelly_size(make):
    o = _tick(make(bankroll_usd=500.0), 102, [(0.50, 10_000)], 30)[0]
    u = cost_per_share(0.50, FEE)
    spend = o.take_size / S * u
    assert spend == pytest.approx(0.5 * 500.0 * (0.60 - u) / (1 - u), abs=u)
    assert spend < 250.0, "the Kelly size, not the cap, set this order"


def test_the_kelly_fraction_scales_the_size(make):
    half = _tick(make(bankroll_usd=500.0, kelly_fraction=0.5), 102, [(0.50, 10_000)], 30)[0]
    full = _tick(make(bankroll_usd=500.0, kelly_fraction=1.0), 102, [(0.50, 10_000)], 30)[0]
    assert full.take_size == pytest.approx(2 * half.take_size, abs=S // 100)


def test_a_held_kelly_size_is_topped_up_only_when_the_price_improves(make):
    s = make(bankroll_usd=500.0)
    o = _tick(s, 102, [(0.50, 10_000)], 30)[0]
    shares = o.take_size / S
    s.positions.fill(102, shares=shares, price=0.50, fee=shares * taker_fee(0.50, FEE))
    _done(s, 102)
    assert _tick(s, 102, [(0.50, 10_000)], 29) == [], "already at the Kelly size for this price"
    assert _tick(s, 202, [(0.45, 10_000)], 28), "a cheaper ask anywhere means a bigger Kelly size"


def test_a_thin_book_is_topped_up_to_the_kelly_size_once_it_deepens(make):
    s = make(bankroll_usd=500.0)
    o = _tick(s, 102, [(0.50, 5)], 30)[0]
    assert o.take_size == 5 * S, "only 5 shares on offer, short of the Kelly size"
    s.positions.fill(102, shares=5, price=0.50, fee=5 * taker_fee(0.50, FEE))
    _done(s, 102)
    top_up = _tick(s, 102, [(0.50, 10_000)], 29)
    u = cost_per_share(0.50, FEE)
    assert (5 + top_up[0].take_size / S) * u == pytest.approx(0.5 * 500.0 * (0.60 - u) / (1 - u), abs=u)


def test_the_stake_cap_is_respected_including_fees(make):
    o = _tick(make(max_stake_usd=50.0), 102, [(0.50, 10_000)], 30)[0]
    assert o.take_size / S * cost_per_share(0.50, FEE) <= 50.0 + 1e-6


def test_a_thin_book_is_walked_to_the_level_the_stake_needs_and_no_further(make):
    asks = [(0.50, 20), (0.51, 20), (0.52, 10_000), (0.53, 10_000)]
    o = _tick(make(max_stake_usd=50.0), 102, asks, 30)[0]
    assert o.take_limit_price == round(0.52 * P), "40 shares at 0.50-0.51 leave most of the stake for 0.52"


def test_edges_under_the_floor_are_not_bought(make):
    assert _tick(make(min_edge=0.02), 102, [(0.57, 1000)], 30) == [], "0.60 - 0.57 - fee is under 2c"
    assert _tick(make(min_edge=0.02), 102, [(0.56, 1000)], 30) != []


def test_no_second_order_on_any_venue_while_one_is_in_flight(make):
    s = make()
    assert _tick(s, 102, [(0.50, 10)], 30)
    assert _tick(s, 202, B_ASK, 29) == []
    _done(s, 102)
    assert _tick(s, 202, B_ASK, 28), "a terminal report frees the match to top up"


def test_an_order_under_the_venues_minimum_size_is_not_sent(make):
    s = make(spec=_Spec(min_size=5 * S))
    assert _tick(s, 102, [(0.50, 4)], 30) == [], "only 4 shares at the price; the venue wants 5"
    assert _tick(s, 102, [(0.50, 10)], 30) != []


def test_unknown_listing_data_is_ignored(make):
    assert make().on_market_data(_Book(99999, [(0.01, 10)], K_NS - HOUR, [])) == [], "a stream of no listing it trades"


def test_restart_reads_committed_money_from_positions(make):
    """A fresh instance must see an earlier session's fills and not buy them again."""
    first = make(max_stake_usd=50.0)
    first.positions.fill(102, shares=97, price=0.50, fee=1.7)
    restarted = make(max_stake_usd=50.0)
    restarted._position_view = first.positions
    assert _tick(restarted, 102, B_ASK, 30) == []


# ---- rebalancing on a model change -----------------------------------------------------

def _holding_b(make, features, shares=100, **kw):
    """A session that bought team B 40 minutes out, at the model's price then."""
    s = make(features=features, **kw)
    assert _tick(s, 102, [(0.50, shares)], 40)
    s.positions.fill(102, shares=shares, price=0.50, fee=shares * taker_fee(0.50, FEE))
    _done(s, 102)
    return s


def test_a_lineup_change_sells_into_bids_that_beat_holding(make):
    """A stand-in on team B makes it worth 0.40: bids at 0.45 fetch more than that after the fee, 0.43 don't."""
    s = _holding_b(make, [_feature(EARLY), _feature(_before(20), lineup_b=STANDIN_B)])
    out = _tick(s, 102, [(0.47, 1000)], 10, bids=[(0.45, 60), (0.43, 1000)])
    assert len(out) == 1
    o = out[0]
    assert _key(o) == LISTINGS[102] and o.take_side == Side.ASK and o.take_order_type == OrderType.LIMIT
    assert o.take_size == 60 * S and o.take_limit_price == round(0.45 * P)


def test_a_sale_never_sells_more_than_the_venue_holds(make):
    s = _holding_b(make, [_feature(EARLY), _feature(_before(20), lineup_b=STANDIN_B)], shares=30)
    o = _tick(s, 102, [(0.47, 1000)], 10, bids=[(0.45, 1000)])[0]
    assert o.take_side == Side.ASK and o.take_size == 30 * S


def test_a_lineup_change_keeps_the_position_when_no_bid_beats_holding(make):
    s = _holding_b(make, [_feature(EARLY), _feature(_before(20), lineup_b=STANDIN_B)])
    assert _tick(s, 102, [(0.43, 1000)], 10, bids=[(0.41, 1000)]) == [], "0.41 after the fee is worth less than 0.40"


def test_a_market_move_alone_never_sells(make):
    s = _holding_b(make, [_feature(EARLY)])
    assert _tick(s, 102, [(0.72, 1000)], 10, bids=[(0.70, 1000)]) == [], "the model still says 0.60; only the market moved"


def test_once_sold_out_the_other_team_is_bought(make):
    s = _holding_b(make, [_feature(EARLY), _feature(_before(20), lineup_b=STANDIN_B)])
    _tick(s, 102, [(0.47, 1000)], 12, bids=[(0.30, 1000)])
    assert _tick(s, 101, [(0.50, 1000)], 12) == [], "team A is now worth 0.60, but team B is still held"
    sale = _tick(s, 102, [(0.47, 1000)], 11, bids=[(0.45, 1000)])[0]
    assert _key(sale) == LISTINGS[102] and sale.take_side == Side.ASK and sale.take_size == 100 * S
    s.positions.fill(102, shares=-100, price=0.45, fee=100 * taker_fee(0.45, FEE))
    _done(s, 102)
    buy = _tick(s, 101, [(0.50, 1000)], 10)[0]
    assert _key(buy) == LISTINGS[101] and buy.take_side == Side.BID


def test_a_model_move_in_the_positions_favour_tops_it_up_and_becomes_the_new_reference(make):
    s = _holding_b(make, [_feature(EARLY), _feature(_before(20), lineup_a=STANDIN_A)])
    o = _tick(s, 102, [(0.70, 1000)], 10, bids=[(0.30, 1000)])[0]
    assert o.take_side == Side.BID, "team B is now worth 0.75"
    s.positions.fill(102, shares=o.take_size / S, price=0.70)
    _done(s, 102)
    assert _tick(s, 102, [(0.97, 1000)], 9, bids=[(0.95, 1000)]) == [], \
        "bought at the new price: a market move alone doesn't sell it"


def test_a_restarted_session_takes_the_model_price_it_finds_as_the_reference(make):
    first = _holding_b(make, [_feature(EARLY), _feature(_before(20), lineup_b=STANDIN_B)])
    restarted = make(features=[_feature(EARLY), _feature(_before(20), lineup_b=STANDIN_B)])
    restarted._position_view = first.positions
    assert _tick(restarted, 102, [(0.47, 1000)], 10, bids=[(0.45, 1000)]) == [], "a restart alone never sells"


# ---- artifacts and metrics -------------------------------------------------------------

def test_artifacts_are_fetched_at_the_version_they_resolved_to(monkeypatch):
    found = []

    class _Store:
        def find(self, query):
            found.append(query)
            return ArtifactRef(query.artifact_type, query.name, "20261103T233431215199Z", "s3://b/e12.npz")

        def resolve(self, ref):
            return "/tmp/e12.npz"
    monkeypatch.setattr(mod, "ArtifactStore", _Store)
    path, resolved = mod._fetch(mod.DEFAULT_MODEL + "@production")
    assert str(path) == "/tmp/e12.npz" and found[0].alias == "production"
    assert resolved == "artifact://e12_model/cs2_series:20261103T233431215199Z"
    monkeypatch.setattr(mod, "resolve_artifact_path", lambda path: "/local/e12.npz")
    assert mod._fetch("/local/e12.npz") == (mod.Path("/local/e12.npz"), "/local/e12.npz")


def test_the_model_and_blend_load_from_their_saved_files(tmp_path):
    fit = SkillFit(np.arange(1, 11), np.zeros(10), np.full(10, 20), np.zeros(12), {"players": 10})
    E12Model(pd.Timestamp("2026-11-02"), fit, fit, {}, {1: 0.1}, {}).save(tmp_path / "e12.npz")
    MODEL_ONLY.save(tmp_path / "blend.json")
    models = mod.load_models(str(tmp_path / "e12.npz"), str(tmp_path / "blend.json"))
    assert models.model.cutoff == pd.Timestamp("2026-11-02") and models.blend == MODEL_ONLY
    assert 0 < models.model.price(1, LINEUP_A, 2, LINEUP_B) < 1


def _recording(make, **kw):
    s = make(**kw)
    s._metric_recorder = _Metrics()
    s.register_metrics()
    return s


def test_each_order_records_the_decision_behind_it(make):
    s = _recording(make)
    out = _tick(s, 102, B_ASK, 30)
    rows = s.metrics.streams["cs2_series_orders"].rows
    assert len(out) == len(rows) == 1
    row = rows[0]
    assert row["timestamp"] == K_NS - 30 * MINUTE
    assert (row["exchange_id"], row["listing_id"], row["team"], row["direction"]) == (4, 102, 1, 1)
    assert row["q"] == pytest.approx(0.40) and row["fair"] == pytest.approx(0.60)
    assert row["mid"] == pytest.approx(0.51), "team A's mid, from team B's book"
    assert row["best_price"] == pytest.approx(0.50) and row["spread"] == pytest.approx(0.02)
    assert round(row["limit_price"] * P) == out[0].take_limit_price
    assert 0 < row["edge_at_best"] < row["fair"] - row["best_price"], "the edge is what's left after the taker fee"
    assert row["shares"] > 0 and row["usd"] == pytest.approx(row["shares"] * cost_per_share(0.50, FEE))
    assert row["kelly_bankroll"] == pytest.approx(0.5 * 10_000.0) and row["held_shares"] == 0.0


def test_a_sale_is_recorded_with_its_proceeds(make):
    s = _holding_b(make, [_feature(EARLY), _feature(_before(20), lineup_b=STANDIN_B)])
    s._metric_recorder = _Metrics()
    s.register_metrics()
    _tick(s, 102, [(0.47, 1000)], 10, bids=[(0.45, 1000)])
    row = s.metrics.streams["cs2_series_orders"].rows[0]
    assert row["direction"] == -1 and row["fair"] == pytest.approx(0.40) and row["held_shares"] == 100.0
    assert row["usd"] == pytest.approx(100 * proceeds_per_share(0.45, FEE))


def test_each_new_model_price_is_recorded_once(make):
    s = _recording(make, features=[_feature(EARLY), _feature(_before(20), lineup_b=STANDIN_B)])
    for minutes in (90, 80, 15, 10):
        _tick(s, 102, [(0.95, 10)], minutes)
    rows = s.metrics.streams["cs2_series_model"].rows
    assert [r["q"] for r in rows] == pytest.approx([0.40, 0.60])
    assert rows[0]["lineups_hash"] != rows[1]["lineups_hash"] and rows[1]["lineup_known"] == 1
    assert rows[1]["model_cutoff"] == pd.Timestamp("2026-10-02").value


# ---- retries: back off after a miss, decide only on what changed ------------------------------------------------

def _decisions(strat):
    """Counts the decisions the strategy runs: each starts by reading the mid."""
    calls = []
    real = strat._mid_team_a
    strat._mid_team_a = lambda: calls.append(1) or real()
    return calls


def test_a_miss_backs_its_listing_off_until_its_top_of_book_moves(make):
    s = make()
    assert _key(_tick(s, 102, B_ASK, 30)[0]) == LISTINGS[102]
    _done(s, 102, OrderStatus.CANCELED)
    assert _tick(s, 102, B_ASK, 29) == [], "the same book: asking again would only miss again"
    assert _tick(s, 102, B_ASK, 20) == [], "however long it waits"
    assert _tick(s, 102, [(0.50, 900)], 19) != [], "its top moved: the book has something new to say"


@pytest.mark.parametrize("status", [OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED])
def test_any_order_that_ends_short_backs_off(make, status):
    s = make()
    _tick(s, 102, B_ASK, 30)
    s.on_execution_report(_Report(*LISTINGS[102], status))
    assert s._legs[[leg.listing_id for leg in s._legs].index(102)].backoff_top is not None


def test_a_miss_on_one_venue_leaves_the_others_trading(make):
    s = make()
    _tick(s, 102, B_ASK, 30)
    _done(s, 102, OrderStatus.CANCELED)
    out = _tick(s, 202, [(0.52, 1000)], 29)
    assert [_key(o) for o in out] == [LISTINGS[202]], "Polymarket backed off; Kalshi's ask still clears"


def test_a_full_fill_does_not_back_off_and_decides_again_on_the_next_update(make):
    s = make()
    o = _tick(s, 102, [(0.50, 20), (0.55, 1000)], 30)[0]
    s.positions.fill(102, shares=o.take_size / S, price=0.50, fee=o.take_size / S * taker_fee(0.50, FEE))
    _done(s, 102, OrderStatus.FILLED)
    calls = _decisions(s)
    _tick(s, 202, [(0.95, 1000)], 29)
    assert calls, "holdings changed, so the Kelly size did: decide again"
    assert s._legs[[leg.listing_id for leg in s._legs].index(102)].backoff_top is None


def test_only_a_top_of_book_change_or_a_new_model_price_triggers_a_decision(make):
    s = make()
    calls = _decisions(s)
    _tick(s, 102, [(0.58, 1000), (0.60, 10)], 30)
    assert len(calls) == 1, "the window opened: decide"
    _tick(s, 102, [(0.58, 1000), (0.61, 10)], 29)
    _tick(s, 102, [(0.58, 1000), (0.61, 10)], 28)
    assert len(calls) == 1, "only a deeper level moved, then nothing did"
    _tick(s, 102, [(0.57, 1000), (0.61, 10)], 27)
    assert len(calls) == 2, "the best ask moved"


def test_a_book_seen_before_the_window_is_decided_on_when_it_opens(make):
    s = make()
    assert _tick(s, 102, B_ASK, 90) == []
    assert _tick(s, 102, B_ASK, 30) != [], "nothing in the book changed, but the window opened"


def test_a_change_while_an_order_is_in_flight_is_decided_on_after_it(make):
    s = make()
    _tick(s, 102, B_ASK, 30)
    _tick(s, 202, [(0.52, 1000)], 29)
    _done(s, 102, OrderStatus.CANCELED)
    assert [_key(o) for o in _tick(s, 102, B_ASK, 28)] == [LISTINGS[202]], \
        "Kalshi's new ask arrived while Polymarket's order was out; it is acted on once that order is back"


def test_each_order_and_each_back_off_is_logged(make, caplog):
    s = make()
    with caplog.at_level("INFO", logger=mod.__name__):
        _tick(s, 102, [(0.50, 100), (0.51, 50), (0.70, 5)], 30)
        _done(s, 102, OrderStatus.CANCELED)
    order, backoff = [r.getMessage() for r in caplog.records]
    assert order.startswith("match 7 BUY polymarket team B (listing 102): model 0.400")
    assert "fair 0.600" in order and "best ask 0.500" in order
    assert "over [0.500x100.00, 0.510x50.00]" in order, "the levels the order reaches, not the ones past its limit"
    assert backoff.startswith("match 7 polymarket team B (listing 102): order CANCELED with shares left, backing off")


# ---- regression: a prod paper session's own books ----------------------------------------------------------------

# Paper session 7450c387 (2026-11-04, Azuolas v Walczaki), 11:05:30-11:10 UTC: every book update of its six listings
# from its journal, kickoff 12:00. Prod sent 110 IOCs in that span, about one per update, 105 filling nothing.
JOURNAL_BOOKS = Path(__file__).parent / "fixtures" / "cs2_series_books_7450c387.csv"
JOURNAL_KICKOFF = pd.Timestamp("2026-11-04 12:00", tz="UTC").value
# Its book streams, as this file's listings: team A is Azuolas.
JOURNAL_STREAMS = {2230413: 101, 2230414: 102, 2224916: 201, 2224917: 202, 2223067: 301, 2223066: 302}
# The blend weights in prod that day; the stand-in model's 0.40 is close to the session's 0.403.
PROD_BLEND = MarketBlend(market=0.6123418580625587, model=0.5130722002876456, intercept=0.008852116804822632)


class _PaperExchange:
    """Fills IOCs the way the paper simulator does: a level offers its displayed size less what we already took there
    (the feed never shows our fills), and what we took is given back only as the level shrinks."""

    def __init__(self):
        self.asks, self.consumed = {}, {}

    def on_book(self, listing, asks):
        old, taken = self.asks.get(listing, {}), self.consumed.setdefault(listing, {})
        for price in list(taken):
            shrink = max(0.0, old.get(price, 0.0) - asks.get(price, 0.0))
            taken[price] = min(max(0.0, taken[price] - shrink), asks.get(price, 0.0))
        self.asks[listing] = asks

    def buy(self, listing, shares, limit):
        taken, got, cost = self.consumed.setdefault(listing, {}), 0.0, 0.0
        for price, size in sorted(self.asks[listing].items()):
            if price > limit + 1e-9 or got >= shares - 1e-9:
                break
            take = min(max(0.0, size - taken.get(price, 0.0)), shares - got)
            got, cost = got + take, cost + take * price
            taken[price] = taken.get(price, 0.0) + take
        return got, cost / got if got else 0.0


def test_a_prod_paper_sessions_books_no_longer_draw_an_order_per_update(make):
    books = pd.read_csv(JOURNAL_BOOKS)
    s, exchange, orders = make(models=_models(blend=PROD_BLEND)), _PaperExchange(), []
    for r in books.itertuples():
        listing = JOURNAL_STREAMS[r.stream]
        level = lambda side, i: (getattr(r, f"{side}p{i}"), getattr(r, f"{side}s{i}"))
        asks = [level("a", i) for i in range(10) if not pd.isna(getattr(r, f"as{i}"))]
        bids = [level("b", i) for i in range(10) if not pd.isna(getattr(r, f"bs{i}"))]
        exchange.on_book(listing, dict(asks))
        out = _tick(s, listing, asks, (JOURNAL_KICKOFF - r.ts) / MINUTE, bids)
        for o in out:
            bought = next(lid for lid, venue in LISTINGS.items() if venue == _key(o))
            shares, limit = o.take_size / S, o.take_limit_price / P
            got, price = exchange.buy(bought, shares, limit)
            if got:
                s.positions.fill(bought, shares=got, price=price, fee=got * taker_fee(price, FEE))
            orders.append((bought, shares, got))
            s.on_execution_report(SimpleNamespace(
                exchange_id=o.exchange_id, security_id=o.security_id, filled_qty=round(got * S),
                order_status=OrderStatus.FILLED if got >= shares - 1e-9 else OrderStatus.CANCELED))
    assert orders[0][0] == 301 and orders[0][2] == orders[0][1], "Polymarket US's Azuolas ask, filled in full"
    assert len(orders) <= 25, f"{len(orders)} orders for {len(books)} book updates"
    assert sum(got > 0 for _, _, got in orders) >= len(orders) // 4, "orders go out when the book has something new, so a fair share of them fill"
