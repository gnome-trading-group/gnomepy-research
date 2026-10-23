"""
CS2Series: entry maths, the match feed's view of time, windows, lineups, budgets, restart safety.

The strategy resolves listings through RegistryClient and loads its models in __init__, so the
registry and the model loader are replaced with in-memory stand-ins (a blend that keeps only the
pre-map price, unless a test says otherwise); the match feed reads real
parquet extracts; positions and market data are plain objects with the engine's interface.
Nothing here needs a JVM.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gnomepy import OrderStatus, OrderType, Scales, Side
from gnomepy.data import ArtifactRef
from gnomepy_research.models.cs2.hltv.bayes_series import SeriesBlend
from gnomepy_research.models.cs2.win_probability.series_pricing import price
from gnomepy_research.sessions.cs2_series import strategy as mod
from gnomepy_research.sessions.cs2_series.decision import (
    Level,
    choose_entry,
    cost_per_share,
    evaluate_side,
    kelly_shares,
    kelly_take,
    max_entry_price,
    taker_fee,
)
from gnomepy_research.sessions.cs2_series.match_feed import MatchFeed
from tests.sessions.conftest import FakeFeeSchedule

P, S = Scales.PRICE, Scales.SIZE
HOUR = 3_600_000_000_000
KICKOFF = pd.Timestamp("2026-10-05T18:00:00Z")
K_NS = KICKOFF.value
LISTINGS = {101: (5, 1101), 102: (5, 1102)}
LINEUP_A, LINEUP_B = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]


# ---- decision maths ------------------------------------------------------------------

def test_taker_fee_is_the_parametric_formula():
    assert taker_fee(0.5, 0.07) == pytest.approx(0.0175)
    assert taker_fee(0.9, 0.07) == pytest.approx(0.0063)


def test_max_entry_price_leaves_exactly_the_threshold():
    for fair, edge in ((0.6, 0.05), (0.75, 0.10), (0.35, 0.03)):
        cap = max_entry_price(fair, edge, 0.07)
        assert fair - cap - taker_fee(cap, 0.07) == pytest.approx(edge, abs=1e-9)


def test_max_entry_price_without_fee_is_fair_minus_edge():
    assert max_entry_price(0.6, 0.05, 0.0) == pytest.approx(0.55)


def _side(fair, asks, fee_rate=0.07):
    return evaluate_side(0, fair, asks, min_edge=0.05, fee_rate=fee_rate, tick=0.01)


def test_no_entry_when_best_ask_does_not_clear_the_threshold():
    assert _side(0.6, [Level(0.58, 100)]) is None


def test_the_entry_reaches_every_level_that_still_clears_the_threshold():
    # fair 0.70, 5c edge, 7% fee: the cap is the price p with 0.70 - p - 0.07 p (1 - p) = 0.05, about 0.6339.
    asks = [Level(0.40, 50), Level(0.50, 50), Level(0.63, 50), Level(0.64, 50)]
    e = _side(0.70, asks)
    assert e.limit_price == pytest.approx(0.63), "floored to the tick below the cap, however far above the best ask"
    assert [lv.price for lv in e.levels] == [0.40, 0.50, 0.63] and e.shares_available == pytest.approx(150)
    for lv in e.levels:
        assert 0.70 - lv.price - taker_fee(lv.price, 0.07) >= 0.05 - 1e-12


def test_the_threshold_price_cuts_the_book():
    e = _side(0.58, [Level(0.50, 50), Level(0.53, 50)])
    assert e.limit_price <= max_entry_price(0.58, 0.05, 0.07)
    assert e.shares_available == pytest.approx(50), "the 0.53 level is above the cap"


def test_only_best_asks_inside_the_price_band_are_bought():
    band = dict(min_edge=0.02, fee_rate=0.0, tick=0.01, min_price=0.20, max_price=0.90)
    assert evaluate_side(0, 0.30, [Level(0.19, 100)], **band) is None, "a long shot under the floor"
    assert evaluate_side(0, 0.30, [Level(0.20, 100)], **band) is not None, "the floor itself is in the band"
    assert evaluate_side(0, 0.99, [Level(0.89, 100)], **band) is not None
    assert evaluate_side(0, 0.99, [Level(0.90, 100)], **band) is None, "the top of the band is out"


def test_no_share_is_bought_at_or_above_the_top_of_the_band():
    e = evaluate_side(0, 0.99, [Level(0.88, 10), Level(0.89, 10), Level(0.90, 10), Level(0.91, 10)],
                      min_edge=0.02, fee_rate=0.0, tick=0.01, min_price=0.20, max_price=0.90)
    assert e.limit_price == pytest.approx(0.89) and [lv.price for lv in e.levels] == [0.88, 0.89]


def test_choose_entry_applies_the_band_to_each_side():
    e = choose_entry(0.20, [Level(0.10, 100)], [Level(0.75, 100)], min_edge=0.02, fee_rates=(0.0, 0.0), tick=0.01,
                     min_price=0.20, max_price=0.90)
    assert e.side == 1, "team A's 10c ask has the bigger edge but is under the floor"


def test_choose_entry_buys_the_underpriced_side_and_respects_allowed_sides():
    asks_a, asks_b = [Level(0.62, 100)], [Level(0.30, 100)]
    e = choose_entry(0.60, asks_a, asks_b, min_edge=0.05, fee_rates=(0.07, 0.07), tick=0.01)
    assert e.side == 1 and e.fair == pytest.approx(0.40)
    assert choose_entry(0.60, asks_a, asks_b, min_edge=0.05, fee_rates=(0.07, 0.07), tick=0.01,
                        allowed_sides=(0,)) is None


def test_choose_entry_charges_each_side_its_own_listings_fee():
    asks_a, asks_b = [Level(0.62, 100)], [Level(0.30, 100)]
    e = choose_entry(0.60, asks_a, asks_b, min_edge=0.05, fee_rates=(0.0, 0.07), tick=0.01)
    assert e.edge_at_best == pytest.approx(0.40 - 0.30 - taker_fee(0.30, 0.07))
    assert choose_entry(0.60, asks_a, asks_b, min_edge=0.05, fee_rates=(0.07, 0.5), tick=0.01) is None, \
        "side B's 10c gross edge does not survive a 0.5 rate"


def test_from_nothing_kelly_spends_the_fraction_for_a_contract_paying_one():
    u = cost_per_share(0.40, 0.07)
    spent = kelly_shares(0.60, u, 1000.0, 0.0, 0.0) * u
    assert spent == pytest.approx(1000.0 * (0.60 - u) / (1 - u))


def test_kelly_tops_up_to_the_same_holding_whatever_was_bought_on_the_way():
    u = cost_per_share(0.40, 0.07)
    full = kelly_shares(0.60, u, 1000.0, 0.0, 0.0)
    part = full / 3
    assert kelly_shares(0.60, u, 1000.0, part, part * u) == pytest.approx(full - part)
    assert kelly_shares(0.60, u, 1000.0, full, full * u) == pytest.approx(0.0, abs=1e-9)
    assert kelly_shares(0.60, u, 1000.0, 2 * full, 2 * full * u) < 0, "past the Kelly size: nothing more to buy"


def test_no_kelly_size_without_an_edge():
    assert kelly_shares(0.40, cost_per_share(0.40, 0.07), 1000.0, 0.0, 0.0) < 0


def test_a_take_stops_at_the_kelly_size_in_a_deep_book():
    e = _side(0.80, [Level(0.40, 10_000), Level(0.45, 10_000)])
    take = kelly_take(e, 100.0, 0.0, 0.0, max_cost=1e9, fee_rate=0.07, lot=1.0)
    u = cost_per_share(0.40, 0.07)
    assert take.limit_price == pytest.approx(0.40), "the best level alone holds the Kelly size"
    assert take.shares == pytest.approx(math.floor(kelly_shares(0.80, u, 100.0, 0.0, 0.0)))


def test_a_take_walks_levels_each_at_its_own_price_and_fee_until_the_kelly_size():
    e = _side(0.80, [Level(0.40, 20), Level(0.45, 30), Level(0.50, 10_000)])
    take = kelly_take(e, 100.0, 0.0, 0.0, max_cost=1e9, fee_rate=0.07, lot=1.0)
    by_hand = 20 * cost_per_share(0.40, 0.07) + 30 * cost_per_share(0.45, 0.07)
    u = cost_per_share(0.50, 0.07)
    rest = math.floor(kelly_shares(0.80, u, 100.0, 50.0, by_hand))
    assert take.shares == pytest.approx(50 + rest)
    assert take.cost == pytest.approx(by_hand + rest * u) and take.limit_price == pytest.approx(0.50)


def test_a_take_stops_at_the_cap_before_the_kelly_size():
    e = _side(0.80, [Level(0.40, 10_000)])
    take = kelly_take(e, 10_000.0, 0.0, 0.0, max_cost=50.0, fee_rate=0.07, lot=1.0)
    assert 50.0 - cost_per_share(0.40, 0.07) < take.cost <= 50.0


def test_a_take_counts_what_is_already_held_against_the_cap_and_the_kelly_size():
    e = _side(0.80, [Level(0.40, 10_000)])
    u = cost_per_share(0.40, 0.07)
    assert kelly_take(e, 10_000.0, 100.0, 45.0, max_cost=50.0, fee_rate=0.07, lot=1.0).cost <= 5.0
    full = kelly_shares(0.80, u, 100.0, 0.0, 0.0)
    assert kelly_take(e, 100.0, full, full * u, max_cost=1e9, fee_rate=0.07, lot=1.0).shares == 0.0


def test_a_take_is_whole_lots_and_stops_with_the_book():
    e = _side(0.80, [Level(0.40, 7.5)])
    take = kelly_take(e, 10_000.0, 0.0, 0.0, max_cost=50.0, fee_rate=0.07, lot=2.0)
    assert take.shares == pytest.approx(6.0) and take.cost == pytest.approx(6 * cost_per_share(0.40, 0.07))


# ---- the match's rows ----------------------------------------------------------------

class _MapModel:
    """Prices every map at the row's p_map, so a test sets the series price through it."""

    feature_names = ["p_map"]

    def predict_batch(self, X):
        return X[:, 0]


def series_of(p_map: float) -> float:
    return float(price(_MapModel(), pd.DataFrame([{"match_id": 7, "p_map": p_map}])).series.iloc[0])


# A map price whose series price is about 0.40: team B looks underpriced against a 0.45 ask.
LOW = 0.43


def _feature(at, p_map=LOW, lineup_a=LINEUP_A, match_id=7):
    return {"match_id": match_id, "map_name": "", "generated_at": at, "p_map": p_map, "match_date": KICKOFF.normalize(),
            "team_a_id": 1, "team_b_id": 2, "team_a_lineup_ids": lineup_a, "team_b_lineup_ids": LINEUP_B}


class _Posterior:
    """Prices every series at a fixed P(team A); fitted as_of a date before KICKOFF unless a test says otherwise."""

    def __init__(self, p=0.5, as_of=f"{KICKOFF - pd.Timedelta(days=3):%Y-%m-%d}"):
        self.p, self.as_of = p, as_of

    def price(self, *_):
        return self.p


def _models(posterior=None, blend=SeriesBlend(0.0, 1.0)):
    return mod.Models(_MapModel(), posterior or _Posterior(), blend, {"premap": "m", "posterior": "p", "blend": "b"})


def _state(at, kickoff=KICKOFF, status="upcoming", venues=("polymarket",)):
    markets = [{"venue": v, "market": "series", "listing_team_a": 101, "listing_team_b": 102} for v in venues]
    return {"match_id": 7, "generated_at": at, "kickoff": kickoff, "status": status, "markets": json.dumps(markets)}


EARLY = KICKOFF - pd.Timedelta(hours=24)
RUN_EVERY = pd.Timedelta(minutes=15)


def _states(*changes, until=KICKOFF + pd.Timedelta(hours=6)):
    """A matches row every pipeline run from EARLY, as cs2_upcoming writes them; changes are
    (from_time, {"kickoff": ..., "status": ...}) applied from that run on."""
    rows, current = [], {"kickoff": KICKOFF, "status": "upcoming", "venues": ("polymarket",)}
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
    def pricer(rows):
        calls.append(len(rows))
        return price(_MapModel(), rows)["series"].tolist()
    return pricer


def test_the_feed_is_point_in_time(tmp_path):
    t0 = KICKOFF - pd.Timedelta(hours=10)
    f, m = _write(tmp_path, [_feature(t0, p_map=0.5), _feature(t0 + pd.Timedelta(hours=1), p_map=0.6)],
                  [_state(t0), _state(t0 + pd.Timedelta(hours=2), kickoff=KICKOFF + pd.Timedelta(hours=2))])
    feed = MatchFeed(7, _pricer([]), f, m, reload_seconds=0)
    assert feed.quote(t0.value - 1) is None and feed.state(t0.value - 1) is None, "nothing before it was written"
    assert feed.quote(t0.value).p_team_a == pytest.approx(series_of(0.5))
    assert feed.quote((t0 + pd.Timedelta(hours=5)).value).p_team_a == pytest.approx(series_of(0.6))
    assert feed.state((t0 + pd.Timedelta(hours=1)).value).kickoff_ns == K_NS
    assert feed.state((t0 + pd.Timedelta(hours=3)).value).kickoff_ns == (KICKOFF + pd.Timedelta(hours=2)).value


def test_the_feed_reads_only_its_match_and_carries_lineups(tmp_path):
    f, m = _write(tmp_path, [_feature(EARLY, lineup_a=[]), _feature(EARLY, p_map=0.9, match_id=8)], [_state(EARLY)])
    q = MatchFeed(7, _pricer([]), f, m, reload_seconds=0).quote(K_NS)
    assert q.p_team_a == pytest.approx(series_of(LOW)) and not q.lineup_known
    assert q.lineups == ((), tuple(LINEUP_B))


def test_reload_prices_only_new_rows_and_survives_a_bad_read(tmp_path):
    calls = []
    f, m = _write(tmp_path, [_feature(EARLY, p_map=0.5)], [_state(EARLY)])
    feed = MatchFeed(7, _pricer(calls), f, m, reload_seconds=0.05)
    later = EARLY + pd.Timedelta(hours=1)
    _write(tmp_path, [_feature(EARLY, p_map=0.5), _feature(later, p_map=0.6)], [_state(EARLY)])
    deadline = time.monotonic() + 5
    while feed.quote(K_NS).at_ns != later.value and time.monotonic() < deadline:
        time.sleep(0.02)
    assert feed.quote(K_NS).p_team_a == pytest.approx(series_of(0.6))
    assert calls == [1, 1], "the first row is not priced again"
    (tmp_path / "features.parquet").write_bytes(b"not a parquet file")
    time.sleep(0.2)
    assert feed.quote(K_NS).at_ns == later.value, "a failed reload keeps the rows already held"
    # The reload thread outlives the test (it is a daemon, as in production); give it a readable file.
    _write(tmp_path, [_feature(EARLY, p_map=0.5)], [_state(EARLY)])


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
    orders."""

    def __init__(self):
        self.filled: dict = {}
        self.pending: dict = {}

    def fill(self, key, shares, price, fee=0.0):
        q, avg, fees = self.filled.get(key, (0, 0, 0.0))
        nq = q + int(shares * S)
        self.filled[key] = (nq, int((q * avg + int(shares * S) * int(price * P)) / nq), fees + fee * P)

    def get_position(self, exchange_id, security_id):
        f = self.filled.get((exchange_id, security_id))
        return _Pos(*f) if f else None

    def get_effective_quantity(self, exchange_id, security_id):
        key = (exchange_id, security_id)
        return self.filled.get(key, (0, 0, 0.0))[0] + self.pending.get(key, 0)

    def fee_schedule(self, exchange_id, security_id):
        return FakeFeeSchedule(taker=0.07, maker=0.0)


class _Book:
    def __init__(self, key, asks, ts, bids=()):
        self.exchange_id, self.security_id = key
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
    def _make(features=None, states=None, spec=None, models=None, **kw):
        f, m = _write(tmp_path, features or [_feature(EARLY)], states or _states())
        r = tmp_path / "rankings.parquet"
        pd.DataFrame({"date": [KICKOFF - pd.Timedelta(days=7)] * 2, "team_id": [1, 2], "rank": [5, 9]}).to_parquet(r)
        with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry(spec or _Spec())), \
                patch.object(mod, "load_models", lambda *paths: models or _models()):
            kw.setdefault("bankroll_usd", 10_000.0)
            strat = mod.CS2Series(match_id=7, listing_team_a=101, listing_team_b=102, reload_minutes=0,
                                  features_source=f, matches_source=m, rankings_source=str(r), **kw)
        strat._position_view = _Positions()
        strat._control = _Control()
        return strat
    return _make


def _tick(strat, listing, asks, hours_before, bids=()):
    return strat.on_market_data(_Book(LISTINGS[listing], asks, K_NS - int(hours_before * HOUR), bids))


def test_buys_the_underpriced_team_with_an_ioc_limit(make):
    out = _tick(make(), 102, [(0.45, 1000)], hours_before=6)
    assert len(out) == 1
    o = out[0]
    assert (o.exchange_id, o.security_id) == LISTINGS[102], "team B is underpriced, so buy team B's listing"
    assert o.take_side == Side.BID and o.take_order_type == OrderType.LIMIT
    assert 0.45 * P <= o.take_limit_price <= 0.47 * P


def test_nothing_outside_the_entry_window(make):
    s = make()
    assert _tick(s, 102, [(0.45, 1000)], hours_before=13) == [], "the window opens 12h out"
    assert _tick(s, 102, [(0.45, 1000)], hours_before=3.01) != [], "and is open until the cutoff"
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.CANCELED))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=3) == [], "it closes 3h before kickoff"


def test_the_window_follows_the_latest_kickoff(make):
    s = make(states=_states((EARLY + pd.Timedelta(hours=1), {"kickoff": KICKOFF + pd.Timedelta(hours=3)})))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=1) != [], "past the first cutoff, 4h before the new kickoff"
    assert not s.is_exiting, "a delayed kickoff moves the cutoff"
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.CANCELED))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=-0.1) == [], "and closed at the new cutoff"
    assert s._control.exits == [("match 7 reached its entry cutoff", False)]


@pytest.mark.parametrize("status", ["live", "missing"])
def test_the_session_ends_once_hltv_says_live_or_drops_the_match(make, status):
    s = make(states=_states((KICKOFF - pd.Timedelta(hours=5), {"status": status})))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=6) != []
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.CANCELED))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=4.5) == []
    assert s._control.exits == [(f"match 7 is {status}", False)]


@pytest.mark.parametrize("holding", [False, True])
def test_the_session_ends_at_the_cutoff_keeping_whatever_it_holds(make, holding):
    s = make()
    if holding:
        s.positions.fill(LISTINGS[102], shares=10, price=0.45)
    _tick(s, 102, [(0.95, 1000)], hours_before=3.5)
    assert not s.is_exiting
    assert _tick(s, 102, [(0.45, 1000)], hours_before=2.99) == []
    assert s._control.exits == [("match 7 reached its entry cutoff", False)], "positions are kept, not flattened"


def test_the_session_waits_for_an_order_in_flight_before_ending(make):
    s = make()
    assert _tick(s, 102, [(0.45, 1000)], hours_before=3.01) != []
    _tick(s, 102, [(0.45, 1000)], hours_before=2.99)
    assert not s.is_exiting, "an IOC still working"
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.FILLED))
    _tick(s, 102, [(0.45, 1000)], hours_before=2.98)
    assert s._control.exits == [("match 7 reached its entry cutoff", False)]


def test_the_session_ends_at_the_cutoff_even_on_stale_state(make):
    s = make(states=_states(until=KICKOFF - pd.Timedelta(hours=8)))
    _tick(s, 102, [(0.45, 1000)], hours_before=3.5)
    assert not s.is_exiting, "stale state stops trading"
    _tick(s, 102, [(0.45, 1000)], hours_before=3)
    assert s._control.exits == [("match 7 reached its entry cutoff", False)]


@pytest.mark.parametrize("kw", [{"entry_cutoff_hours": 12.0}, {"min_price": 0.9, "max_price": 0.9}])
def test_a_session_needs_a_window_and_a_band(make, kw):
    with pytest.raises(ValueError):
        make(**kw)


def test_a_side_priced_outside_the_band_is_not_bought(make):
    s = make()
    assert _tick(s, 102, [(0.19, 1000)], hours_before=6) == [], "team B is cheap enough to clear the edge, but a long shot"


def test_no_trading_on_stale_match_state(make):
    s = make(states=_states(until=KICKOFF - pd.Timedelta(hours=6)))
    for hours_before, trades in ((6, True), (5.75, True), (5.4, False)):
        out = _tick(s, 102, [(0.45, 1000)], hours_before=hours_before)
        assert (out != []) == trades, f"{hours_before}h before kickoff, the pipeline stalled 6h before"
        s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.CANCELED))


def test_the_feed_rereads_faster_near_kickoff(tmp_path):
    f, m = _write(tmp_path, [_feature(EARLY)], _states())
    feed = MatchFeed(7, _pricer([]), f, m, reload_seconds=300, near_kickoff_reload_seconds=60, near_kickoff_seconds=3600)
    assert feed.reload_interval(K_NS - 2 * HOUR) == 300
    assert feed.reload_interval(K_NS - HOUR // 2) == 60
    assert feed.reload_interval(K_NS + HOUR) == 60, "and after kickoff, until HLTV shows it live"


def test_no_trade_before_the_first_features_row(make):
    s = make(features=[_feature(KICKOFF - pd.Timedelta(hours=5))])
    assert _tick(s, 102, [(0.45, 1000)], hours_before=6) == [], "a replay must not trade on a future row"
    assert _tick(s, 102, [(0.45, 1000)], hours_before=4.9) != []


def test_no_trade_until_both_lineups_are_announced(make):
    s = make(features=[_feature(EARLY, lineup_a=[]), _feature(KICKOFF - pd.Timedelta(hours=5))])
    assert _tick(s, 102, [(0.45, 1000)], hours_before=6) == []
    assert _tick(s, 102, [(0.45, 1000)], hours_before=4.9) != []


def test_a_lineup_change_stops_top_ups(make):
    changed = [9, 2, 3, 4, 5]
    s = make(features=[_feature(EARLY), _feature(KICKOFF - pd.Timedelta(hours=4.5), lineup_a=changed)])
    assert _tick(s, 102, [(0.45, 10)], hours_before=6)
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.FILLED))
    s.positions.fill(LISTINGS[102], shares=10, price=0.45)
    assert _tick(s, 102, [(0.45, 1000)], hours_before=5) != [], "same lineups: topping up is fine"
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.FILLED))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=4) == [], "a stand-in is news the model cannot price"


def test_a_lineup_change_before_any_fill_does_not_block_entry(make):
    s = make(features=[_feature(EARLY), _feature(KICKOFF - pd.Timedelta(hours=4.5), lineup_a=[9, 2, 3, 4, 5])])
    assert _tick(s, 102, [(0.45, 1000)], hours_before=6)
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.CANCELED))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=4) != [], "nothing was bought on the old lineups"


def test_no_second_order_while_one_is_in_flight(make):
    s = make()
    assert _tick(s, 102, [(0.45, 10)], hours_before=6)
    assert _tick(s, 102, [(0.45, 1000)], hours_before=5.9) == []
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.FILLED))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=5.8), "a terminal report frees the market to top up"


def test_the_stake_cap_is_respected_including_fees(make):
    o = _tick(make(max_stake_usd=50.0), 102, [(0.45, 10_000)], hours_before=6)[0]
    assert o.take_size / S * cost_per_share(o.take_limit_price / P, 0.07) <= 50.0 + 1e-6


def test_a_thin_book_is_walked_to_the_level_the_stake_needs_and_no_further(make):
    asks = [(0.45, 20), (0.46, 20), (0.47, 10_000), (0.48, 10_000)]
    o = _tick(make(max_stake_usd=50.0), 102, asks, hours_before=6)[0]
    assert o.take_limit_price == round(0.47 * P), "40 shares at 0.45-0.46 leave most of the stake for 0.47"
    shares = o.take_size / S
    cost = 20 * cost_per_share(0.45, 0.07) + 20 * cost_per_share(0.46, 0.07) + (shares - 40) * cost_per_share(0.47, 0.07)
    assert 50.0 - cost_per_share(0.47, 0.07) < cost <= 50.0 + 1e-9


def test_stake_already_spent_blocks_further_orders(make):
    s = make(max_stake_usd=50.0)
    s.positions.fill(LISTINGS[102], shares=105, price=0.45, fee=1.8)
    assert _tick(s, 102, [(0.45, 10_000)], hours_before=6) == []


FAIR_B = 1 - series_of(LOW)


def _kelly_spend(bankroll, ask=0.45, fraction=0.5, venues=1):
    u = cost_per_share(ask, 0.07)
    return fraction * bankroll / venues * (FAIR_B - u) / (1 - u)


def test_a_bankroll_too_small_for_the_cap_buys_the_kelly_size(make):
    o = _tick(make(bankroll_usd=100.0, max_stake_usd=50.0), 102, [(0.45, 10_000)], hours_before=6)[0]
    spend = o.take_size / S * cost_per_share(0.45, 0.07)
    assert spend == pytest.approx(_kelly_spend(100.0), abs=cost_per_share(0.45, 0.07))
    assert spend < 50.0, "the Kelly size, not the cap, set this order"


def test_the_kelly_fraction_scales_the_size(make):
    half = _tick(make(bankroll_usd=100.0, kelly_fraction=0.5), 102, [(0.45, 10_000)], hours_before=6)[0]
    full = _tick(make(bankroll_usd=100.0, kelly_fraction=1.0), 102, [(0.45, 10_000)], hours_before=6)[0]
    assert full.take_size == pytest.approx(2 * half.take_size, abs=S // 100)


def test_the_bankroll_is_split_across_the_venues_with_a_series_market(make):
    one = _tick(make(bankroll_usd=100.0), 102, [(0.45, 10_000)], hours_before=6)[0]
    three = make(bankroll_usd=100.0, states=_states((EARLY, {"venues": ("polymarket", "kalshi", "polymarket_us")})))
    o = _tick(three, 102, [(0.45, 10_000)], hours_before=6)[0]
    assert o.take_size == pytest.approx(one.take_size / 3, abs=S // 100), "three sessions take the same view"


def test_a_held_kelly_size_is_topped_up_only_when_the_price_improves(make):
    s = make(bankroll_usd=100.0)
    o = _tick(s, 102, [(0.45, 10_000)], hours_before=6)[0]
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.FILLED))
    shares = o.take_size / S
    s.positions.fill(LISTINGS[102], shares=shares, price=0.45, fee=shares * taker_fee(0.45, 0.07))
    assert _tick(s, 102, [(0.45, 10_000)], hours_before=5) == [], "already at the Kelly size for this price"
    top_up = _tick(s, 102, [(0.40, 10_000)], hours_before=4)
    assert top_up, "a cheaper ask means a bigger edge and a bigger Kelly size"
    spent = shares * cost_per_share(0.45, 0.07) + top_up[0].take_size / S * cost_per_share(0.40, 0.07)
    assert spent < _kelly_spend(100.0, ask=0.40) + 1.0, "never past the Kelly size at the better price"


def test_a_thin_book_is_topped_up_to_the_kelly_size_once_it_deepens(make):
    s = make(bankroll_usd=100.0)
    o = _tick(s, 102, [(0.45, 5)], hours_before=6)[0]
    assert o.take_size == 5 * S, "only 5 shares on offer, short of the Kelly size"
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.FILLED))
    s.positions.fill(LISTINGS[102], shares=5, price=0.45, fee=5 * taker_fee(0.45, 0.07))
    top_up = _tick(s, 102, [(0.45, 10_000)], hours_before=5.9)
    assert top_up, "the fee paid on the first fill doesn't use up the stake"
    spent = (5 + top_up[0].take_size / S) * cost_per_share(0.45, 0.07)
    assert spent == pytest.approx(_kelly_spend(100.0), abs=cost_per_share(0.45, 0.07))


def test_edges_under_the_floor_are_not_bought(make):
    s = make(min_edge=0.02)
    ask = FAIR_B - 0.015 - taker_fee(FAIR_B - 0.015, 0.07)
    assert _tick(s, 102, [(math.floor(ask * 100) / 100 + 0.01, 10_000)], hours_before=6) == []
    assert _tick(s, 102, [(0.45, 10_000)], hours_before=6) != [], "the same session buys a bigger edge"


def test_a_session_needs_a_positive_bankroll(make):
    with pytest.raises(ValueError, match="bankroll_usd"):
        make(bankroll_usd=0.0)


def test_never_buys_the_other_side_once_holding_one(make):
    s = make(features=[_feature(EARLY, p_map=0.6)])
    s.positions.fill(LISTINGS[102], shares=10, price=0.30)
    assert _tick(s, 101, [(0.50, 1000)], hours_before=6) == [], "holding team B; team A now looks cheap - no hedge"


def test_adverse_move_stops_top_ups(make):
    s = make(adverse_move=0.08)
    s.positions.fill(LISTINGS[102], shares=10, price=0.45)
    assert _tick(s, 102, [(0.36, 1000)], hours_before=6) == [], "price fell 9c below entry: likely news"
    assert _tick(s, 102, [(0.44, 1000)], hours_before=6) != []


def test_restart_reads_committed_money_from_positions(make):
    """A fresh instance must see an earlier session's fills and not buy them again."""
    first = make(max_stake_usd=50.0)
    first.positions.fill(LISTINGS[102], shares=110, price=0.45, fee=1.9)
    restarted = make(max_stake_usd=50.0)
    restarted._position_view = first.positions
    assert _tick(restarted, 102, [(0.45, 10_000)], hours_before=6) == []


def test_restart_takes_the_lineups_it_finds_as_the_reference(make):
    s = make(features=[_feature(EARLY), _feature(KICKOFF - pd.Timedelta(hours=4.5), lineup_a=[9, 2, 3, 4, 5])])
    s.positions.fill(LISTINGS[102], shares=10, price=0.45)
    assert _tick(s, 102, [(0.45, 1000)], hours_before=6) != []
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.FILLED))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=4) == []


def test_an_order_under_the_venues_minimum_size_is_not_sent(make):
    s = make(spec=_Spec(min_size=5 * S))
    assert _tick(s, 102, [(0.45, 4)], hours_before=6) == [], "only 4 shares at the price; the venue wants 5"
    assert _tick(s, 102, [(0.45, 10)], hours_before=6) != []


def test_unknown_listing_data_is_ignored(make):
    assert make().on_market_data(_Book((5, 99999), [(0.01, 10)], K_NS - HOUR)) == []


def test_artifacts_are_fetched_at_the_version_they_resolved_to(monkeypatch):
    found = []

    class _Store:
        def find(self, query):
            found.append(query)
            return ArtifactRef(query.artifact_type, query.name, "20261015T163515680176Z", "s3://b/model.json")

        def resolve(self, ref):
            return "/tmp/model.json"
    monkeypatch.setattr(mod, "ArtifactStore", _Store)
    path, resolved = mod._fetch(mod.DEFAULT_MODEL + "@production")
    assert str(path) == "/tmp/model.json" and found[0].alias == "production"
    assert resolved == "artifact://xgboost_model/cs2_prematch_preveto:20261015T163515680176Z"
    monkeypatch.setattr(mod, "resolve_artifact_path", lambda path: "/local/model.json")
    assert mod._fetch("/local/model.json") == (mod.Path("/local/model.json"), "/local/model.json")


@pytest.mark.parametrize("as_of, trades", [
    (KICKOFF - pd.Timedelta(days=3), True),
    (KICKOFF - pd.Timedelta(days=20), False),
    (KICKOFF + pd.Timedelta(days=1), False),
])
def test_no_trading_on_a_stale_or_future_posterior(make, as_of, trades):
    s = make(models=_models(_Posterior(as_of=f"{as_of:%Y-%m-%d}")))
    assert (_tick(s, 102, [(0.45, 1000)], hours_before=6) != []) is trades


def test_the_bayesian_price_moves_the_blend(make):
    """All weight on the Bayesian model, which says team A at 0.8: team A's 0.6 ask is now the cheap side."""
    s = make(models=_models(_Posterior(p=0.8), blend=SeriesBlend(1.0, 0.0)))
    out = _tick(s, 101, [(0.60, 1000)], hours_before=6)
    assert len(out) == 1 and (out[0].exchange_id, out[0].security_id) == LISTINGS[101]


def test_series_pricing_goes_through_the_map_model():
    assert series_of(0.5) == pytest.approx(0.5)
    assert series_of(LOW) < LOW < 0.5, "a BO3 is more lopsided than one map"
    assert np.isclose(series_of(LOW), 0.40, atol=0.01)


def test_each_order_records_the_decision_behind_it(make):
    strat = make()
    strat._metric_recorder = _Metrics()
    strat.register_metrics()

    out = _tick(strat, 102, [(0.45, 1000)], hours_before=6)

    rows = strat.metrics.streams["cs2_series_orders"].rows
    assert len(out) == len(rows) == 1
    row = rows[0]
    assert row["timestamp"] == K_NS - 6 * HOUR
    assert row["best_ask"] == pytest.approx(0.45)
    assert round(row["limit_price"] * P) == out[0].take_limit_price
    assert 0 < row["edge_at_best"] < row["fair"] - row["best_ask"], "the edge is what's left after the taker fee"
    assert row["shares"] > 0
    assert row["kelly_bankroll"] == pytest.approx(0.5 * 10_000.0) and row["held_shares"] == 0.0
    assert row["fair_xgb"] == pytest.approx(series_of(LOW)) and row["fair_bayes"] == pytest.approx(0.5)
    assert math.isnan(row["spread"]), "no bid on the book"


def test_the_order_row_records_the_spread_it_was_taken_at(make):
    strat = make()
    strat._metric_recorder = _Metrics()
    strat.register_metrics()
    _tick(strat, 102, [(0.45, 1000)], hours_before=6, bids=[(0.41, 50)])
    assert strat.metrics.streams["cs2_series_orders"].rows[0]["spread"] == pytest.approx(0.04)
