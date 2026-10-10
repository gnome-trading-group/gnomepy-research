"""
CS2PreMatch: entry maths, windows, budgets, restart safety.

The strategy resolves listings through RegistryClient in __init__, so the
registry is replaced with an in-memory stand-in; positions and market data are
plain objects with the engine's interface. Nothing here needs a JVM.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd
import pytest

from gnomepy import OrderStatus, OrderType, Scales, Side
from gnomepy_research.sessions.cs2_prematch import cs2_prematch as mod
from gnomepy_research.sessions.cs2_prematch import make_config, polymarket, predictions
from gnomepy_research.sessions.cs2_prematch.decision import (
    Level,
    choose_entry,
    cost_per_share,
    evaluate_side,
    max_entry_price,
    shares_to_buy,
    taker_fee,
)

P, S = Scales.PRICE, Scales.SIZE
HOUR = 3_600_000_000_000
KICKOFF = pd.Timestamp("2026-10-05T18:00:00Z")
K_NS = KICKOFF.value
PM = 4
LISTINGS = {101: (PM, 1101), 102: (PM, 1102), 201: (PM, 1201), 202: (PM, 1202)}


# ---- decision maths ------------------------------------------------------------------

def test_taker_fee_matches_polymarket_formula():
    assert taker_fee(0.5, 0.07) == pytest.approx(0.0175)
    assert taker_fee(0.9, 0.07) == pytest.approx(0.0063)


def test_max_entry_price_leaves_exactly_the_threshold():
    for fair, edge in ((0.6, 0.05), (0.75, 0.10), (0.35, 0.03)):
        cap = max_entry_price(fair, edge, 0.07)
        assert fair - cap - taker_fee(cap, 0.07) == pytest.approx(edge, abs=1e-9)


def test_max_entry_price_without_fee_is_fair_minus_edge():
    assert max_entry_price(0.6, 0.05, 0.0) == pytest.approx(0.55)


def test_no_entry_when_best_ask_does_not_clear_the_threshold():
    asks = [Level(0.58, 100)]
    assert evaluate_side(0, 0.6, asks, min_edge=0.05, fee_rate=0.07, max_slippage=0.02, tick=0.01) is None


def test_limit_is_capped_by_slippage_and_floored_to_tick():
    asks = [Level(0.40, 50), Level(0.41, 50), Level(0.43, 50)]
    e = evaluate_side(0, 0.70, asks, min_edge=0.05, fee_rate=0.07, max_slippage=0.02, tick=0.01)
    assert e.limit_price == pytest.approx(0.42)
    assert e.shares_available == pytest.approx(100)


def test_limit_is_capped_by_the_threshold_price_when_that_is_lower():
    asks = [Level(0.50, 50), Level(0.53, 50)]
    e = evaluate_side(0, 0.58, asks, min_edge=0.05, fee_rate=0.07, max_slippage=0.05, tick=0.01)
    cap = max_entry_price(0.58, 0.05, 0.07)
    assert e.limit_price <= cap
    assert e.shares_available == pytest.approx(50), "the 0.53 level is above the cap"


def test_choose_entry_buys_the_underpriced_side_and_respects_allowed_sides():
    asks_a, asks_b = [Level(0.62, 100)], [Level(0.30, 100)]
    e = choose_entry(0.60, asks_a, asks_b, min_edge=0.05, fee_rate=0.07, max_slippage=0.02, tick=0.01)
    assert e.side == 1 and e.fair == pytest.approx(0.40)
    assert choose_entry(0.60, asks_a, asks_b, min_edge=0.05, fee_rate=0.07, max_slippage=0.02,
                        tick=0.01, allowed_sides=(0,)) is None


def test_shares_to_buy_includes_the_fee_in_the_budget():
    e = evaluate_side(0, 0.80, [Level(0.50, 10_000)], min_edge=0.05, fee_rate=0.07, max_slippage=0.0, tick=0.01)
    shares = shares_to_buy(e, 50.0, 0.07)
    assert shares * cost_per_share(e.limit_price, 0.07) == pytest.approx(50.0)


# ---- predictions table ---------------------------------------------------------------

def _preds(rows, generated_at=KICKOFF - pd.Timedelta(hours=24)):
    df = pd.DataFrame(rows)
    df["model_version"] = "test"
    df["kickoff"] = df["kickoff"].fillna(KICKOFF) if "kickoff" in df else KICKOFF
    df["generated_at"] = df["generated_at"].fillna(generated_at) if "generated_at" in df else generated_at
    return df


def test_predictions_validation_rejects_bad_tables():
    good = _preds([{"match_id": 1, "market": "series", "p_team_a": 0.6, "rank_known_both": 1.0}])
    predictions.validate(good)
    with pytest.raises(ValueError, match="strictly inside"):
        predictions.validate(good.assign(p_team_a=1.0))
    with pytest.raises(ValueError, match="unknown market"):
        predictions.validate(good.assign(market="game3"))
    with pytest.raises(ValueError, match="duplicate"):
        predictions.validate(pd.concat([good, good]))
    predictions.validate(pd.concat([good, good.assign(generated_at=KICKOFF)])), "a history of rows is allowed"
    with pytest.raises(ValueError, match="required"):
        predictions.validate(good.assign(kickoff=pd.NaT))
    with pytest.raises(ValueError, match="missing columns"):
        predictions.validate(good.drop(columns="model_version"))


def test_prediction_book_is_point_in_time():
    t0 = KICKOFF - pd.Timedelta(hours=10)
    df = predictions.validate(_preds([
        {"match_id": 1, "market": "series", "p_team_a": 0.55, "rank_known_both": 1.0, "generated_at": t0},
        {"match_id": 1, "market": "series", "p_team_a": 0.62, "rank_known_both": 1.0,
         "generated_at": t0 + pd.Timedelta(hours=1), "kickoff": KICKOFF + pd.Timedelta(hours=2)},
    ]))
    book = predictions.PredictionBook(df)
    assert book.as_of(1, "series", (t0 - pd.Timedelta(seconds=1)).value) is None, "nothing before it was written"
    assert book.as_of(1, "series", t0.value)["p_team_a"] == 0.55
    later = book.as_of(1, "series", (t0 + pd.Timedelta(hours=5)).value)
    assert later["p_team_a"] == 0.62 and later["kickoff_ns"] == (KICKOFF + pd.Timedelta(hours=2)).value
    assert book.as_of(1, "game1", t0.value) is None


def test_walk_forward_rows_are_visible_for_the_whole_entry_window():
    wf = pd.DataFrame({"match_id": [1, 1], "market": ["series", "game1"], "q": [0.6, 0.7]})
    priors = pd.DataFrame({"match_id": [1], "map_position_in_series": [1], "rank_known_both": [1.0]})
    df = predictions.from_walk_forward(wf, priors, pd.Series({1: KICKOFF}), "wf")
    book = predictions.PredictionBook(df)
    assert book.as_of(1, "series", (KICKOFF - pd.Timedelta(hours=12)).value)["p_team_a"] == 0.6


# ---- strategy ------------------------------------------------------------------------

@dataclass
class _Listing:
    exchange_id: int
    security_id: int


@dataclass
class _Spec:
    tick_size: int = P // 100
    lot_size: int = 1


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
    """Engine position view: filled quantity, average entry, fees, plus pending orders."""

    def __init__(self):
        self.filled: dict = {}
        self.pending: dict = {}

    def fill(self, key, shares, price, fee=0.0):
        q, avg, fees = self.filled.get(key, (0, 0, 0.0))
        nq = q + int(shares * S)
        navg = int((q * avg + int(shares * S) * int(price * P)) / nq)
        self.filled[key] = (nq, navg, fees + fee)

    def get_position(self, exchange_id, security_id):
        f = self.filled.get((exchange_id, security_id))
        return _Pos(*f) if f else None

    def get_effective_quantity(self, exchange_id, security_id):
        key = (exchange_id, security_id)
        return self.filled.get(key, (0, 0, 0.0))[0] + self.pending.get(key, 0)


class _Book:
    def __init__(self, key, asks, ts):
        self.exchange_id, self.security_id = key
        self.event_timestamp = ts
        self._asks = asks

    def ask_price(self, i):
        return int(self._asks[i][0] * P) if i < len(self._asks) else 0

    def ask_size(self, i):
        return int(self._asks[i][1] * S) if i < len(self._asks) else 0


@dataclass
class _Report:
    exchange_id: int
    security_id: int
    order_status: OrderStatus


def _market(market, a, b):
    return {"match_id": 7, "market": market, "listing_team_a": a, "listing_team_b": b}


@pytest.fixture
def make(tmp_path):
    def _make(p_series=0.60, p_map1=0.60, rank=1.0, markets=None, rows=None, **kw):
        path = tmp_path / "preds.parquet"
        _preds(rows or [{"match_id": 7, "market": "series", "p_team_a": p_series, "rank_known_both": rank},
                        {"match_id": 7, "market": "game1", "p_team_a": p_map1, "rank_known_both": rank}]).to_parquet(path)
        markets = markets or [_market("series", 101, 102), _market("game1", 201, 202)]
        with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()):
            strat = mod.CS2PreMatch(markets=markets, predictions_path=str(path), **kw)
        strat._position_view = _Positions()
        strat._test_path = path
        return strat
    return _make


def _tick(strat, listing, asks, hours_before):
    return strat.on_market_data(_Book(LISTINGS[listing], asks, K_NS - int(hours_before * HOUR)))


def test_buys_the_underpriced_team_with_an_ioc_limit(make):
    s = make(p_series=0.40)
    out = _tick(s, 102, [(0.45, 1000)], hours_before=6)
    assert len(out) == 1
    o = out[0]
    assert (o.exchange_id, o.security_id) == LISTINGS[102], "team B is underpriced, so buy team B's token"
    assert o.take_side == Side.BID and o.take_order_type == OrderType.LIMIT
    assert 0.45 * P <= o.take_limit_price <= 0.47 * P


def test_nothing_outside_the_entry_window(make):
    s = make(p_series=0.40)
    assert _tick(s, 102, [(0.45, 1000)], hours_before=13) == [], "series window opens 12h out"
    assert _tick(s, 102, [(0.45, 1000)], hours_before=-0.1) == [], "nothing after kickoff"


def test_map1_only_trades_in_the_final_hour_and_needs_ten_cents(make):
    s = make(p_map1=0.40)
    assert _tick(s, 202, [(0.42, 1000)], hours_before=3) == [], "map 1 window is the final hour"
    assert 0.60 - 0.49 - taker_fee(0.49, 0.07) < 0.10
    assert _tick(s, 202, [(0.49, 1000)], hours_before=0.5) == [], "9.3c after fee is under map 1's 10c"
    assert _tick(s, 202, [(0.42, 1000)], hours_before=0.5) != []


def test_no_second_order_while_one_is_in_flight(make):
    s = make(p_series=0.40)
    assert _tick(s, 102, [(0.45, 10)], hours_before=6)
    assert _tick(s, 102, [(0.45, 1000)], hours_before=5.9) == []
    s.on_execution_report(_Report(*LISTINGS[102], OrderStatus.FILLED))
    assert _tick(s, 102, [(0.45, 1000)], hours_before=5.8), "a terminal report frees the market to top up"


def test_stake_is_respected_including_fees(make):
    s = make(p_series=0.40, stake_usd=50.0, max_match_usd=500.0)
    o = _tick(s, 102, [(0.45, 10_000)], hours_before=6)[0]
    shares, limit = o.take_size / S, o.take_limit_price / P
    assert shares * cost_per_share(limit, 0.07) <= 50.0 + 1e-6


def test_stake_already_spent_blocks_further_orders(make):
    s = make(p_series=0.40, stake_usd=50.0)
    s.positions.fill(LISTINGS[102], shares=105, price=0.45, fee=1.8)
    assert _tick(s, 102, [(0.45, 10_000)], hours_before=6) == []


def test_match_cap_spans_series_and_map1(make):
    s = make(p_series=0.40, p_map1=0.40, stake_usd=80.0, max_match_usd=100.0)
    s.positions.fill(LISTINGS[102], shares=170, price=0.45, fee=3.0)
    o = _tick(s, 202, [(0.30, 10_000)], hours_before=0.5)
    assert o, "map 1 can still trade with the match's remaining room"
    assert o[0].take_size / S * cost_per_share(o[0].take_limit_price / P, 0.07) <= 100.0 - (170 * 0.45 + 3.0) + 1e-6


def test_never_buys_the_other_side_once_holding_one(make):
    s = make(p_series=0.60)
    s.positions.fill(LISTINGS[102], shares=10, price=0.30)
    assert _tick(s, 101, [(0.50, 1000)], hours_before=6) == [], "holding team B; team A now looks cheap - no hedge"


def test_adverse_move_stops_top_ups(make):
    s = make(p_series=0.40, adverse_move=0.08)
    s.positions.fill(LISTINGS[102], shares=10, price=0.45)
    assert _tick(s, 102, [(0.36, 1000)], hours_before=6) == [], "price fell 9c below entry: likely news"
    assert _tick(s, 102, [(0.44, 1000)], hours_before=6) != []


def test_restart_reads_committed_money_from_positions(make):
    """A fresh instance must see an earlier session's fills and not buy them again."""
    first = make(p_series=0.40, stake_usd=50.0)
    first.positions.fill(LISTINGS[102], shares=110, price=0.45, fee=1.9)
    restarted = make(p_series=0.40, stake_usd=50.0)
    restarted._position_view = first.positions
    assert _tick(restarted, 102, [(0.45, 10_000)], hours_before=6) == []


def test_unranked_filter_is_opt_in(make):
    assert _tick(make(p_series=0.40, rank=0.0), 102, [(0.45, 1000)], hours_before=6)
    assert _tick(make(p_series=0.40, rank=0.0, skip_unranked_series=True), 102, [(0.45, 1000)], hours_before=6) == []


def test_market_without_a_prediction_never_trades(make):
    s = make(markets=[{**_market("series", 101, 102), "match_id": 999}])
    assert _tick(s, 102, [(0.01, 1000)], hours_before=6) == []


def test_unknown_listing_data_is_ignored(make):
    s = make()
    assert s.on_market_data(_Book((PM, 99999), [(0.01, 10)], K_NS - HOUR)) == []


# ---- market matching and config building --------------------------------------------


def _pm(rows):
    return polymarket.index_markets(pd.DataFrame([{"condition_id": f"0xc{i}", "closed": True, **r}
                                                  for i, r in enumerate(rows)]))


def _mk(slug, outcomes, tokens, final=("0", "0")):
    return {"market_slug": slug, "outcomes": list(outcomes), "tokens": list(tokens), "final": list(final)}


def test_match_market_reads_orientation_from_outcomes():
    pm = _pm([_mk("cs2-x-y-2026-10-05", ["Team Spirit", "NAVI"], ["tS", "tN"], ("0", "1"))])
    cond, a, b, why = polymarket.match_market(pm, "Natus Vincere", "Spirit", pd.Timestamp("2026-10-05").date(),
                                              "series", team_a_won=None)
    assert why == "no market", "NAVI vs Natus Vincere do not normalise to the same name"
    cond, a, b, why = polymarket.match_market(pm, "NAVI", "Team Spirit", pd.Timestamp("2026-10-05").date(),
                                              "series", team_a_won=1)
    assert why is None and (a, b) == ("tN", "tS") and cond == "0xc0"


def test_match_market_separates_series_and_map_markets_and_refuses_ambiguity():
    pm = _pm([_mk("cs2-a-b-2026-10-05", ["A", "B"], ["s1", "s2"]),
              _mk("cs2-a-b-2026-10-05-game1", ["A", "B"], ["g1", "g2"]),
              _mk("cs2-a-b-2026-10-05-game1-round-total-21pt5", ["Over", "Under"], ["o", "u"])])
    d = pd.Timestamp("2026-10-05").date()
    assert polymarket.match_market(pm, "A", "B", d, "game1")[1:3] == ("g1", "g2")
    assert polymarket.match_market(pm, "A", "B", d, "series")[1:3] == ("s1", "s2")
    twice = _pm([_mk("cs2-a-b-2026-10-05", ["A", "B"], ["s1", "s2"]),
                 _mk("cs2-a1-b1-2026-10-05", ["Team A", "B Esports"], ["s3", "s4"])])
    assert polymarket.match_market(twice, "A", "B", d, "series")[3] == "ambiguous", \
        "two same-day markets between the same teams must be refused, not guessed"


def test_match_market_rejects_a_settlement_that_disagrees_with_hltv():
    pm = _pm([_mk("cs2-a-b-2026-10-05", ["A", "B"], ["s1", "s2"], ("1", "0"))])
    assert polymarket.match_market(pm, "A", "B", pd.Timestamp("2026-10-05").date(), "series",
                                   team_a_won=0)[3] == "settlement disagrees with HLTV"


def test_build_scenarios_emits_listings_and_windows_per_match():
    history = pd.DataFrame([{"match_id": 7, "match_date": "2026-10-05", "match_time": KICKOFF,
                             "map_position_in_series": p, "team_a_name": "A", "team_b_name": "B",
                             "team_a_won": w} for p, w in ((1, 1), (2, 1))])
    preds = _preds([{"match_id": 7, "market": "series", "p_team_a": 0.6, "rank_known_both": 1.0},
                    {"match_id": 7, "market": "game1", "p_team_a": 0.6, "rank_known_both": 1.0}])
    pm = _pm([_mk("cs2-a-b-2026-10-05", ["A", "B"], ["s1", "s2"], ("1", "0")),
              _mk("cs2-a-b-2026-10-05-game1", ["B", "A"], ["g2", "g1"], ("0", "1"))])
    ids = {"s1": 101, "s2": 102, "g1": 201, "g2": 202}
    scenarios, skipped = make_config.build_scenarios(preds, history, pm, lambda cond, tok: ids.get(tok))
    sc = scenarios["m7"]
    assert not skipped
    assert [l["listing_id"] for l in sc["listings"]] == [101, 102, 201, 202]
    legs = {leg["market"]: leg for leg in sc["strategy_args"]["markets"]}
    assert (legs["game1"]["listing_team_a"], legs["game1"]["listing_team_b"]) == (201, 202), \
        "map-1 outcomes are listed B-first; team A must still map to A's token"
    assert pd.Timestamp(sc["start_date"]) == (KICKOFF - pd.Timedelta(hours=12, minutes=15)).tz_localize(None)


def test_build_scenarios_skips_unregistered_tokens():
    history = pd.DataFrame([{"match_id": 7, "match_date": "2026-10-05", "match_time": KICKOFF,
                             "map_position_in_series": 1, "team_a_name": "A", "team_b_name": "B", "team_a_won": 1}])
    preds = _preds([{"match_id": 7, "market": "series", "p_team_a": 0.6, "rank_known_both": 1.0}])
    pm = _pm([_mk("cs2-a-b-2026-10-05", ["A", "B"], ["s1", "s2"])])
    scenarios, skipped = make_config.build_scenarios(preds, history, pm, lambda cond, tok: None)
    assert scenarios == {} and skipped == {"series: token not in registry": 1}


def test_no_trade_before_the_prediction_was_written(make):
    written = KICKOFF - pd.Timedelta(hours=3)
    s = make(rows=[{"match_id": 7, "market": "series", "p_team_a": 0.40, "rank_known_both": 1.0,
                    "generated_at": written}])
    assert _tick(s, 102, [(0.45, 1000)], hours_before=6) == [], "a backtest must not trade on a future prediction"
    assert _tick(s, 102, [(0.45, 1000)], hours_before=2.9) != []


def test_rescheduled_kickoff_moves_the_window(make):
    """HLTV pushes the match back 3h; the pipeline writes a row with the new kickoff."""
    early = KICKOFF - pd.Timedelta(hours=20)
    s = make(rows=[
        {"match_id": 7, "market": "series", "p_team_a": 0.40, "rank_known_both": 1.0, "generated_at": early},
        {"match_id": 7, "market": "series", "p_team_a": 0.40, "rank_known_both": 1.0,
         "generated_at": early + pd.Timedelta(hours=1), "kickoff": KICKOFF + pd.Timedelta(hours=3)},
    ])
    assert _tick(s, 102, [(0.45, 1000)], hours_before=-1) != [], "past the original kickoff, but 2h before the new one"
    assert _tick(s, 102, [(0.45, 1000)], hours_before=-3.1) == [], "and closed at the new kickoff"


def test_reload_picks_up_new_predictions(make):
    s = make(p_series=0.50, reload_minutes=5)
    assert _tick(s, 102, [(0.45, 1000)], hours_before=6) == [], "no edge on the first prediction"
    _preds([{"match_id": 7, "market": "series", "p_team_a": 0.40, "rank_known_both": 1.0,
             "generated_at": KICKOFF - pd.Timedelta(hours=6)}]).to_parquet(s._test_path)
    assert _tick(s, 102, [(0.45, 1000)], hours_before=5.95) == [], "not reloaded within five minutes"
    assert _tick(s, 102, [(0.45, 1000)], hours_before=5.9) != [], "reloaded after five minutes"


def test_failed_reload_keeps_the_predictions_already_held(make):
    s = make(p_series=0.40, reload_minutes=5)
    _tick(s, 101, [(0.99, 1)], hours_before=7)
    s._test_path.write_bytes(b"not a parquet file")
    assert _tick(s, 102, [(0.45, 1000)], hours_before=6) != []
