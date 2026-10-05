from __future__ import annotations

from types import SimpleNamespace

import pytest
from gnomepy import OrderType, Side

from gnomepy_research.strategies import random_trader
from gnomepy_research.strategies.random_trader import RandomTrader

SECOND = 1_000_000_000
DOLLAR = 1_000_000_000
CENT = DOLLAR // 100


class FakePositions:
    def __init__(self, min_size: int = 1):
        self.quantity: dict[tuple[int, int], int] = {}
        self.min_size = min_size

    def get_effective_quantity(self, exchange_id, security_id):
        return self.quantity.get((exchange_id, security_id), 0)

    def compliant_size(self, exchange_id, security_id, desired_size, price):
        return max(desired_size, self.min_size)


class Book:
    def __init__(self, ts: int, exchange_id: int = 1, security_id: int = 2, bid: int = 100, ask: int = 101):
        self.event_timestamp = ts
        self.exchange_id = exchange_id
        self.security_id = security_id
        self._bid, self._ask = bid, ask

    def bid_price(self, level):
        return self._bid

    def ask_price(self, level):
        return self._ask


@pytest.fixture(autouse=True)
def fake_intent(monkeypatch):
    # The real Intent wraps a Java object and the real Scales reads Java statics; both need a running JVM.
    monkeypatch.setattr(random_trader, "Intent", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(random_trader, "Scales", SimpleNamespace(PRICE=DOLLAR))


def make(positions: FakePositions, **kwargs) -> RandomTrader:
    strategy = RandomTrader(seed=7, **kwargs)
    strategy._position_view = positions
    return strategy


def test_trades_once_per_interval():
    strategy = make(FakePositions(), interval_seconds=5)
    fired = [bool(strategy.on_market_data(Book(t * SECOND))) for t in range(0, 16)]
    assert [t for t, f in enumerate(fired) if f] == [0, 5, 10, 15]


def test_skips_empty_books():
    strategy = make(FakePositions())
    assert strategy.on_market_data(Book(0, bid=0)) == []
    assert strategy.on_market_data(Book(0))


def test_paces_each_listing_independently():
    strategy = make(FakePositions(), interval_seconds=5)
    [first] = strategy.on_market_data(Book(0, security_id=2))
    [second] = strategy.on_market_data(Book(SECOND, security_id=3))
    assert (first.security_id, second.security_id) == (2, 3)
    assert strategy.on_market_data(Book(2 * SECOND, security_id=2)) == []


def test_never_exceeds_max_position():
    positions = FakePositions()
    strategy = make(positions, interval_seconds=1, max_position=2)
    for t in range(200):
        for intent in strategy.on_market_data(Book(t * SECOND)):
            key = (intent.exchange_id, intent.security_id)
            delta = intent.take_size if intent.take_side == Side.BID else -intent.take_size
            positions.quantity[key] = positions.quantity.get(key, 0) + delta
            assert abs(positions.quantity[key]) <= 2


def test_rounds_up_to_listing_minimum():
    strategy = make(FakePositions(min_size=10), max_position=50)
    [intent] = strategy.on_market_data(Book(0))
    assert intent.take_size == 10


def test_takes_are_market_orders_by_default():
    [intent] = make(FakePositions()).on_market_data(Book(0))
    assert intent.take_order_type == OrderType.MARKET
    assert not hasattr(intent, "bid_size")


def test_limit_takes_are_priced_through_the_touch():
    strategy = make(FakePositions(), take_type="limit", take_through=0.20)
    book = Book(0, bid=40 * CENT, ask=42 * CENT)
    [intent] = strategy.on_market_data(book)
    assert intent.take_order_type == OrderType.LIMIT
    expected = 62 * CENT if intent.take_side == Side.BID else 20 * CENT
    assert intent.take_limit_price == expected


def test_quote_only_rests_both_sides_outside_the_touch_without_taking():
    strategy = make(FakePositions(min_size=5), trade=False, quote=True, quote_offset=0.02)
    [intent] = strategy.on_market_data(Book(0, bid=40 * CENT, ask=42 * CENT))
    assert (intent.bid_price, intent.ask_price) == (38 * CENT, 44 * CENT)
    assert (intent.bid_size, intent.ask_size) == (5, 5)
    assert not hasattr(intent, "take_side")


def test_quote_and_trade_together_send_one_intent_with_both():
    strategy = make(FakePositions(), quote=True)
    [intent] = strategy.on_market_data(Book(0, bid=40 * CENT, ask=42 * CENT))
    assert intent.bid_size and intent.take_size


def test_a_quote_that_would_go_below_zero_is_skipped():
    strategy = make(FakePositions(), trade=False, quote=True, quote_offset=0.05)
    assert strategy.on_market_data(Book(0, bid=2 * CENT, ask=4 * CENT)) == []


def test_nothing_enabled_sends_nothing():
    assert make(FakePositions(), trade=False).on_market_data(Book(0)) == []


def test_rejects_an_unknown_take_type():
    with pytest.raises(ValueError):
        RandomTrader(take_type="stop")


def test_execution_reports_are_logged_with_their_reject_reason(capsys):
    report = SimpleNamespace(
        exchange_id=1, security_id=2, client_oid="17", exec_type=SimpleNamespace(name="REJECT"),
        filled_qty=0, fill_price=0, leaves_qty=0, reject_reason=SimpleNamespace(name="RISK_LIMIT"),
    )
    assert make(FakePositions()).on_execution_report(report) == []
    line = capsys.readouterr().out
    assert "order 17 REJECT" in line and "reason=RISK_LIMIT" in line
