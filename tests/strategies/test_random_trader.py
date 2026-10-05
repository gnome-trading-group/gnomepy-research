from __future__ import annotations

from types import SimpleNamespace

import pytest
from gnomepy import Side

from gnomepy_research.strategies import random_trader
from gnomepy_research.strategies.random_trader import RandomTrader

SECOND = 1_000_000_000


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
    # The real Intent wraps a Java object and needs a running JVM.
    monkeypatch.setattr(random_trader, "Intent", lambda **kw: SimpleNamespace(**kw))


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
