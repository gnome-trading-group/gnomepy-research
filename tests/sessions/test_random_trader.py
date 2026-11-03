from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from gnomepy import ExecType, OrderType, RejectReason, Side

from gnomepy_research.sessions.random_trader import strategy as random_trader
from gnomepy_research.sessions.random_trader.strategy import RandomTrader

SECOND = 1_000_000_000
DOLLAR = 1_000_000_000
CENT = DOLLAR // 100


class FakePositions:
    def __init__(self, min_size: int = 1, tick: int = 1):
        self.quantity: dict[tuple[int, int], int] = {}
        self.min_size = min_size
        self.tick = tick

    def compliant_price(self, exchange_id, security_id, price, side):
        return price // self.tick * self.tick if side == Side.BID else -(-price // self.tick) * self.tick

    def get_effective_quantity(self, exchange_id, security_id):
        return self.quantity.get((exchange_id, security_id), 0)

    def compliant_size(self, exchange_id, security_id, desired_size, price):
        return max(desired_size, self.min_size)


class Book:
    def __init__(self, ts: int, exchange_id: int = 1, security_id: int = 2, bid: int = 100, ask: int = 101):
        self.event_timestamp = ts
        self.stream_id = FakeStreams.stream_of(exchange_id, security_id)
        self._bid, self._ask = bid, ask

    def bid_price(self, level):
        return self._bid

    def ask_price(self, level):
        return self._ask


class FakeListing:
    def __init__(self, exchange_id: int, security_id: int):
        self._exchange = SimpleNamespace(exchangeId=lambda: exchange_id)
        self._security = SimpleNamespace(securityId=lambda: security_id)

    def exchange(self):
        return self._exchange

    def security(self):
        return self._security


class FakeStreams:
    """The session's stream directory: every (exchange, security) a book names has a stream and a listing."""

    @staticmethod
    def stream_of(exchange_id: int, security_id: int) -> int:
        return exchange_id * 1_000 + security_id

    def listing_id(self, stream_id: int) -> int:
        return stream_id

    def listing(self, listing_id: int) -> FakeListing:
        return FakeListing(listing_id // 1_000, listing_id % 1_000)


@pytest.fixture(autouse=True)
def fake_intent(monkeypatch):
    # The real Intent wraps a Java object and the real Scales reads Java statics; both need a running JVM.
    monkeypatch.setattr(random_trader, "Intent", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(random_trader, "Scales", SimpleNamespace(PRICE=DOLLAR))


def make(positions: FakePositions, **kwargs) -> RandomTrader:
    strategy = RandomTrader(seed=7, **kwargs)
    strategy._position_view = positions
    strategy._streams = FakeStreams()
    return strategy


def test_trades_once_per_interval():
    strategy = make(FakePositions(), interval_seconds=5)
    fired = [bool(strategy.on_market_data(Book(t * SECOND))) for t in range(0, 16)]
    assert [t for t, f in enumerate(fired) if f] == [0, 5, 10, 15]


def test_skips_a_stream_with_no_listing():
    strategy = make(FakePositions())
    strategy._streams = SimpleNamespace(listing_id=lambda stream_id: None)
    assert strategy.on_market_data(Book(0)) == []


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


def test_quotes_and_limit_takes_land_on_the_listing_tick():
    tick = CENT
    book = Book(0, bid=40 * CENT + 3_000_000, ask=42 * CENT + 3_000_000)  # an off-tick book
    [quote] = make(FakePositions(tick=tick), trade=False, quote=True, quote_offset=0.02).on_market_data(book)
    assert quote.bid_price % tick == 0 and quote.ask_price % tick == 0
    assert (quote.bid_price, quote.ask_price) == (38 * CENT, 45 * CENT)

    [take] = make(FakePositions(tick=tick), take_type="limit", take_through=0.20).on_market_data(book)
    assert take.take_limit_price % tick == 0


class FakeControl:
    def __init__(self):
        self.exits = []

    def requestExit(self, reason, flatten, timeout_millis):
        self.exits.append((reason, flatten, timeout_millis))
        return len(self.exits) == 1

    def isExiting(self):
        return bool(self.exits)


def test_exits_after_the_configured_time_on_exchange_time():
    strategy = make(FakePositions(), exit_after_seconds=30, exit_flatten=True, exit_timeout_seconds=5)
    strategy._control = FakeControl()

    assert strategy.on_market_data(Book(100 * SECOND))
    strategy.on_market_data(Book(129 * SECOND))
    assert strategy._control.exits == []

    assert strategy.on_market_data(Book(130 * SECOND)) == []
    assert strategy._control.exits == [("RandomTrader exit test after 30s", True, 5_000)]


def test_never_exits_by_default():
    strategy = make(FakePositions())
    strategy._control = FakeControl()
    for t in range(0, 10_000, 100):
        strategy.on_market_data(Book(t * SECOND))
    assert strategy._control.exits == []


class FakeStream:
    """Keeps committed rows, holding the strategy to the declare, freeze, begin, set, commit order sessions enforce."""

    def __init__(self):
        self.columns = []
        self.rows = []
        self._frozen = False
        self._row = None

    def _add(self, name):
        assert not self._frozen
        self.columns.append(name)
        return len(self.columns) - 1

    add_long_column = add_double_column = add_int_column = _add

    def freeze(self):
        self._frozen = True

    def begin(self):
        assert self._frozen and self._row is None
        self._row = {}

    def _set(self, column, value):
        assert self._row is not None
        self._row[self.columns[column]] = value

    set_long = set_double = set_int = _set

    def commit(self):
        self.rows.append(self._row)
        self._row = None


class FakeMetrics:
    def __init__(self):
        self.streams = {}

    def create_stream(self, name):
        self.streams[name] = FakeStream()
        return self.streams[name]


def with_metrics(strategy: RandomTrader) -> FakeMetrics:
    metrics = FakeMetrics()
    strategy._metric_recorder = metrics
    strategy.register_metrics()
    return metrics


def test_each_intent_records_the_touch_the_position_and_what_was_sent():
    positions = FakePositions()
    positions.quantity[(1, 2)] = 2
    strategy = make(positions, quote=True, quote_offset=0.01)
    metrics = with_metrics(strategy)

    intents = strategy.on_market_data(Book(0, bid=40 * CENT, ask=42 * CENT))

    [row] = metrics.streams["decisions"].rows
    assert row["timestamp"] == 0
    assert (row["exchange_id"], row["security_id"]) == (1, 2)
    assert (row["best_bid"], row["best_ask"]) == (0.40, 0.42)
    assert row["position"] == 2
    assert row["take_side"] == {Side.BID: 1, Side.ASK: -1}[intents[0].take_side]
    assert row["take_size"] == intents[0].take_size
    assert (row["quote_bid"], row["quote_ask"]) == (0.39, 0.43)


def test_a_take_without_a_quote_records_no_quote_prices():
    strategy = make(FakePositions())
    metrics = with_metrics(strategy)

    strategy.on_market_data(Book(0))

    [row] = metrics.streams["decisions"].rows
    assert math.isnan(row["quote_bid"]) and math.isnan(row["quote_ask"])


def test_no_intent_records_no_decision():
    strategy = make(FakePositions())
    metrics = with_metrics(strategy)

    strategy.on_market_data(Book(0, bid=0))

    assert metrics.streams["decisions"].rows == []


def test_each_execution_report_is_recorded_with_its_type_and_reject_reason():
    strategy = make(FakePositions())
    metrics = with_metrics(strategy)
    fill = SimpleNamespace(
        timestamp_recv=5, exchange_id=1, security_id=2, client_oid="17", exec_type=ExecType.FILL,
        filled_qty=1, fill_price=41 * CENT, leaves_qty=0, reject_reason=RejectReason.UNKNOWN,
    )
    reject = SimpleNamespace(
        timestamp_recv=6, exchange_id=1, security_id=2, client_oid="18", exec_type=ExecType.REJECT,
        filled_qty=0, fill_price=0, leaves_qty=0, reject_reason=RejectReason.RISK_LIMIT_EXCEEDED,
    )

    strategy.on_execution_report(fill)
    strategy.on_execution_report(reject)

    filled, rejected = metrics.streams["reports"].rows
    assert filled == {
        "timestamp": 5, "exchange_id": 1, "security_id": 2, "client_oid": 17,
        "exec_type": list(ExecType).index(ExecType.FILL), "filled_qty": 1, "fill_price": 0.41,
        "leaves_qty": 0, "reject_reason": -1,
    }
    assert rejected["exec_type"] == list(ExecType).index(ExecType.REJECT)
    assert rejected["reject_reason"] == list(RejectReason).index(RejectReason.RISK_LIMIT_EXCEEDED)
