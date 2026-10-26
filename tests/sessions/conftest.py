"""Shared fixtures for cross_prediction_arb tests.

CrossPredictionArb.__init__ builds a RegistryClient and resolves every listing over
the network, so tests patch it. Nothing else in the strategy needs a JVM.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from gnomepy import Scales

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
CENT = PRICE_SCALE // 100

# Polymarket is exchange 4, Kalshi 5 — matching the real registry.
PM, K = 4, 5

# listing_id -> (exchange_id, security_id)
LISTINGS = {
    222852: (PM, 223652),   # PM outcome A
    222853: (PM, 223653),   # PM outcome B
    97203: (K, 97984),      # Kalshi outcome A
    97202: (K, 97983),      # Kalshi outcome B
    900001: (PM, 900101),   # third outcome, PM  (N=3 cases)
    900002: (K, 900102),    # third outcome, Kalshi
}


@dataclass
class _Listing:
    exchange_id: int
    security_id: int


# A listing's book stream id in the fakes below: distinct from every listing id, so a mix-up can't pass.
STREAM_OFFSET = 1_000_000


def stream_of(listing_id: int) -> int:
    return STREAM_OFFSET + listing_id


@dataclass(frozen=True)
class _Id:
    value: int

    def exchangeId(self) -> int:
        return self.value

    def securityId(self) -> int:
        return self.value


@dataclass(frozen=True)
class _JavaListing:
    """The registry's Java Listing, as gnomepy's StreamDirectory.listing returns it."""

    listing_id: int
    exchange_id: int
    security_id: int

    def listingId(self) -> int:
        return self.listing_id

    def exchange(self) -> _Id:
        return _Id(self.exchange_id)

    def security(self) -> _Id:
        return _Id(self.security_id)


class FakeStreams:
    """Stands in for gnomepy's StreamDirectory over listings {listing_id: (exchange_id, security_id)}; each listing's
    book stream is stream_of(listing_id)."""

    def __init__(self, listings: dict[int, tuple[int, int]]):
        self._listings = dict(listings)

    def listing_id(self, stream_id: int) -> int | None:
        listing_id = stream_id - STREAM_OFFSET
        return listing_id if listing_id in self._listings else None

    def listing_id_for(self, exchange_id: int, security_id: int) -> int | None:
        for listing_id, venue in self._listings.items():
            if venue == (exchange_id, security_id):
                return listing_id
        return None

    def stream_id(self, listing_id: int) -> int | None:
        return stream_of(listing_id) if listing_id in self._listings else None

    def listing(self, listing_id: int) -> _JavaListing | None:
        venue = self._listings.get(listing_id)
        return _JavaListing(listing_id, *venue) if venue else None


@dataclass
class _Spec:
    min_notional: int
    lot_size: int
    tick_size: int
    min_size: int = 0


class _FakeRegistry:
    """Stands in for RegistryClient.

    tick_size defaults to the value the real registry currently returns for Kalshi
    ($0.001), which the market data shows is wrong — Kalshi trades in whole cents.
    Tests rely on that being the input so the strategy's own guard is what's exercised.
    """

    def __init__(self, tick_overrides: dict[int, int] | None = None,
                 min_notional: int = 0, lot_size: int = SIZE_SCALE, min_size: int = 0):
        self._ticks = tick_overrides or {}
        self._min_notional = min_notional
        self._lot_size = lot_size
        self._min_size = min_size

    def get_listing(self, *, listing_id: int, **_):
        eid, sid = LISTINGS[listing_id]
        return [_Listing(exchange_id=eid, security_id=sid)]

    def get_listing_spec(self, *, listing_id: int, **_):
        eid, _sid = LISTINGS[listing_id]
        default_tick = PRICE_SCALE // 1000 if eid == K else CENT
        return [_Spec(
            min_notional=self._min_notional,
            lot_size=self._lot_size,
            tick_size=self._ticks.get(listing_id, default_tick),
            min_size=self._min_size,
        )]


@pytest.fixture
def make_strategy():
    """Build a CrossPredictionArb with the registry patched out."""
    def _make(outcomes=None, registry: _FakeRegistry | None = None, **kwargs):
        from gnomepy_research.sessions.cross_prediction_arb import strategy as mod

        if outcomes is None:
            outcomes = [{"pm": 222852, "k": 97203}, {"pm": 222853, "k": 97202}]
        defaults = dict(
            max_position=2000,
            min_edge_cents=1.0,
            min_contract_price=0.10,
            max_price_divergence_cents=5.0,
            imbalance_timeout_ns=60_000_000_000,
            taker_labels=["pm"],
            dutch_book_labels=[],
        )
        defaults.update(kwargs)
        with patch.object(mod, "RegistryClient", lambda *a, **k: registry or _FakeRegistry()):
            strategy = mod.CrossPredictionArb(outcomes=outcomes, **defaults)
        strategy._position_view = FakePositions()
        strategy._streams = FakeStreams(LISTINGS)
        return strategy
    return _make


@pytest.fixture
def strategy(make_strategy):
    return make_strategy()


@dataclass
class _Position:
    net_quantity: int


# The rate x p(1-p) curve rates the venues charged before the registry held their schedules.
FEE_RATES = {PM: (0.07, 0.0), K: (0.07, 0.0175)}


@dataclass(frozen=True)
class FakeFeeSchedule:
    """Stands in for gnomepy.fees.FeeSchedule, which the installed gnomepy predates."""

    taker: float
    maker: float

    def price_curve_rates(self) -> tuple[float, float]:
        return self.taker, self.maker


class FakePositions:
    """Stands in for the engine's position view."""

    def __init__(self, positions: dict | None = None, fee_rates: dict[int, tuple[float, float]] | None = None):
        self._pos = dict(positions or {})
        self._fee_rates = FEE_RATES if fee_rates is None else fee_rates

    def fee_schedule(self, exchange_id, security_id):
        rates = self._fee_rates.get(exchange_id)
        return FakeFeeSchedule(*rates) if rates is not None else None

    def set(self, listing, qty):
        """``listing`` is a venue tuple or a Listing record, which the engine knows by its venue ids too."""
        if hasattr(listing, "exchange_id"):
            listing = (listing.exchange_id, listing.security_id)
        self._pos[listing] = qty

    def get_position(self, exchange_id, security_id):
        qty = self._pos.get((exchange_id, security_id))
        return _Position(net_quantity=qty) if qty is not None else None

    def get_effective_quantity(self, exchange_id, security_id):
        return self._pos.get((exchange_id, security_id), 0)


@pytest.fixture
def positions():
    return FakePositions()
