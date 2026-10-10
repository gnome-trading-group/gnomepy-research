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
        from gnomepy_research.sessions.cross_prediction_arb import cross_prediction_arb as mod

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
            return mod.CrossPredictionArb(outcomes=outcomes, **defaults)
    return _make


@pytest.fixture
def strategy(make_strategy):
    return make_strategy()


@dataclass
class _Position:
    net_quantity: int


class FakePositions:
    """Stands in for the engine's position view."""

    def __init__(self, positions: dict | None = None):
        self._pos = dict(positions or {})

    def set(self, listing, qty):
        self._pos[listing] = qty

    def get_position(self, exchange_id, security_id):
        qty = self._pos.get((exchange_id, security_id))
        return _Position(net_quantity=qty) if qty is not None else None

    def get_effective_quantity(self, exchange_id, security_id):
        return self._pos.get((exchange_id, security_id), 0)


@pytest.fixture
def positions():
    return FakePositions()
