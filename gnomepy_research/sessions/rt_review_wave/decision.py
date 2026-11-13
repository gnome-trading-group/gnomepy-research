"""
Pure order logic for the RT review-wave strategy: no engine, no registry, no JVM. The wave and the signal it trades
on are models/rt/signal.py.

Prices are plain probabilities in [0, 1] and sizes are contracts; the strategy converts to the engine's scaled integers
at the edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True)
class Take:
    contracts: float
    limit_price: float
    cost: float  # at the levels it reaches, fees included


def fee_per_contract(price: float, rate: float) -> float:
    return rate * price * (1.0 - price)


def take(asks: list[Level], limit: float, room_contracts: float, room_usd: float, fee_rate: float,
         lot: float) -> Take | None:
    """How many contracts to buy at asks up to limit: no more than the book offers there, room_contracts, or what
    room_usd pays for at each level's price plus its fee; whole lots only."""
    contracts = cost = 0.0
    for level in asks:
        if level.price > limit + 1e-12:
            break
        unit = level.price + fee_per_contract(level.price, fee_rate)
        n = min(level.size, room_contracts - contracts, (room_usd - cost) / unit)
        n = math.floor(n / lot + 1e-9) * lot if lot > 0 else n
        if n <= 0:
            break
        contracts += n
        cost += n * unit
        if contracts >= room_contracts - 1e-9:
            break
    return Take(contracts, limit, cost) if contracts > 0 else None
