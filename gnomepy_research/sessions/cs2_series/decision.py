"""
Pure entry logic for the CS2 series strategy: no engine, no registry, no JVM.

On every venue each team's outcome is its own listing - a Polymarket token, a Kalshi
team market, a Polymarket US long or short side - so taking a view on team B means
buying team B's listing, never selling team A's. Each side is therefore priced off its
own ask book, and an entry is only ever a taker buy.

An entry may reach as deep into the ask book as the edge allows: every share it buys, at its own
level's price and fee, still clears min_edge. Fees are the venues' rate * p * (1 - p) curve, charged
on each fill at its own price; the venues' rounding (Polymarket to $0.00001 per fill, Kalshi up to
$0.0001 per order) is left out.

Prices here are plain probabilities in [0, 1] and sizes are shares; the strategy
converts to the engine's scaled integers at the edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True)
class Entry:
    side: int            # 0 = team A's listing, 1 = team B's
    fair: float          # probability this side wins
    best_ask: float
    edge_at_best: float  # fair - best_ask - fee(best_ask)
    limit_price: float   # highest price that still clears min_edge, on tick
    levels: tuple[Level, ...]  # the asks at or below limit_price, cheapest first

    @property
    def shares_available(self) -> float:
        return sum(level.size for level in self.levels)


@dataclass(frozen=True)
class Take:
    shares: float
    limit_price: float   # the deepest level the shares reach: the order can't fill above what was costed
    cost: float          # what the shares cost at their levels, fees included


def taker_fee(price: float, rate: float) -> float:
    """The venues' parametric taker fee per share, rate * p * (1 - p), before each venue's cent rounding."""
    return rate * price * (1.0 - price)


def max_entry_price(fair: float, min_edge: float, fee_rate: float) -> float:
    """
    Highest price p with fair - p - fee(p) >= min_edge.

    fair - p - r*p*(1-p) is decreasing in p on [0, 1] for r < 1, so the
    condition holds for every p up to the smaller root of
    r*p^2 - (1+r)*p + (fair - min_edge) = 0.
    """
    target = fair - min_edge
    if target <= 0:
        return 0.0
    if fee_rate == 0:
        return min(target, 1.0)
    a, b, c = fee_rate, -(1.0 + fee_rate), target
    disc = b * b - 4 * a * c
    if disc < 0:
        return 1.0
    return max(0.0, min(1.0, (-b - math.sqrt(disc)) / (2 * a)))


def _floor_to_tick(price: float, tick: float) -> float:
    return math.floor(price / tick + 1e-9) * tick


def evaluate_side(
    side: int,
    fair: float,
    asks: list[Level],
    *,
    min_edge: float,
    fee_rate: float,
    tick: float,
) -> Entry | None:
    """An entry on one side if its best ask clears min_edge after the fee, with every ask that still does."""
    if not asks:
        return None
    best = asks[0].price
    edge_at_best = fair - best - taker_fee(best, fee_rate)
    if edge_at_best < min_edge:
        return None
    limit = _floor_to_tick(max_entry_price(fair, min_edge, fee_rate), tick)
    if limit < best - 1e-12:
        return None
    levels = tuple(level for level in asks if level.price <= limit + 1e-12)
    return Entry(side=side, fair=fair, best_ask=best, edge_at_best=edge_at_best, limit_price=limit, levels=levels)


def choose_entry(
    fair_team_a: float,
    asks_team_a: list[Level],
    asks_team_b: list[Level],
    *,
    min_edge: float,
    fee_rates: tuple[float, float],
    tick: float,
    allowed_sides: tuple[int, ...] = (0, 1),
) -> Entry | None:
    """The better of the two sides, if either clears the threshold; fee_rates are each side's listing's taker rate."""
    candidates = []
    for side, fair, asks in ((0, fair_team_a, asks_team_a), (1, 1.0 - fair_team_a, asks_team_b)):
        if side not in allowed_sides:
            continue
        entry = evaluate_side(side, fair, asks, min_edge=min_edge, fee_rate=fee_rates[side], tick=tick)
        if entry is not None:
            candidates.append(entry)
    return max(candidates, key=lambda e: e.edge_at_best) if candidates else None


def cost_per_share(price: float, fee_rate: float) -> float:
    return price + taker_fee(price, fee_rate)


def plan_take(entry: Entry, budget_usd: float, fee_rate: float, lot: float) -> Take:
    """
    Walk the entry's asks cheapest first, each level's shares costed at that level's price and fee, until the
    budget or the levels run out. Shares are whole lots; the limit is the deepest level they reach.
    """
    shares, remaining = 0.0, budget_usd
    for level in entry.levels:
        unit = cost_per_share(level.price, fee_rate)
        if remaining < unit * lot:
            break
        take = min(level.size, remaining / unit)
        shares += take
        remaining -= take * unit
    shares = math.floor(shares / lot + 1e-9) * lot
    return _cost(entry.levels, shares, fee_rate)


def _cost(levels: tuple[Level, ...], shares: float, fee_rate: float) -> Take:
    cost, left, limit = 0.0, shares, 0.0
    for level in levels:
        if left <= 1e-12:
            break
        take = min(level.size, left)
        cost += take * cost_per_share(level.price, fee_rate)
        left -= take
        limit = level.price
    return Take(shares=shares - max(left, 0.0), limit_price=limit, cost=cost)
