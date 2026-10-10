"""
Pure entry logic for the pre-match strategy: no engine, no registry, no JVM.

Polymarket outcomes are separate tokens, so taking a view on team B means buying
team B's token, never selling team A's. Each side is therefore priced off its own
ask book, and an entry is only ever a taker buy.

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
    side: int            # 0 = team A token, 1 = team B token
    fair: float          # probability this side wins
    best_ask: float
    edge_at_best: float  # fair - best_ask - fee(best_ask)
    limit_price: float   # highest price we will pay, on tick
    shares_available: float


def taker_fee(price: float, rate: float) -> float:
    """Polymarket's parametric taker fee per share: rate * p * (1 - p)."""
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
    max_slippage: float,
    tick: float,
) -> Entry | None:
    """An entry on one side if its best ask clears min_edge after the fee."""
    if not asks:
        return None
    best = asks[0].price
    edge_at_best = fair - best - taker_fee(best, fee_rate)
    if edge_at_best < min_edge:
        return None
    cap = max_entry_price(fair, min_edge, fee_rate)
    limit = _floor_to_tick(min(best + max_slippage, cap), tick)
    if limit < best - 1e-12:
        return None
    available = sum(level.size for level in asks if level.price <= limit + 1e-12)
    return Entry(side=side, fair=fair, best_ask=best, edge_at_best=edge_at_best,
                 limit_price=limit, shares_available=available)


def choose_entry(
    fair_team_a: float,
    asks_team_a: list[Level],
    asks_team_b: list[Level],
    *,
    min_edge: float,
    fee_rate: float,
    max_slippage: float,
    tick: float,
    allowed_sides: tuple[int, ...] = (0, 1),
) -> Entry | None:
    """The better of the two sides, if either clears the threshold."""
    candidates = []
    for side, fair, asks in ((0, fair_team_a, asks_team_a), (1, 1.0 - fair_team_a, asks_team_b)):
        if side not in allowed_sides:
            continue
        entry = evaluate_side(side, fair, asks, min_edge=min_edge, fee_rate=fee_rate,
                              max_slippage=max_slippage, tick=tick)
        if entry is not None:
            candidates.append(entry)
    return max(candidates, key=lambda e: e.edge_at_best) if candidates else None


def cost_per_share(price: float, fee_rate: float) -> float:
    return price + taker_fee(price, fee_rate)


def shares_to_buy(entry: Entry, budget_usd: float, fee_rate: float) -> float:
    """Shares affordable within budget at the limit price including the fee, capped by the book."""
    unit = cost_per_share(entry.limit_price, fee_rate)
    if budget_usd <= 0 or unit <= 0:
        return 0.0
    return max(0.0, min(entry.shares_available, budget_usd / unit))
