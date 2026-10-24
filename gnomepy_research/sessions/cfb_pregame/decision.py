"""
Pure entry logic for the CFB pregame strategy: no engine, no registry, no JVM.

A copy of sessions/cs2_series/decision.py (sessions do not import each other); keep the two in step until the
maths moves somewhere shared.

On every venue each team's outcome is its own listing - a Polymarket token, a Kalshi
team market, a Polymarket US long or short side - so taking a view on team B means
buying team B's listing, never selling team A's. Each side is therefore priced off its
own ask book, and an entry is only ever a taker buy.

Only prices inside a band are bought. At the low end the model overrates long shots and at the high end it has
no edge over a market that already agrees (Mar-Sep 2026 replays), so a side whose best ask is outside the band
is skipped and no share is bought at or above its top.

An entry may reach as deep into the ask book as the edge allows: every share it buys, at its own
level's price and fee, still clears min_edge. How many it buys is the Kelly size: a share costing u
(its price plus fee) pays 1 if the side wins, so log-wealth growth is maximised by holding the n
shares at which p * (1 - u) * (W - C) = (1 - p) * u * (W - C + n), for win probability p, Kelly
bankroll W and C already paid. Sizing against what is already held makes each top-up bring the
holding to the size the current price and book justify, never past it. Fees are the venues'
rate * p * (1 - p) curve, charged on each fill at its own price; the venues' rounding (Polymarket to
$0.00001 per fill, Kalshi up to $0.0001 per order) is left out.

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


def _tick_below(price: float, tick: float) -> float:
    return (math.ceil(price / tick - 1e-9) - 1) * tick


def evaluate_side(
    side: int,
    fair: float,
    asks: list[Level],
    *,
    min_edge: float,
    fee_rate: float,
    tick: float,
    min_price: float = 0.0,
    max_price: float = 1.0,
) -> Entry | None:
    """
    An entry on one side if its best ask clears min_edge after the fee and lies in [min_price, max_price), with
    every ask that still clears it below max_price.
    """
    if not asks:
        return None
    best = asks[0].price
    if best < min_price - 1e-12 or best > max_price - 1e-12:
        return None
    edge_at_best = fair - best - taker_fee(best, fee_rate)
    if edge_at_best < min_edge:
        return None
    limit = min(_floor_to_tick(max_entry_price(fair, min_edge, fee_rate), tick), _tick_below(max_price, tick))
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
    min_price: float = 0.0,
    max_price: float = 1.0,
) -> Entry | None:
    """The better of the two sides, if either clears the threshold inside the price band; fee_rates are each side's
    listing's taker rate."""
    candidates = []
    for side, fair, asks in ((0, fair_team_a, asks_team_a), (1, 1.0 - fair_team_a, asks_team_b)):
        if side not in allowed_sides:
            continue
        entry = evaluate_side(side, fair, asks, min_edge=min_edge, fee_rate=fee_rates[side], tick=tick,
                              min_price=min_price, max_price=max_price)
        if entry is not None:
            candidates.append(entry)
    return max(candidates, key=lambda e: e.edge_at_best) if candidates else None


def cost_per_share(price: float, fee_rate: float) -> float:
    return price + taker_fee(price, fee_rate)


def kelly_shares(p: float, unit: float, kelly_bankroll: float, held_shares: float, held_cost: float) -> float:
    """
    Shares to add at a cost of `unit` each (price plus fee, paying 1 on a win) to reach the Kelly holding, given
    held_shares already bought for held_cost; zero or less when the holding is already there.

    Maximising p * log(W - C + n) + (1 - p) * log(W - C) over x more shares, with C and n growing by x * unit and
    x, gives x = (p * (1 - u) * (W - C) - (1 - p) * u * (W - C + n)) / (u * (1 - u)). From nothing that spends
    (p - u) / (1 - u) of W: the Kelly fraction for a contract that costs u and pays 1.
    """
    wealth = kelly_bankroll - held_cost
    if wealth <= 0 or not 0 < unit < 1:
        return 0.0
    return (p * (1 - unit) * wealth - (1 - p) * unit * (wealth + held_shares)) / (unit * (1 - unit))


def kelly_take(entry: Entry, kelly_bankroll: float, held_shares: float, held_cost: float, max_cost: float,
               fee_rate: float, lot: float) -> Take:
    """
    Walk the entry's asks cheapest first, each level's shares costed at its own price and fee, adding shares while
    the Kelly holding at that level is still above what is held, until the holding's cost would pass max_cost or the
    levels run out. Shares are whole lots; the limit is the deepest level they reach.
    """
    shares, n, cost = 0.0, held_shares, held_cost
    for level in entry.levels:
        unit = cost_per_share(level.price, fee_rate)
        take = min(level.size, kelly_shares(entry.fair, unit, kelly_bankroll, n, cost), (max_cost - cost) / unit)
        if take <= 0:
            break
        shares, n, cost = shares + take, n + take, cost + take * unit
        if take < level.size:
            break
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
