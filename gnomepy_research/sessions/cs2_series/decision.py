"""
Pure entry and exit logic for the CS2 series strategy: no engine, no registry, no JVM.

On every venue each team's outcome is its own listing - a Polymarket token, a Kalshi
team market, a Polymarket US long or short side - so taking a view on team B means
buying team B's listing, never selling team A's. Each listing is priced off its own book:
buys walk its asks, sales of what it holds walk its bids.

An entry may reach as deep into the ask book as the edge allows: every share it buys, at its own
level's price and fee, still clears min_edge. How many it buys is the Kelly size: a share costing u
(its price plus fee) pays 1 if the side wins, so log-wealth growth is maximised by holding the n
shares at which p * (1 - u) * (W - C) = (1 - p) * u * (W - C + n), for win probability p, Kelly
bankroll W and C already paid. Sizing against what is already held makes each top-up bring the
holding to the size the current price and book justify, never past it. Fees are the venues'
rate * p * (1 - p) curve, charged on each fill at its own price; the venues' rounding (Polymarket to
$0.00001 per fill, Kalshi up to $0.0001 per order) is left out.

A sale is taken only where a share's proceeds after the fee beat its fair value by min_edge, the same margin a buy
needs. At such a price the Kelly holding is none at all (a contract sold above what it is worth is better not held),
so a sale takes every held share those bids can absorb.

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
    fair: float          # probability this listing's team wins
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


@dataclass(frozen=True)
class Sale:
    shares: float
    best_bid: float
    edge_at_best: float  # best_bid - fee(best_bid) - fair
    limit_price: float   # the lowest bid the shares reach: the order can't fill below what was priced
    proceeds: float      # what the shares fetch at their levels, after fees


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


def _floor_to_lot(shares: float, lot: float) -> float:
    return math.floor(shares / lot + 1e-9) * lot


def evaluate_side(fair: float, asks: list[Level], *, min_edge: float, fee_rate: float, tick: float) -> Entry | None:
    """An entry on one listing if its best ask clears min_edge after the fee, with every ask that still clears it."""
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
    return Entry(fair=fair, best_ask=best, edge_at_best=edge_at_best, limit_price=limit, levels=levels)


def cost_per_share(price: float, fee_rate: float) -> float:
    return price + taker_fee(price, fee_rate)


def proceeds_per_share(price: float, fee_rate: float) -> float:
    return price - taker_fee(price, fee_rate)


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
    return _cost(entry.levels, _floor_to_lot(shares, lot), fee_rate)


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


def sell_down(fair: float, bids: list[Level], held_shares: float, *, min_edge: float, fee_rate: float,
              lot: float) -> Sale | None:
    """
    Sell held shares into every bid, best first, whose proceeds after the fee beat fair by min_edge; whole lots, never
    more than is held. None when the best bid doesn't clear it or nothing is held.
    """
    if not bids or held_shares <= 0:
        return None
    best = bids[0].price
    edge_at_best = proceeds_per_share(best, fee_rate) - fair
    if edge_at_best < min_edge:
        return None
    levels = [level for level in bids if proceeds_per_share(level.price, fee_rate) - fair >= min_edge - 1e-12]
    shares = _floor_to_lot(min(held_shares, sum(level.size for level in levels)), lot)
    if shares <= 0:
        return None
    proceeds, left, limit = 0.0, shares, best
    for level in levels:
        if left <= 1e-12:
            break
        take = min(level.size, left)
        proceeds += take * proceeds_per_share(level.price, fee_rate)
        left -= take
        limit = level.price
    return Sale(shares=shares, best_bid=best, edge_at_best=edge_at_best, limit_price=limit, proceeds=proceeds)
