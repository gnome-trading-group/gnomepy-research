"""
Pure pricing for the esports cross-venue arb: no engine, no registry, no JVM.

A pair is one contract that pays $1 if team A wins and one that pays $1 if team B wins, so a pair bought for less than
$1 after fees is a locked profit whatever happens in the match. Each team can be bought on a venue through a route:

- buy that team's listing at its asks, or
- on a venue that books a sale of an unheld contract as the opposite side (Kalshi books it as NO), sell the rival
  team's listing at its bids: a sale at b pays when the rival loses, so it costs 1 - b per contract.

A route's cost is its price plus the taker fee rate * c * (1 - c), the venues' curve, which is the same for a sale at b
as for a buy at 1 - b. The venues' fee rounding is left out.

Polymarket lists both teams of one market as mirror images (a token's bids are the other token's asks), so selling
there would only reach liquidity its asks already show; only venues whose teams are separate markets get the sale
route.

Prices here are plain probabilities in [0, 1] and sizes are contracts; the strategy converts to the engine's scaled
integers at the edge.
"""
from __future__ import annotations

from dataclasses import dataclass

BUY, SELL = 1, -1


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True)
class RouteLevel:
    """One book level, as a way to hold one team."""

    cost: float          # what one contract paying out on the team costs, before the fee
    size: float
    listing_id: int      # the listing the order goes to
    direction: int       # BUY at the listing's asks, or SELL at the rival listing's bids
    order_price: float   # the price on that listing: the ask, or the bid sold into
    fee_rate: float

    @property
    def all_in(self) -> float:
        return self.cost + self.fee_rate * self.cost * (1 - self.cost)


@dataclass(frozen=True)
class Order:
    listing_id: int
    direction: int
    contracts: float
    limit: float         # the worst price on the listing: the highest ask bought, or the lowest bid sold into


def team_ladder(own_listing: int, own_asks: list[Level], rival_listing: int, rival_bids: list[Level],
                fee_rate: float, sell_rival: bool) -> list[RouteLevel]:
    """Every way to hold one team on one venue, cheapest first."""
    out = [RouteLevel(lv.price, lv.size, own_listing, BUY, lv.price, fee_rate) for lv in own_asks if lv.size > 0]
    if sell_rival:
        out += [RouteLevel(1 - lv.price, lv.size, rival_listing, SELL, lv.price, fee_rate)
                for lv in rival_bids if lv.size > 0]
    return sorted(out, key=lambda r: (r.all_in, r.direction))


def _orders(taken: list[tuple[RouteLevel, float]]) -> list[Order]:
    """One order per listing and direction, limited at the worst price it took."""
    by: dict[tuple[int, int], Order] = {}
    for lv, n in taken:
        key = (lv.listing_id, lv.direction)
        prev = by.get(key)
        if prev is None:
            by[key] = Order(lv.listing_id, lv.direction, n, lv.order_price)
        else:
            worse = max(prev.limit, lv.order_price) if lv.direction == BUY else min(prev.limit, lv.order_price)
            by[key] = Order(lv.listing_id, lv.direction, prev.contracts + n, worse)
    return list(by.values())


@dataclass(frozen=True)
class PairPlan:
    contracts: float
    profit: float            # locked dollars if every order fills as planned
    edge_first: float        # the edge of the first contract, per contract
    orders_a: list[Order]
    orders_b: list[Order]


def plan_pair(ladder_a: list[RouteLevel], ladder_b: list[RouteLevel], min_edge: float,
              max_contracts: float) -> PairPlan | None:
    """The most pairs the two ladders give while every pair still clears min_edge after fees."""
    qa = [lv.size for lv in ladder_a]
    qb = [lv.size for lv in ladder_b]
    i = j = 0
    contracts = profit = 0.0
    edge_first = None
    taken_a: list[tuple[RouteLevel, float]] = []
    taken_b: list[tuple[RouteLevel, float]] = []
    while i < len(ladder_a) and j < len(ladder_b) and contracts < max_contracts - 1e-9:
        edge = 1 - ladder_a[i].all_in - ladder_b[j].all_in
        if edge < min_edge:
            break
        if edge_first is None:
            edge_first = edge
        n = min(qa[i], qb[j], max_contracts - contracts)
        taken_a.append((ladder_a[i], n))
        taken_b.append((ladder_b[j], n))
        contracts += n
        profit += n * edge
        qa[i] -= n
        qb[j] -= n
        if qa[i] <= 1e-9:
            i += 1
        if qb[j] <= 1e-9:
            j += 1
    if contracts <= 0:
        return None
    return PairPlan(contracts, profit, edge_first, _orders(taken_a), _orders(taken_b))


def whole(plan: PairPlan, min_size: dict[int, float]) -> PairPlan | None:
    """The plan cut to whole contracts on both sides, or None when an order would fall under its listing's minimum.

    Both sides keep the same count: the plan only clears its edge as pairs."""
    target = float(int(plan.contracts + 1e-9))
    if target < 1:
        return None
    scale = target / plan.contracts

    def cut(orders: list[Order]) -> list[Order] | None:
        out, left = [], target
        for k, o in enumerate(sorted(orders, key=lambda o: -o.contracts)):
            n = left if k == len(orders) - 1 else float(int(o.contracts * scale + 1e-9))
            n = min(n, left)
            if n <= 0:
                continue
            if n < min_size.get(o.listing_id, 0):
                return None
            out.append(Order(o.listing_id, o.direction, n, o.limit))
            left -= n
        return out if left <= 1e-9 else None

    a, b = cut(plan.orders_a), cut(plan.orders_b)
    if a is None or b is None:
        return None
    return PairPlan(target, plan.profit * scale, plan.edge_first, a, b)


def exposure(holdings: list[tuple[int, float]]) -> tuple[float, float]:
    """(contracts paying if team A wins, contracts paying if team B wins) from (team, net position) per listing: a
    long pays when its team wins, a short when its team loses."""
    pays = [0.0, 0.0]
    for team, net in holdings:
        if net > 0:
            pays[team] += net
        elif net < 0:
            pays[1 - team] += -net
    return pays[0], pays[1]


def plan_completion(ladder: list[RouteLevel], contracts: float, max_slippage: float,
                    min_size: dict[int, float]) -> list[Order]:
    """Orders that buy up to `contracts` of the short team, no dearer than the best level's cost + max_slippage."""
    if not ladder or contracts < 1:
        return []
    cap = ladder[0].cost + max_slippage
    taken, left = [], float(int(contracts + 1e-9))
    for lv in ladder:
        if left <= 0 or lv.cost > cap + 1e-12:
            break
        n = min(lv.size, left)
        taken.append((lv, n))
        left -= n
    orders = []
    for o in _orders(taken):
        n = float(int(o.contracts + 1e-9))
        if n >= max(1.0, min_size.get(o.listing_id, 0)):
            orders.append(Order(o.listing_id, o.direction, n, o.limit))
    return orders
