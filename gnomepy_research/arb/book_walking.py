from __future__ import annotations

from enum import IntEnum
from math import exp
from typing import Callable, Protocol

from gnomepy import Scales, Side
from gnomepy.java.backtest.orders import ExecutionReport

from gnomepy_research.arb.types import Listing


class OrderMode(IntEnum):
    TAKER = 0
    MAKER = 1


class CostModel(Protocol):
    def on_book_update(
        self, listing: Listing,
        bids: list[tuple[int, int]], asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None: ...

    def on_fill(
        self, listing: Listing, report: ExecutionReport,
        order_mode: OrderMode,
    ) -> None: ...

    def expected_cost(
        self, listing: Listing, price: int, side: Side,
        order_mode: OrderMode, qty: int,
    ) -> float: ...


class CostComponent(Protocol):
    def on_book_update(
        self, listing: Listing,
        bids: list[tuple[int, int]], asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None: ...

    def on_fill(
        self, listing: Listing, report: ExecutionReport,
        order_mode: OrderMode,
    ) -> None: ...

    def cost(self, listing: Listing, price: int, side: Side, qty: int) -> float: ...


class CompositeCostModel:
    def __init__(
        self,
        shared: list[CostComponent] | None = None,
        maker: list[CostComponent] | None = None,
        taker: list[CostComponent] | None = None,
    ):
        self._shared = shared or []
        self._maker = maker or []
        self._taker = taker or []

    def on_book_update(
        self, listing: Listing,
        bids: list[tuple[int, int]], asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None:
        for c in self._shared + self._maker + self._taker:
            c.on_book_update(listing, bids, asks, timestamp)

    def on_fill(
        self, listing: Listing, report: ExecutionReport,
        order_mode: OrderMode,
    ) -> None:
        for c in self._shared + self._maker + self._taker:
            c.on_fill(listing, report, order_mode)

    def expected_cost(
        self, listing: Listing, price: int, side: Side,
        order_mode: OrderMode, qty: int,
    ) -> float:
        components = self._shared + (self._maker if order_mode == OrderMode.MAKER else self._taker)
        return sum(c.cost(listing, price, side, qty) for c in components)


class FeeCost:
    """Prediction market fee component: rate * p * (1 - p)."""

    def __init__(self, rates: dict[int, float]):
        self._rates = rates

    def on_book_update(self, listing: Listing, bids, asks, timestamp) -> None:
        pass

    def on_fill(self, listing: Listing, report, order_mode) -> None:
        pass

    def cost(self, listing: Listing, price: int, side: Side, qty: int) -> float:
        rate = self._rates.get(listing[0], 0.0)
        if rate == 0.0:
            return 0.0
        p = price / Scales.PRICE
        return rate * p * (1.0 - p)


class FillRiskCost:
    """Fill risk penalty for maker orders based on bid-ask spread.

    P(fill) ~ exp(-lambda * spread). Penalty = (1 - P_fill) * spread * mult.
    """

    def __init__(self, fill_risk_lambda: float = 10.0, unwind_spread_mult: float = 2.0):
        self._lambda = fill_risk_lambda
        self._mult = unwind_spread_mult
        self._spreads: dict[Listing, float] = {}

    def on_book_update(
        self, listing: Listing,
        bids: list[tuple[int, int]], asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None:
        if bids and asks and bids[0][0] > 0 and asks[0][0] > 0:
            new_spread = (asks[0][0] - bids[0][0]) / Scales.PRICE
            if new_spread > 0:
                self._spreads[listing] = new_spread
        else:
            self._spreads[listing] = 1.0

    def on_fill(self, listing: Listing, report, order_mode) -> None:
        pass

    def cost(self, listing: Listing, price: int, side: Side, qty: int) -> float:
        spread = self._spreads.get(listing, 0.0)
        if spread <= 0:
            return 0.0
        p_fill = exp(-self._lambda * spread)
        return (1.0 - p_fill) * spread * self._mult

    def get_spread(self, listing: Listing) -> float:
        return self._spreads.get(listing, 0.0)

    def get_fill_prob(self, listing: Listing) -> float:
        spread = self._spreads.get(listing, 0.0)
        if spread <= 0:
            return 1.0
        return exp(-self._lambda * spread)


class DepthCoverageCost:
    """Taker slippage from insufficient cheap depth to absorb our order.

    At order submission time, if order_size / depth_within_price_window exceeds
    coverage_multiplier, competing flow has likely swept cheap levels and we'll
    walk into expensive ones. Penalizes by the fraction of order that lacks coverage
    times the price span of the window.

    Validated cross-event: fill ratio (order_size / cheap_depth_2c) has r=0.919
    with actual slippage across 4 events, 18 taker fills, zero false positives at
    multiplier=2.0.
    """

    def __init__(
        self, coverage_multiplier: float = 2.0,
        price_window_cents: int = 2,
    ):
        self._coverage_mult = coverage_multiplier
        self._price_window = int(price_window_cents * 0.01 * Scales.PRICE)
        self._ask_cheap_depth: dict[Listing, float] = {}
        self._bid_cheap_depth: dict[Listing, float] = {}
        self._ask_span: dict[Listing, float] = {}
        self._bid_span: dict[Listing, float] = {}

    def on_book_update(
        self, listing: Listing,
        bids: list[tuple[int, int]], asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None:
        if asks:
            best = asks[0][0]
            ceiling = best + self._price_window
            depth = sum(s for p, s in asks if p <= ceiling) / Scales.SIZE
            self._ask_cheap_depth[listing] = depth
            self._ask_span[listing] = self._price_window / Scales.PRICE
        else:
            self._ask_cheap_depth[listing] = 0.0

        if bids:
            best = bids[0][0]
            floor = best - self._price_window
            depth = sum(s for p, s in bids if p >= floor) / Scales.SIZE
            self._bid_cheap_depth[listing] = depth
            self._bid_span[listing] = self._price_window / Scales.PRICE
        else:
            self._bid_cheap_depth[listing] = 0.0

    def on_fill(
        self, listing: Listing, report: ExecutionReport,
        order_mode: OrderMode,
    ) -> None:
        pass

    def cost(self, listing: Listing, price: int, side: Side, qty: int) -> float:
        if side == Side.BID:
            depth = self._ask_cheap_depth.get(listing, 0.0)
            span = self._ask_span.get(listing, 0.0)
        else:
            depth = self._bid_cheap_depth.get(listing, 0.0)
            span = self._bid_span.get(listing, 0.0)

        if depth <= 0.0 or span <= 0.0:
            return 0.0

        qty_f = qty / Scales.SIZE
        if qty_f <= 0.0 or depth >= qty_f * self._coverage_mult:
            return 0.0

        walk_fraction = min(1.0, (qty_f - depth) / qty_f)
        return walk_fraction * span

    def get_ask_depth(self, listing: Listing) -> float:
        return self._ask_cheap_depth.get(listing, 0.0)

    def get_bid_depth(self, listing: Listing) -> float:
        return self._bid_cheap_depth.get(listing, 0.0)


def walk_books(
    books: list[list[tuple[int, int]]],
    cost_fns: list[Callable[[int, int], float]],
    max_qty: int = 0,
    buy: bool = True,
) -> tuple[int, float]:
    """Walk order books for multiple legs simultaneously.

    For buy: walks ask books, accumulates until sum(prices) + sum(costs) >= 1.0.
    For sell: walks bid books, accumulates until sum(prices) - sum(costs) <= 1.0.

    Uses a segment-based approach: precomputes quantity breakpoints from cumulative
    book sizes, then walks segments where prices are constant. Within a segment,
    if the cost function is non-linear and the segment end is unprofitable, binary
    search finds the exact profitable cutoff at SIZE granularity.

    Returns (total_qty_raw, edge_bps).
    """
    num_legs = len(books)

    breakpoints: list[int] = []
    for book in books:
        cum = 0
        for _, size in book:
            cum += size
            breakpoints.append(cum)
    breakpoints = sorted(set(breakpoints))
    if max_qty > 0:
        breakpoints = [b for b in breakpoints if b <= max_qty]
        if not breakpoints or breakpoints[-1] < max_qty:
            breakpoints.append(max_qty)

    def get_prices(q: int) -> list[int]:
        result = []
        for book in books:
            cum = 0
            for lvl, (price, size) in enumerate(book):
                cum += size
                if q < cum or lvl == len(book) - 1:
                    result.append(price)
                    break
        return result

    def is_profitable(q: int) -> bool:
        prices_raw = get_prices(q)
        prices_f = [p / Scales.PRICE for p in prices_raw]
        costs_f = [cost_fns[i](prices_raw[i], q) for i in range(num_legs)]
        if buy:
            return sum(prices_f) + sum(costs_f) < 1.0
        return sum(prices_f) - sum(costs_f) > 1.0

    total_qty_raw = 0
    total_value = [0.0] * num_legs
    total_costs = [0.0] * num_legs

    prev = 0
    for bp in breakpoints:
        prices_raw = get_prices(prev)
        prices_f = [p / Scales.PRICE for p in prices_raw]
        costs_at_start = [cost_fns[i](prices_raw[i], prev) for i in range(num_legs)]

        if buy and sum(prices_f) + sum(costs_at_start) >= 1.0:
            break
        if not buy and sum(prices_f) - sum(costs_at_start) <= 1.0:
            break

        costs_at_end = [cost_fns[i](prices_raw[i], bp) for i in range(num_legs)]
        end_profitable = (
            (sum(prices_f) + sum(costs_at_end) < 1.0) if buy
            else (sum(prices_f) - sum(costs_at_end) > 1.0)
        )

        if end_profitable:
            chunk_raw = bp - prev
        else:
            lo, hi = prev, bp
            while hi - lo > Scales.SIZE:
                mid = (lo + hi) // 2
                mid = mid - (mid % Scales.SIZE)
                if mid <= lo:
                    mid = lo + Scales.SIZE
                if mid >= hi:
                    break
                if is_profitable(mid):
                    lo = mid
                else:
                    hi = mid
            chunk_raw = lo - prev
            if chunk_raw <= 0:
                break

        chunk_f = chunk_raw / Scales.SIZE
        total_qty_raw += chunk_raw
        for i in range(num_legs):
            cost_mid = cost_fns[i](prices_raw[i], prev + chunk_raw // 2)
            total_value[i] += prices_f[i] * chunk_f
            total_costs[i] += cost_mid * chunk_f

        prev = bp

    if total_qty_raw <= 0:
        return 0, float("-inf")

    total_qty_f = total_qty_raw / Scales.SIZE
    if buy:
        edge_bps = (total_qty_f - sum(total_value) - sum(total_costs)) / total_qty_f * 10_000
    else:
        edge_bps = (sum(total_value) - total_qty_f - sum(total_costs)) / total_qty_f * 10_000
    return total_qty_raw, edge_bps
