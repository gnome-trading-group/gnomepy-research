"""
Pure pricing for the Polymarket International maker branch: the bid to rest on one team so that, if it fills, buying
the other team elsewhere at today's prices still locks min_edge per pair.
"""
from __future__ import annotations

from collections import deque

from gnomepy.price_grid import PriceGrid

from gnomepy_research.sessions.esports_cross_venue_arb.decision import RouteLevel


PRICE_SCALE = 1_000_000_000


def maker_bid(hedge: list[RouteLevel], quote_size: float, min_edge: float, maker_rate: float, grid: PriceGrid,
              best_ask: float | None, depth_band: float) -> tuple[float, float] | None:
    """(price, contracts) to bid, or None.

    The size is what the hedge ladder offers within depth_band of its best all-in cost, up to quote_size; the price is
    the highest price on the listing's grid whose cost with the maker fee still leaves min_edge below $1 after the
    deepest hedge level used. It stays a grid step under the best ask, so it rests rather than takes.
    """
    if not hedge:
        return None
    floor = hedge[0].all_in + depth_band
    size, worst = 0.0, hedge[0].all_in
    for lv in hedge:
        if lv.all_in > floor + 1e-12 or size >= quote_size:
            break
        size += min(lv.size, quote_size - size)
        worst = lv.all_in
    budget = 1 - worst - min_edge
    if budget * PRICE_SCALE < grid.min:
        return None
    price = grid.floor(min(round(budget * PRICE_SCALE), grid.max))
    while price > 0 and _cost(price, maker_rate) > budget + 1e-12:
        price = grid.down(price)
    if best_ask is not None:
        price = min(price, grid.down(grid.ceil(round(best_ask * PRICE_SCALE))))
    size = float(int(size + 1e-9))
    if price < grid.min or size < 1:
        return None
    return price / PRICE_SCALE, size


def _cost(scaled_price: int, maker_rate: float) -> float:
    price = scaled_price / PRICE_SCALE
    return price + maker_rate * price * (1 - price)


class MoveWindow:
    """How far a price has ranged over the last `seconds`: a book that has just jumped is likely to jump again, and a
    bid resting through the next jump is filled by whoever saw it first."""

    def __init__(self, seconds: float):
        self.window_ns = int(seconds * 1e9)
        self._points: deque[tuple[int, float]] = deque()

    def add(self, now_ns: int, price: float) -> None:
        self._points.append((now_ns, price))
        while self._points and self._points[0][0] < now_ns - self.window_ns:
            self._points.popleft()

    def range(self) -> float:
        if not self._points:
            return 0.0
        prices = [p for _, p in self._points]
        return max(prices) - min(prices)
