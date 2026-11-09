"""
Pure pricing for the Polymarket International maker branch: the bid to rest on one team so that, if it fills, buying
the other team elsewhere at today's prices still locks min_edge per pair.
"""
from __future__ import annotations

import math

from gnomepy_research.sessions.cs2_cross_venue_arb.decision import RouteLevel


def maker_bid(hedge: list[RouteLevel], quote_size: float, min_edge: float, maker_rate: float, tick: float,
              best_ask: float | None, depth_band: float) -> tuple[float, float] | None:
    """(price, contracts) to bid, or None.

    The size is what the hedge ladder offers within depth_band of its best all-in cost, up to quote_size; the price is
    the highest tick whose cost with the maker fee still leaves min_edge below $1 after the deepest hedge level used.
    It stays a tick under the best ask, so it rests rather than takes.
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
    price = math.floor(budget / tick + 1e-9) * tick
    while price > 0 and price + maker_rate * price * (1 - price) > budget + 1e-12:
        price -= tick
    if best_ask is not None:
        price = min(price, best_ask - tick)
    price = round(price / tick) * tick
    size = float(int(size + 1e-9))
    if price < tick or size < 1:
        return None
    return round(price, 10), size
