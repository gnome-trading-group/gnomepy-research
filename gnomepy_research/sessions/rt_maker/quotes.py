"""
Pure quoting rules for the RT maker: no engine, no registry, no JVM.

A side is quoted at the best price other makers show there: the first level, walking away from the touch, holding at
least min_join_size of someone else's size. A live book includes our own resting order, so our size is taken off the
level it rests at first; otherwise a level we were left alone on would keep us there after the others had gone.

Sizes are capped by the strike's position limit, so a side stops once the position reaches it, and by the money the
film may commit. Prices are plain probabilities in [0, 1] and sizes contracts; the strategy converts at the edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True)
class Quote:
    bid_price: float = 0.0
    bid_size: float = 0.0
    ask_price: float = 0.0
    ask_size: float = 0.0

    @property
    def empty(self) -> bool:
        return self.bid_size <= 0 and self.ask_size <= 0


NONE = Quote()


def join_price(levels: list[Level], ours: tuple[float, float] | None, min_join_size: float) -> float | None:
    """The best price on one side (levels best first) with at least min_join_size of others' size; ours is our resting
    (price, size) on that side when the book shows it."""
    for level in levels:
        others = level.size
        if ours is not None and math.isclose(level.price, ours[0]):
            others -= ours[1]
        if others >= min_join_size:
            return level.price
    return None


def lots(contracts: float, lot: float, min_size: float) -> float:
    """Contracts rounded down to the lot, or 0 under the venue's minimum."""
    n = math.floor(contracts / lot + 1e-9) * lot if lot > 0 else contracts
    return n if n >= max(min_size, lot) else 0.0


def quote(bids: list[Level], asks: list[Level], position: float, resting: Quote | None, quote_size: float,
          max_position: float, min_price: float, max_price: float, min_join_size: float, lot: float,
          min_size: float) -> Quote:
    """Both sides of one strike. position is YES contracts held (negative: sold); resting is what we have in the book
    when the book shows it (None otherwise)."""
    bid = join_price(bids, (resting.bid_price, resting.bid_size) if resting and resting.bid_size else None,
                     min_join_size)
    ask = join_price(asks, (resting.ask_price, resting.ask_size) if resting and resting.ask_size else None,
                     min_join_size)
    if bid is not None and ask is not None and bid >= ask:
        return NONE
    bid_size = ask_size = 0.0
    if bid is not None and min_price <= bid <= max_price:
        bid_size = lots(min(quote_size, max_position - position), lot, min_size)
    if ask is not None and min_price <= ask <= max_price:
        ask_size = lots(min(quote_size, max_position + position), lot, min_size)
    return Quote(bid if bid_size else 0.0, bid_size, ask if ask_size else 0.0, ask_size)


def adds(q: Quote, position: float) -> float:
    """What filling the quote's sides could add to the money the strike commits: a bid commits its price per contract
    beyond what it would buy back of a short, an ask one minus its price beyond what it would sell of a long."""
    bid = max(0.0, q.bid_size - max(0.0, -position)) * q.bid_price
    ask = max(0.0, q.ask_size - max(0.0, position)) * (1 - q.ask_price)
    return bid + ask


def reducing(q: Quote, position: float, lot: float, min_size: float) -> Quote:
    """Only the side that trades the position back towards flat, at most the position."""
    bid = lots(min(q.bid_size, max(0.0, -position)), lot, min_size)
    ask = lots(min(q.ask_size, max(0.0, position)), lot, min_size)
    return Quote(q.bid_price if bid else 0.0, bid, q.ask_price if ask else 0.0, ask)


def held_cost(position: float, avg_price: float) -> float:
    """Money a held position commits: a long cost its price, a short (YES sold) one minus it."""
    return position * avg_price if position >= 0 else -position * (1 - avg_price)
