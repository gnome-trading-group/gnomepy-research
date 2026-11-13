"""
Pure decision logic for the RT review-wave strategy: no engine, no registry, no JVM.

A film's first review wave starts when the third review of a cluster is seen: three reviews seen within cluster_ns of
each other. Only reviews seen soon after they were published (within max_lag_ns) count towards the start, so an old
review found late, or a backlog given its verdict at once after an outage, can't look like an embargo lifting.

The signal is the share of reviews (everything Claude called a review, as seen so far) that it called fresh, minus the
baseline: the median share across past films at the same point in their wave. Far enough from the baseline, with
enough reviews known, it says which way the film's Tomatometer will land relative to what the market expected.

Prices are plain probabilities in [0, 1] and sizes are contracts; the strategy converts to the engine's scaled integers
at the edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from gnomepy_research.models.rt.review_feed import Review

YES, NO = 1, -1


@dataclass(frozen=True)
class Signal:
    reviews: int
    fresh_share: float
    signal: float
    side: int  # YES, NO, or 0 for no trade


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True)
class Take:
    contracts: float
    limit_price: float
    cost: float  # at the levels it reaches, fees included


def wave_start(reviews: list[Review], min_reviews: int, cluster_ns: int, max_lag_ns: int) -> int | None:
    """When the first wave started: the time the min_reviews-th review of the first cluster was seen."""
    seen = sorted(r.seen_ns for r in reviews
                  if r.published_ns is None or r.seen_ns - r.published_ns <= max_lag_ns)
    for i in range(len(seen) - min_reviews + 1):
        if seen[i + min_reviews - 1] - seen[i] <= cluster_ns:
            return seen[i + min_reviews - 1]
    return None


def signal(reviews: list[Review], baseline: float, threshold: float, min_reviews: int) -> Signal:
    if not reviews:
        return Signal(0, math.nan, math.nan, 0)
    share = sum(r.fresh for r in reviews) / len(reviews)
    s = share - baseline
    side = 0 if len(reviews) < min_reviews or abs(s) < threshold - 1e-12 else (YES if s > 0 else NO)
    return Signal(len(reviews), share, s, side)


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
