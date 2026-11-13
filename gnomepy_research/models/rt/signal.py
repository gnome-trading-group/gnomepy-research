"""
What a film's reviews, as seen so far, say about where its Tomatometer will land: shared by the RT sessions.

A film's first review wave starts when the third review of a cluster is seen: three reviews seen within cluster_ns of
each other. Only reviews seen soon after they were published (within max_lag_ns) count towards the start, so an old
review found late, or a backlog given its verdict at once after an outage, can't look like an embargo lifting.

The signal is the share of reviews (everything Claude called a review, as seen so far) that it called fresh, minus the
baseline: the median share across past films at the same point in their wave. Far enough from the baseline, with
enough reviews known, it says which way the film's Tomatometer will land relative to what the market expected.
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
    side: int  # YES, NO, or 0 for no side


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
