"""The wave and the signal the RT sessions read from a film's reviews."""
from __future__ import annotations

import pandas as pd

from gnomepy_research.models.rt.review_feed import Review
from gnomepy_research.models.rt.signal import NO, YES, signal, wave_start

MINUTE = 60_000_000_000
HOUR = 60 * MINUTE
T0 = pd.Timestamp("2026-11-18T14:00:00Z").value


def r(i, seen_min, fresh=True, published_min=None):
    return Review(f"u{i}", T0 + seen_min * MINUTE, T0 + (seen_min if published_min is None else published_min) * MINUTE,
                  fresh)


def test_the_wave_starts_when_the_third_review_of_a_cluster_is_seen():
    reviews = [r(1, 0), r(2, 1), r(3, 2), r(4, 30)]
    assert wave_start(reviews, 3, 6 * HOUR, 6 * HOUR) == T0 + 2 * MINUTE
    assert wave_start(reviews[:2], 3, 6 * HOUR, 6 * HOUR) is None


def test_reviews_spread_wider_than_the_cluster_are_no_wave():
    assert wave_start([r(1, 0), r(2, 7 * 60), r(3, 14 * 60)], 3, 6 * HOUR, 6 * HOUR) is None


def test_old_reviews_seen_late_never_start_a_wave():
    # A backlog given its verdicts at once (Claude back after an outage, or a restart): published days before.
    backlog = [r(i, 0, published_min=-3 * 24 * 60) for i in range(5)]
    assert wave_start(backlog, 3, 6 * HOUR, 6 * HOUR) is None
    assert wave_start(backlog + [r(9, 1), r(10, 2), r(11, 3)], 3, 6 * HOUR, 6 * HOUR) == T0 + 3 * MINUTE


def test_the_signal_is_the_fresh_share_against_the_baseline():
    fresh = [r(1, 0), r(2, 0), r(3, 0)]
    assert signal(fresh, 0.667, 0.10, 3).side == YES
    rotten = [r(1, 0, False), r(2, 0, False), r(3, 0)]
    assert signal(rotten, 0.667, 0.10, 3).side == NO
    near = [r(1, 0), r(2, 0), r(3, 0, False)]
    assert signal(near, 0.667, 0.10, 3).side == 0, "2/3 is the baseline itself"
    assert signal(fresh[:2], 0.667, 0.10, 3).side == 0, "too few reviews"
