"""Fee schedules for the listings a backtest traded, as the registry records them over time."""
from __future__ import annotations

import pandas as pd

from gnomepy.fees import FeeSchedule, FeeTimeline, load_fee_timelines
from gnomepy.registry import RegistryClient


class ListingFees:
    """The fee schedule each (exchange_id, security_id) listing charged at a given time."""

    def __init__(self, keys, registry: RegistryClient | None = None):
        registry = registry or RegistryClient()
        self._listing: dict[tuple[int, int], int] = {}
        for exchange_id, security_id in sorted(set(keys)):
            hits = registry.get_listing(exchange_id=exchange_id, security_id=security_id)
            if not hits:
                raise ValueError(f"no registry listing for exchange {exchange_id} security {security_id}")
            self._listing[(exchange_id, security_id)] = int(hits[0].listing_id)
        ids = sorted(set(self._listing.values()))
        self._timelines = load_fee_timelines(registry, ids) if ids else {}

    def at(self, exchange_id: int, security_id: int, timestamp_nanos: int) -> FeeSchedule:
        listing = self._listing[(exchange_id, security_id)]
        schedule = self._timelines.get(listing, FeeTimeline()).at(timestamp_nanos)
        if schedule is None:
            raise ValueError(f"registry has no fee schedule for listing {listing} (exchange {exchange_id} "
                             f"security {security_id}) at {pd.Timestamp(timestamp_nanos, unit='ns', tz='UTC')}")
        return schedule
