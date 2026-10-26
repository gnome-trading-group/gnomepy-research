"""A listing a strategy trades. Its registry id keys the strategy's state; its venue's exchange and security ids are
what orders, positions and execution reports carry. Market data names only its stream: a strategy's
``self.streams`` maps a stream, or a venue's ids, to the listing id."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Listing:
    listing_id: int
    exchange_id: int
    security_id: int

    @classmethod
    def from_registry(cls, listing) -> Listing:
        """From a registry client listing (gnomepy.registry)."""
        return cls(int(listing.listing_id), int(listing.exchange_id), int(listing.security_id))

    @classmethod
    def from_streams(cls, streams, listing_id: int) -> Listing:
        """From the strategy's stream directory, for a listing it reads."""
        listing = streams.listing(listing_id)
        if listing is None:
            raise ValueError(f"Listing {listing_id} is not one of the strategy's streams")
        return cls(int(listing_id), int(listing.exchange().exchangeId()), int(listing.security().securityId()))
