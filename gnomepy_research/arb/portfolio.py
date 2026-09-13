from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from gnomepy import Scales

from gnomepy_research.arb.book_walking import OrderMode
from gnomepy_research.arb.types import Listing

if TYPE_CHECKING:
    from gnomepy_research.arb.pricing import LegConstraint

_PRICE_SCALE = Scales.PRICE
_SIZE_SCALE = Scales.SIZE


class ArbPhase(IntEnum):
    IDLE = 0
    ENTERING = 1
    FILLED = 2
    EXITING = 3


@dataclass
class ArbLeg:
    listing: Listing
    target_qty: int = 0
    filled_qty: int = 0
    fill_cost: float = 0.0

    @property
    def is_filled(self) -> bool:
        return self.target_qty > 0 and self.filled_qty >= self.target_qty

    @property
    def avg_fill_price(self) -> float:
        if self.filled_qty <= 0:
            return 0.0
        return self.fill_cost / (self.filled_qty / _SIZE_SCALE)

    def record_fill(self, fill_price: int, fill_qty: int) -> None:
        self.filled_qty += fill_qty
        self.fill_cost += (fill_price / _PRICE_SCALE) * (fill_qty / _SIZE_SCALE)

    def reset(self) -> None:
        self.target_qty = 0
        self.filled_qty = 0
        self.fill_cost = 0.0


class ArbPortfolio:
    def __init__(
        self,
        legs: list[ArbLeg],
        pairing_index: int,
        order_mode: OrderMode,
        cancel_grace_ns: int = 0,
    ) -> None:
        self._legs: dict[Listing, ArbLeg] = {leg.listing: leg for leg in legs}
        self.pairing_index = pairing_index
        self.order_mode = order_mode
        self.phase: ArbPhase = ArbPhase.ENTERING
        self._cancel_grace_ns = cancel_grace_ns
        self._cancel_trigger_ts: int | None = None

    def __contains__(self, listing: Listing) -> bool:
        return listing in self._legs

    @property
    def legs(self) -> list[ArbLeg]:
        return list(self._legs.values())

    def target_qty(self, listing: Listing) -> int:
        leg = self._legs.get(listing)
        return leg.target_qty if leg is not None else 0

    def as_target_dict(self) -> dict[Listing, int]:
        return {leg.listing: leg.target_qty for leg in self._legs.values()}

    def set_all_zero(self) -> None:
        for leg in self._legs.values():
            leg.target_qty = 0

    def begin_exit(self) -> None:
        self.set_all_zero()
        self.phase = ArbPhase.EXITING

    def add_target_qty(self, listing: Listing, delta: int) -> None:
        leg = self._legs.get(listing)
        if leg is not None:
            leg.target_qty += delta

    def record_fill(self, listing: Listing, fill_price: int, fill_qty: int) -> None:
        self._legs[listing].record_fill(fill_price, fill_qty)
        if self.phase == ArbPhase.ENTERING and all(leg.is_filled for leg in self._legs.values()):
            self.phase = ArbPhase.FILLED

    def should_cancel_entry(self, constraint: LegConstraint, timestamp: int = 0) -> bool:
        if self.order_mode == OrderMode.TAKER:
            return False
        if constraint.is_economically_valid():
            self._cancel_trigger_ts = None
            return False
        if self._cancel_trigger_ts is None:
            self._cancel_trigger_ts = timestamp
        return (timestamp - self._cancel_trigger_ts) >= self._cancel_grace_ns

    def reset(self) -> None:
        for leg in self._legs.values():
            leg.reset()
