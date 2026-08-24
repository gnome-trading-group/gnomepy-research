from __future__ import annotations

from gnomepy.java.enums import OrderType

from gnomepy_research.strategies.equivalent_event_arb import EquivalentEventArb


class EquivalentEventArbLimit(EquivalentEventArb):
    def __init__(
        self,
        event_ids: list[int],
        max_position: int = 0,
        taker_fee_rate: float = 0.07,
        min_pure_arb_bps: float = 5.0,
        max_staleness_ns: int = 5_000_000_000,
        processing_time_ns: int = 5_000_000,
        allow_scaling: bool = False,
        imbalance_timeout_ns: int = 30_000_000_000,
        max_unhedged_qty: int = 0,
    ):
        super().__init__(
            event_ids=event_ids,
            max_position=max_position,
            taker_fee_rate=taker_fee_rate,
            min_pure_arb_bps=min_pure_arb_bps,
            max_staleness_ns=max_staleness_ns,
            processing_time_ns=processing_time_ns,
            allow_scaling=allow_scaling,
            imbalance_timeout_ns=imbalance_timeout_ns,
            max_unhedged_qty=max_unhedged_qty,
        )
        self._taker_order_type = OrderType.LIMIT
