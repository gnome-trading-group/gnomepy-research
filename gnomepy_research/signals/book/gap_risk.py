from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.book.base import BookSignal


class GapRisk(BookSignal):
    """Weighted price gaps between consecutive book levels.

    For each side: sum of (gap_bps_i / (size_i + 1)) across adjacent level pairs.
    Thin levels with large gaps behind them contribute most to gap risk.

    High values = a market order would find large price gaps and little liquidity.
    Output in bps per lot (larger = riskier book structure).

    Args:
        num_levels: Number of price levels to include (max 10).
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(self, num_levels: int = 5, warmup: int = 10):
        self.num_levels = num_levels
        self.warmup = warmup
        self._value = 0.0
        self._count = 0

    def _side_gap_risk(self, data: Mbp10Schema, mid: float, use_bid: bool) -> float:
        risk = 0.0
        prev_price = 0.0
        for i in range(self.num_levels):
            price = data.bid_price(i) if use_bid else data.ask_price(i)
            size = data.bid_size(i) if use_bid else data.ask_size(i)
            if price <= 0:
                break
            if prev_price > 0:
                gap_bps = abs(price - prev_price) / mid * 10000.0
                risk += gap_bps / (size + 1)
            prev_price = price
        return risk

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid0 = data.bid_price(0)
        ask0 = data.ask_price(0)
        if bid0 <= 0 or ask0 <= 0:
            return
        mid = (bid0 + ask0) / 2.0

        bid_risk = self._side_gap_risk(data, mid, use_bid=True)
        ask_risk = self._side_gap_risk(data, mid, use_bid=False)
        self._value = (bid_risk + ask_risk) / 2.0
        self._count += 1

    def value(self) -> float:
        return self._value

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0
