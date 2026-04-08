from __future__ import annotations

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class Aggression(FlowSignal):
    """How far past mid the aggressor reached, in bps.

    aggression_bps = (trade_price - pre_mid) / pre_mid * 10000
    """

    def __init__(self, warmup_trades: int = 20):
        self.warmup_trades = warmup_trades
        self._pre_mid = 0
        self._aggression_bps = 0.0
        self._trade_count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2

        if is_trade_event(data) and self._pre_mid > 0:
            self._aggression_bps = (data.price - self._pre_mid) / self._pre_mid * 10000
            self._trade_count += 1

        self._pre_mid = mid

    def value(self) -> float:
        return self._aggression_bps

    def is_ready(self) -> bool:
        return self._trade_count >= self.warmup_trades

    def reset(self) -> None:
        self._pre_mid = 0
        self._aggression_bps = 0.0
        self._trade_count = 0
