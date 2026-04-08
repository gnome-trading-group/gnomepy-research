from __future__ import annotations

from gnomepy.java.schemas import JavaMBP10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class Impact(FlowSignal):
    """Net book displacement from a trade, in bps.

    impact_bps = (post_mid - pre_mid) / pre_mid * 10000
    """

    def __init__(self, warmup_trades: int = 20):
        self.warmup_trades = warmup_trades
        self._pre_trade_mid = 0
        self._pending_trade = False
        self._impact_bps = 0.0
        self._trade_count = 0

    def update(self, timestamp: int, data: JavaMBP10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2

        if is_trade_event(data):
            if not self._pending_trade and mid > 0:
                self._pre_trade_mid = mid
                self._pending_trade = True
        elif self._pending_trade and self._pre_trade_mid > 0:
            self._impact_bps = (mid - self._pre_trade_mid) / self._pre_trade_mid * 10000
            self._trade_count += 1
            self._pending_trade = False

    def value(self) -> float:
        return self._impact_bps

    def is_ready(self) -> bool:
        return self._trade_count >= self.warmup_trades

    def reset(self) -> None:
        self._pre_trade_mid = 0
        self._pending_trade = False
        self._impact_bps = 0.0
        self._trade_count = 0
