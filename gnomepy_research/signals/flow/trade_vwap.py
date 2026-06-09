from __future__ import annotations

from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class TradeVWAP(FlowSignal):
    """Deviation of rolling trade VWAP from current mid, in basis points.

    Positive = trades have been executing above mid (buy-side aggression).
    Negative = trades executing below mid (sell-side aggression).

    Args:
        horizon_ns: Rolling trade window in nanoseconds.
        warmup_trades: Minimum trades before is_ready() returns True.
    """

    def __init__(self, horizon_ns: int = 1_000_000_000, warmup_trades: int = 20):
        self.horizon_ns = horizon_ns
        self.warmup_trades = warmup_trades

        self._trades: deque[tuple[int, int, int]] = deque()  # (ts, price, size)
        self._price_vol = 0.0
        self._vol = 0
        self._last_mid = 0
        self._trade_count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid > 0 and ask > 0:
            self._last_mid = (bid + ask) // 2

        if not is_trade_event(data):
            return

        self._trade_count += 1
        self._evict(timestamp)
        self._price_vol += data.price * data.size
        self._vol += data.size
        self._trades.append((timestamp, data.price, data.size))

    def _evict(self, now: int) -> None:
        cutoff = now - self.horizon_ns
        while self._trades and self._trades[0][0] < cutoff:
            _, price, size = self._trades.popleft()
            self._price_vol -= price * size
            self._vol -= size

    def value(self) -> float:
        if self._vol <= 0 or self._last_mid <= 0:
            return 0.0
        vwap = self._price_vol / self._vol
        return (vwap - self._last_mid) / self._last_mid * 10000.0

    def is_ready(self) -> bool:
        return self._trade_count >= self.warmup_trades

    def reset(self) -> None:
        self._trades.clear()
        self._price_vol = 0.0
        self._vol = 0
        self._last_mid = 0
        self._trade_count = 0
