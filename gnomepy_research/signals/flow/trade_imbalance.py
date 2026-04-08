from __future__ import annotations

from collections import deque

from gnomepy.java.enums import Side
from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal


class TradeImbalance(FlowSignal):
    """Trade-flow imbalance signal over a rolling horizon.

    Tracks buy vs sell volume within a time window and outputs imbalance
    in [-1, +1]. Trades older than the horizon are dropped.

    imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)

    Args:
        horizon_ns: Rolling window in nanoseconds.
        warmup_trades: Minimum trades before is_ready() returns True.
    """

    def __init__(self, horizon_ns: int = 1_000_000_000, warmup_trades: int = 20):
        self.horizon_ns = horizon_ns
        self.warmup_trades = warmup_trades

        self._trades: deque[tuple[int, int]] = deque()
        self._buy_volume = 0
        self._sell_volume = 0
        self._trade_count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        size = data.size
        side = data.side

        if size <= 0 or side == Side.NONE:
            return

        self._evict(timestamp)

        if side == Side.ASK:
            self._buy_volume += size
            self._trades.append((timestamp, size))
        else:
            self._sell_volume += size
            self._trades.append((timestamp, -size))

        self._trade_count += 1

    def _evict(self, now: int) -> None:
        cutoff = now - self.horizon_ns
        while self._trades and self._trades[0][0] < cutoff:
            _, signed_size = self._trades.popleft()
            if signed_size > 0:
                self._buy_volume -= signed_size
            else:
                self._sell_volume -= (-signed_size)

    def value(self) -> float:
        total = self._buy_volume + self._sell_volume
        if total <= 0:
            return 0.0
        return (self._buy_volume - self._sell_volume) / total

    def is_ready(self) -> bool:
        return self._trade_count >= self.warmup_trades

    def reset(self) -> None:
        self._trades.clear()
        self._buy_volume = 0
        self._sell_volume = 0
        self._trade_count = 0
