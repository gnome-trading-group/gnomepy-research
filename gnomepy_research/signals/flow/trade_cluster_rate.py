from __future__ import annotations

from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class TradeClusterRate(FlowSignal):
    """Fraction of trades arriving within cluster_ns of the previous trade.

    High rates indicate algorithmic or informed burst flow.
    Low rates = evenly spaced uninformed flow.

    Output in [0, 1].

    Args:
        horizon_ns: Rolling window over which the rate is computed.
        cluster_ns: Max inter-trade gap to be considered clustered.
        warmup_trades: Minimum trades before is_ready() returns True.
    """

    def __init__(
        self,
        horizon_ns: int = 10_000_000_000,
        cluster_ns: int = 1_000_000,
        warmup_trades: int = 50,
    ):
        self.horizon_ns = horizon_ns
        self.cluster_ns = cluster_ns
        self.warmup_trades = warmup_trades
        self._trades: deque[tuple[int, bool]] = deque()  # (ts, is_clustered)
        self._clustered_count = 0
        self._last_trade_ts = -1
        self._total_trades = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if not is_trade_event(data):
            return

        self._total_trades += 1
        is_clustered = (
            self._last_trade_ts >= 0
            and timestamp - self._last_trade_ts <= self.cluster_ns
        )
        self._last_trade_ts = timestamp

        self._evict(timestamp)
        if is_clustered:
            self._clustered_count += 1
        self._trades.append((timestamp, is_clustered))

    def _evict(self, now: int) -> None:
        cutoff = now - self.horizon_ns
        while self._trades and self._trades[0][0] < cutoff:
            _, was_clustered = self._trades.popleft()
            if was_clustered:
                self._clustered_count -= 1

    def value(self) -> float:
        n = len(self._trades)
        if n == 0:
            return 0.0
        return self._clustered_count / n

    def is_ready(self) -> bool:
        return self._total_trades >= self.warmup_trades

    def reset(self) -> None:
        self._trades.clear()
        self._clustered_count = 0
        self._last_trade_ts = -1
        self._total_trades = 0
