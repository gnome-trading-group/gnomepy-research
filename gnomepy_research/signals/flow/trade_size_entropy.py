from __future__ import annotations

import math
from collections import deque

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class TradeSizeEntropy(FlowSignal):
    """Rolling Shannon entropy of the trade size distribution.

    Sizes are bucketed logarithmically (bucket = floor(log10(size))).
    High entropy = uniform size distribution (algorithmic flow).
    Low entropy = concentrated in a few size buckets (mixed or block flow).

    Output in bits (log base 2). Max = log2(num_buckets).

    Args:
        horizon_ns: Rolling window in nanoseconds.
        num_buckets: Number of log10-scale size buckets (default 10 covers
            sizes 1 through 10^10).
        warmup_trades: Minimum trades before is_ready() returns True.
    """

    def __init__(
        self,
        horizon_ns: int = 60_000_000_000,
        num_buckets: int = 10,
        warmup_trades: int = 50,
    ):
        self.horizon_ns = horizon_ns
        self.num_buckets = num_buckets
        self.warmup_trades = warmup_trades
        self._bucket_counts = [0] * num_buckets
        self._trades: deque[tuple[int, int]] = deque()  # (ts, bucket)
        self._total = 0
        self._total_trades = 0

    def _bucket(self, size: int) -> int:
        if size <= 0:
            return 0
        return min(int(math.log10(max(size, 1))), self.num_buckets - 1)

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if not is_trade_event(data):
            return

        self._total_trades += 1
        bucket = self._bucket(data.size)
        self._evict(timestamp)
        self._bucket_counts[bucket] += 1
        self._total += 1
        self._trades.append((timestamp, bucket))

    def _evict(self, now: int) -> None:
        cutoff = now - self.horizon_ns
        while self._trades and self._trades[0][0] < cutoff:
            _, b = self._trades.popleft()
            self._bucket_counts[b] -= 1
            self._total -= 1

    def value(self) -> float:
        if self._total <= 0:
            return 0.0
        h = 0.0
        for count in self._bucket_counts:
            if count > 0:
                p = count / self._total
                h -= p * math.log2(p)
        return h

    def is_ready(self) -> bool:
        return self._total_trades >= self.warmup_trades

    def reset(self) -> None:
        for i in range(self.num_buckets):
            self._bucket_counts[i] = 0
        self._trades.clear()
        self._total = 0
        self._total_trades = 0
