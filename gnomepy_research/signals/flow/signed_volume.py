from __future__ import annotations

from collections import deque

from gnomepy.java.enums import Side
from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class SignedVolume(FlowSignal):
    """Cumulative signed trade volume over a rolling window.

    Positive = net buy flow; negative = net sell flow. Unlike TradeImbalance,
    magnitude is preserved so large directional bursts show up at full scale.

    Args:
        horizon_ns: Rolling window in nanoseconds.
        warmup_trades: Minimum trades before is_ready() returns True.
    """

    def __init__(self, horizon_ns: int = 1_000_000_000, warmup_trades: int = 20):
        self.horizon_ns = horizon_ns
        self.warmup_trades = warmup_trades
        self._trades: deque[tuple[int, int]] = deque()
        self._signed_vol = 0
        self._trade_count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if not is_trade_event(data):
            return
        side = data.side
        if side == Side.NONE:
            return

        self._trade_count += 1
        self._evict(timestamp)

        signed = data.size if side == Side.ASK else -data.size
        self._signed_vol += signed
        self._trades.append((timestamp, signed))

    def _evict(self, now: int) -> None:
        cutoff = now - self.horizon_ns
        while self._trades and self._trades[0][0] < cutoff:
            _, signed = self._trades.popleft()
            self._signed_vol -= signed

    def value(self) -> float:
        return float(self._signed_vol)

    def is_ready(self) -> bool:
        return self._trade_count >= self.warmup_trades

    def reset(self) -> None:
        self._trades.clear()
        self._signed_vol = 0
        self._trade_count = 0
