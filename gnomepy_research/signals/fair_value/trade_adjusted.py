from __future__ import annotations

from collections import deque

from gnomepy.java.enums import Side
from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.fair_value.base import FairValueSignal
from gnomepy_research.signals.flow.base import is_trade_event


class TradeAdjustedFairValue(FairValueSignal):
    """Fair value: microprice adjusted by recent signed trade flow.

    fair_value = microprice + flow_weight * net_signed_volume

    Incorporates information from both the current book state (microprice)
    and recent trade direction.

    Args:
        flow_weight: Raw price units per net lot (tunable).
        flow_horizon_ns: Rolling trade flow window in nanoseconds.
        warmup: Ticks before is_ready() returns True.
    """

    def __init__(
        self,
        flow_weight: float,
        flow_horizon_ns: int = 1_000_000_000,
        warmup: int = 10,
    ):
        self.flow_weight = flow_weight
        self.flow_horizon_ns = flow_horizon_ns
        self.warmup = warmup

        self._trades: deque[tuple[int, int]] = deque()  # (ts, signed_size)
        self._signed_vol = 0
        self._microprice = 0.0
        self._count = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        bid_sz = data.bid_size(0)
        ask_sz = data.ask_size(0)
        total = bid_sz + ask_sz
        if bid > 0 and ask > 0 and total > 0:
            self._microprice = (bid * ask_sz + ask * bid_sz) / total
            self._count += 1

        if is_trade_event(data) and data.side != Side.NONE:
            self._evict(timestamp)
            signed = data.size if data.side == Side.ASK else -data.size
            self._signed_vol += signed
            self._trades.append((timestamp, signed))

    def _evict(self, now: int) -> None:
        cutoff = now - self.flow_horizon_ns
        while self._trades and self._trades[0][0] < cutoff:
            _, signed = self._trades.popleft()
            self._signed_vol -= signed

    def value(self) -> int:
        return int(self._microprice + self.flow_weight * self._signed_vol)

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._trades.clear()
        self._signed_vol = 0
        self._microprice = 0.0
        self._count = 0
