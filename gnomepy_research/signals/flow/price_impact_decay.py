from __future__ import annotations

from collections import deque

from gnomepy.java.enums import Side
from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class PriceImpactDecay(FlowSignal):
    """Fraction of trade price impact remaining after observation_ns.

    For each trade, records the pre-trade mid and trade price. After
    observation_ns have elapsed, measures how much of the initial price
    displacement (trade_price - pre_mid) remains. EWMA-smoothed.

    Output in [0, 1]:
    - Near 0 = full mean-reversion (noise-driven trade)
    - Near 1 = permanent displacement (informed trade)

    Args:
        observation_ns: Horizon over which to measure impact decay.
        alpha: EWMA smoothing for the output.
        warmup_trades: Minimum completed observations before is_ready().
    """

    def __init__(
        self,
        observation_ns: int = 1_000_000_000,
        alpha: float = 0.95,
        warmup_trades: int = 20,
    ):
        self.observation_ns = observation_ns
        self.alpha = alpha
        self.warmup_trades = warmup_trades

        self._pending: deque[tuple[int, int, int]] = deque()  # (expire_ts, pre_mid, trade_price)
        self._prev_mid = 0
        self._ewma_decay = 0.5
        self._completed = 0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return
        mid = (bid + ask) // 2

        while self._pending and self._pending[0][0] <= timestamp:
            _, pre_mid, trade_price = self._pending.popleft()
            initial_disp = trade_price - pre_mid
            if initial_disp != 0:
                remaining = mid - pre_mid
                decay = remaining / initial_disp
                if decay < 0.0:
                    decay = 0.0
                elif decay > 1.0:
                    decay = 1.0
                self._ewma_decay = self.alpha * self._ewma_decay + (1.0 - self.alpha) * decay
                self._completed += 1

        if is_trade_event(data) and data.side != Side.NONE and self._prev_mid > 0:
            expire_ts = timestamp + self.observation_ns
            self._pending.append((expire_ts, self._prev_mid, data.price))

        self._prev_mid = mid

    def value(self) -> float:
        return self._ewma_decay

    def is_ready(self) -> bool:
        return self._completed >= self.warmup_trades

    def reset(self) -> None:
        self._pending.clear()
        self._prev_mid = 0
        self._ewma_decay = 0.5
        self._completed = 0
