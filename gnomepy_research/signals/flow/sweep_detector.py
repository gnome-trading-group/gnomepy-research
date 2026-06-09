from __future__ import annotations

from gnomepy.java.enums import Side
from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.flow.base import FlowSignal, is_trade_event


class SweepDetector(FlowSignal):
    """Detects multi-level sweeps: consecutive same-side trades spanning multiple prices.

    A sweep is detected when >= min_levels consecutive trades occur on the same side
    within window_ns and the first/last trade prices differ (indicating multiple
    price levels were consumed). Outputs the signed total swept volume when a
    sweep is active, 0.0 otherwise.

    Positive = buy sweep; negative = sell sweep.

    Args:
        window_ns: Max elapsed time between consecutive trades in a sweep.
        min_levels: Minimum number of consecutive same-side trades to qualify.
        warmup_trades: Minimum trades before is_ready() returns True.
    """

    def __init__(
        self,
        window_ns: int = 10_000_000,
        min_levels: int = 2,
        warmup_trades: int = 20,
    ):
        self.window_ns = window_ns
        self.min_levels = min_levels
        self.warmup_trades = warmup_trades

        self._streak_side = Side.NONE
        self._streak_count = 0
        self._streak_start_ns = 0
        self._streak_first_price = 0
        self._streak_last_price = 0
        self._streak_volume = 0
        self._trade_count = 0
        self._sweep_volume = 0.0

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if not is_trade_event(data):
            return

        self._trade_count += 1
        side = data.side
        if side == Side.NONE:
            return

        if (side == self._streak_side
                and timestamp - self._streak_start_ns <= self.window_ns):
            self._streak_count += 1
            self._streak_last_price = data.price
            self._streak_volume += data.size
        else:
            self._streak_side = side
            self._streak_count = 1
            self._streak_start_ns = timestamp
            self._streak_first_price = data.price
            self._streak_last_price = data.price
            self._streak_volume = data.size

        if (self._streak_count >= self.min_levels
                and self._streak_first_price != self._streak_last_price):
            vol = float(self._streak_volume)
            self._sweep_volume = vol if self._streak_side == Side.ASK else -vol
        else:
            self._sweep_volume = 0.0

    def value(self) -> float:
        return self._sweep_volume

    def is_ready(self) -> bool:
        return self._trade_count >= self.warmup_trades

    def reset(self) -> None:
        self._streak_side = Side.NONE
        self._streak_count = 0
        self._streak_start_ns = 0
        self._streak_first_price = 0
        self._streak_last_price = 0
        self._streak_volume = 0
        self._trade_count = 0
        self._sweep_volume = 0.0
