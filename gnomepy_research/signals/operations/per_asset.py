from __future__ import annotations

from typing import Generic, TypeVar

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.base import Signal
from gnomepy_research.signals.fair_value.base import FairValueSignal
from gnomepy_research.signals.flow.base import FlowSignal
from gnomepy_research.signals.volatility.base import VolatilitySignal

T = TypeVar("T", int, float)


class _PerAsset(Signal[T], Generic[T]):
    """Filters update() calls by security_id (and optionally exchange_id),
    forwarding only matching ticks to the wrapped signal.

    Use to build cross-asset signals via composition:

        btc_micro = PerAsset(MicropriceFairValue(), security_id=1)
        eth_micro = PerAsset(MicropriceFairValue(), security_id=2)
        spread = btc_micro - eth_micro
    """

    def __init__(self, signal: Signal[T], security_id: int, exchange_id: int | None = None):
        self.signal = signal
        self.security_id = security_id
        self.exchange_id = exchange_id

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if data.security_id != self.security_id:
            return
        if self.exchange_id is not None and data.exchange_id != self.exchange_id:
            return
        self.signal.update(timestamp, data)

    def is_ready(self) -> bool:
        return self.signal.is_ready()

    def reset(self) -> None:
        self.signal.reset()


class _PerAssetFairValue(_PerAsset[int], FairValueSignal):
    def value(self) -> int:
        return self.signal.value()


class _PerAssetVolatility(_PerAsset[float], VolatilitySignal):
    def value(self) -> float:
        return self.signal.value()


class _PerAssetFlow(_PerAsset[float], FlowSignal):
    def value(self) -> float:
        return self.signal.value()


def PerAsset(signal: Signal, security_id: int, exchange_id: int | None = None) -> Signal:
    """Wrap a signal so it only processes ticks matching security_id /
    exchange_id. Preserves the wrapped signal's type.
    """
    if isinstance(signal, FairValueSignal):
        return _PerAssetFairValue(signal, security_id, exchange_id)
    if isinstance(signal, VolatilitySignal):
        return _PerAssetVolatility(signal, security_id, exchange_id)
    if isinstance(signal, FlowSignal):
        return _PerAssetFlow(signal, security_id, exchange_id)
    raise TypeError(
        f"Cannot wrap {type(signal).__name__}. "
        "Expected FairValueSignal, VolatilitySignal, or FlowSignal."
    )
