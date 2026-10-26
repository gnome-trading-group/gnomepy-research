from __future__ import annotations

from typing import Generic, TypeVar

from gnomepy.java.schemas import Mbp10Schema
from gnomepy_research.signals.base import Signal
from gnomepy_research.signals.book.base import BookSignal
from gnomepy_research.signals.fair_value.base import FairValueSignal
from gnomepy_research.signals.flow.base import FlowSignal
from gnomepy_research.signals.market_state.base import MarketStateSignal
from gnomepy_research.signals.volatility.base import VolatilitySignal

T = TypeVar("T", int, float)


class _PerAsset(Signal[T], Generic[T]):
    """Filters update() calls to one stream (a listing's book), forwarding only its ticks to the wrapped signal.

    Use to build cross-asset signals via composition, with each listing's stream from the strategy's directory:

        btc_micro = PerAsset(MicropriceFairValue(), stream_id=self.streams.stream_id(btc_listing_id))
        eth_micro = PerAsset(MicropriceFairValue(), stream_id=self.streams.stream_id(eth_listing_id))
        spread = btc_micro - eth_micro
    """

    def __init__(self, signal: Signal[T], stream_id: int):
        self.signal = signal
        self.stream_id = stream_id

    def update(self, timestamp: int, data: Mbp10Schema) -> None:
        if data.stream_id != self.stream_id:
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


class _PerAssetBook(_PerAsset[float], BookSignal):
    def value(self) -> float:
        return self.signal.value()


class _PerAssetMarketState(_PerAsset[float], MarketStateSignal):
    def value(self) -> float:
        return self.signal.value()


def PerAsset(signal: Signal, stream_id: int) -> Signal:
    """Wrap a signal so it only processes one stream's ticks. Preserves the wrapped signal's type."""
    if isinstance(signal, FairValueSignal):
        return _PerAssetFairValue(signal, stream_id)
    if isinstance(signal, VolatilitySignal):
        return _PerAssetVolatility(signal, stream_id)
    if isinstance(signal, FlowSignal):
        return _PerAssetFlow(signal, stream_id)
    if isinstance(signal, BookSignal):
        return _PerAssetBook(signal, stream_id)
    if isinstance(signal, MarketStateSignal):
        return _PerAssetMarketState(signal, stream_id)
    raise TypeError(
        f"Cannot wrap {type(signal).__name__}. "
        "Expected FairValueSignal, VolatilitySignal, FlowSignal, BookSignal, or MarketStateSignal."
    )
