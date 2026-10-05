"""Infrastructure smoke test: takes a random side every few seconds.

Has no edge and is not meant to make money. It exists to exercise a live or paper
session end to end (market data in, intents out, fills back) at a steady, predictable
order rate. It trades every listing the session subscribes to, each paced and capped
independently; a take that would breach ``max_position`` is flipped to the other side.
"""
from __future__ import annotations

import random

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema


class RandomTrader(Strategy):
    def __init__(
        self,
        interval_seconds: float = 5.0,
        size: int = 1,
        max_position: int = 3,
        seed: int | None = None,
    ):
        self.interval_ns = int(interval_seconds * 1_000_000_000)
        self.size = size
        self.max_position = max_position
        self._rng = random.Random(seed)
        self._last_trade_ns: dict[tuple[int, int], int] = {}

    def on_market_data(self, data: Schema) -> list[Intent]:
        exchange_id, security_id = data.exchange_id, data.security_id

        # Paced on exchange time, not the wall clock, so a backtest replay trades at the same rate.
        now = data.event_timestamp
        last = self._last_trade_ns.get((exchange_id, security_id))
        if last is not None and now - last < self.interval_ns:
            return []
        bid, ask = data.bid_price(0), data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return []
        self._last_trade_ns[(exchange_id, security_id)] = now

        side = self._rng.choice((Side.BID, Side.ASK))
        price = ask if side == Side.BID else bid
        size = self.positions.compliant_size(exchange_id, security_id, self.size, price)
        position = self.positions.get_effective_quantity(exchange_id, security_id)
        if side == Side.BID and position + size > self.max_position:
            side = Side.ASK
        elif side == Side.ASK and position - size < -self.max_position:
            side = Side.BID

        print(f"RandomTrader: {exchange_id}/{security_id} {side.name} {size} at position {position}", flush=True)
        return [
            Intent(
                exchange_id=exchange_id,
                security_id=security_id,
                take_side=side,
                take_size=size,
                take_order_type=OrderType.MARKET,
            )
        ]

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []
