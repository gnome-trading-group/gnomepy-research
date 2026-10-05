"""Infrastructure smoke test: takes a random side every few seconds.

Has no edge and is not meant to make money. It exists to exercise a live or paper
session end to end (market data in, intents out, fills back) at a steady, predictable
order rate. It trades every listing the session subscribes to, each paced and capped
independently; a take that would breach ``max_position`` is flipped to the other side.

Options for exercising risk controls:

* ``quote``: also rest a two-sided quote ``quote_offset`` dollars outside the touch, so
  kills have orders to cancel and open-order limits have orders to count.
* ``trade``: turn the random takes off, to run quote-only and stay flat.
* ``take_type="limit"``: send takes as IOC limits ``take_through`` dollars through the
  touch rather than as market orders, so a price collar can judge them.

Every execution report is printed, rejects with their reason, so the session log shows
what the OMS refused and why.
"""
from __future__ import annotations

import random

from gnomepy import ExecutionReport, Intent, OrderType, Scales, Side, Strategy
from gnomepy.java.schemas import Schema


class RandomTrader(Strategy):
    def __init__(
        self,
        interval_seconds: float = 5.0,
        size: int = 1,
        max_position: int = 3,
        seed: int | None = None,
        trade: bool = True,
        quote: bool = False,
        quote_offset: float = 0.02,
        take_type: str = "market",
        take_through: float = 0.0,
    ):
        if take_type not in ("market", "limit"):
            raise ValueError(f"take_type must be 'market' or 'limit', not {take_type!r}")
        self.interval_ns = int(interval_seconds * 1_000_000_000)
        self.size = size
        self.max_position = max_position
        self.trade = trade
        self.quote = quote
        self.quote_offset = quote_offset
        self.take_type = take_type
        self.take_through = take_through
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

        fields: dict = {}
        if self.quote:
            fields.update(self._quote(exchange_id, security_id, bid, ask))
        if self.trade:
            fields.update(self._take(exchange_id, security_id, bid, ask))
        if not fields:
            return []
        return [Intent(exchange_id=exchange_id, security_id=security_id, **fields)]

    def _quote(self, exchange_id: int, security_id: int, bid: int, ask: int) -> dict:
        offset = _to_price(self.quote_offset)
        bid_price = self.positions.compliant_price(exchange_id, security_id, bid - offset, Side.BID)
        ask_price = self.positions.compliant_price(exchange_id, security_id, ask + offset, Side.ASK)
        if bid_price <= 0:
            return {}
        print(
            f"RandomTrader: {exchange_id}/{security_id} quote {_dollars(bid_price)} / {_dollars(ask_price)}",
            flush=True,
        )
        return {
            "bid_price": bid_price,
            "bid_size": self.positions.compliant_size(exchange_id, security_id, self.size, bid_price),
            "ask_price": ask_price,
            "ask_size": self.positions.compliant_size(exchange_id, security_id, self.size, ask_price),
        }

    def _take(self, exchange_id: int, security_id: int, bid: int, ask: int) -> dict:
        side = self._rng.choice((Side.BID, Side.ASK))
        price = ask if side == Side.BID else bid
        size = self.positions.compliant_size(exchange_id, security_id, self.size, price)
        position = self.positions.get_effective_quantity(exchange_id, security_id)
        if side == Side.BID and position + size > self.max_position:
            side = Side.ASK
        elif side == Side.ASK and position - size < -self.max_position:
            side = Side.BID

        fields: dict = {"take_side": side, "take_size": size}
        if self.take_type == "limit":
            through = _to_price(self.take_through)
            limit = self.positions.compliant_price(
                exchange_id, security_id, ask + through if side == Side.BID else bid - through, side
            )
            fields.update(take_order_type=OrderType.LIMIT, take_limit_price=limit)
            detail = f"limit {_dollars(limit)}"
        else:
            fields.update(take_order_type=OrderType.MARKET)
            detail = "market"
        print(
            f"RandomTrader: {exchange_id}/{security_id} {side.name} {size} {detail} at position {position}",
            flush=True,
        )
        return fields

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        reason = f" reason={report.reject_reason.name}" if report.reject_reason is not None else ""
        print(
            f"RandomTrader: {report.exchange_id}/{report.security_id} order {report.client_oid} "
            f"{report.exec_type.name} filled={report.filled_qty} at {_dollars(report.fill_price)} "
            f"leaves={report.leaves_qty}{reason}",
            flush=True,
        )
        return []


def _to_price(dollars: float) -> int:
    return int(round(dollars * Scales.PRICE))


def _dollars(price: int) -> str:
    return f"${price / Scales.PRICE:.4f}"
