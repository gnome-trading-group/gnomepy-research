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

Options for exercising a strategy ending its own session:

* ``exit_after_seconds``: ask to exit this long after the first market data (exchange
  time); 0 never does.
* ``exit_flatten``: close every position on the way out rather than keep it.
* ``exit_timeout_seconds``: how long closing may take; a short one on a thin book leaves
  a position behind, which the session's exit alert should report.

Every execution report is printed, rejects with their reason, so the session log shows
what the OMS refused and why.

Two metric streams record the same, so a session's journal exercises metrics too:

* ``decisions``: one row per intent, with the touch, the position and what was sent. ``take_side``
  is 1 to buy, -1 to sell and 0 for none; a missing quote side is NaN.
* ``reports``: one row per execution report. ``exec_type`` and ``reject_reason`` are positions
  in gnomepy's ``ExecType`` and ``RejectReason`` enums; ``reject_reason`` is -1 on anything but a reject.

Sizes and positions are in the scaled units orders use.
"""
from __future__ import annotations

import random
from datetime import timedelta
from math import nan

from gnomepy import ExecType, ExecutionReport, Intent, OrderType, RejectReason, Scales, Side, Strategy
from gnomepy.java.schemas import Schema

_TAKE_SIDES = {Side.BID: 1, Side.ASK: -1}
_EXEC_TYPES = {exec_type: code for code, exec_type in enumerate(ExecType)}
_REJECT_REASONS = {reason: code for code, reason in enumerate(RejectReason)}


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
        exit_after_seconds: float = 0.0,
        exit_flatten: bool = False,
        exit_timeout_seconds: float = 300.0,
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
        self.exit_after_ns = int(exit_after_seconds * 1_000_000_000)
        self.exit_flatten = exit_flatten
        self.exit_timeout = timedelta(seconds=exit_timeout_seconds)
        self._rng = random.Random(seed)
        self._last_trade_ns: dict[tuple[int, int], int] = {}
        self._first_ns: int | None = None
        self._decisions = None
        self._reports = None

    def register_metrics(self) -> None:
        decisions = self.metrics.create_stream("decisions")
        self._d_ts = decisions.add_long_column("timestamp")
        self._d_exchange = decisions.add_int_column("exchange_id")
        self._d_security = decisions.add_int_column("security_id")
        self._d_bid = decisions.add_double_column("best_bid")
        self._d_ask = decisions.add_double_column("best_ask")
        self._d_position = decisions.add_long_column("position")
        self._d_take_side = decisions.add_int_column("take_side")
        self._d_take_size = decisions.add_long_column("take_size")
        self._d_quote_bid = decisions.add_double_column("quote_bid")
        self._d_quote_ask = decisions.add_double_column("quote_ask")
        decisions.freeze()
        self._decisions = decisions

        reports = self.metrics.create_stream("reports")
        self._r_ts = reports.add_long_column("timestamp")
        self._r_exchange = reports.add_int_column("exchange_id")
        self._r_security = reports.add_int_column("security_id")
        self._r_client_oid = reports.add_long_column("client_oid")
        self._r_exec_type = reports.add_int_column("exec_type")
        self._r_filled = reports.add_long_column("filled_qty")
        self._r_price = reports.add_double_column("fill_price")
        self._r_leaves = reports.add_long_column("leaves_qty")
        self._r_reject = reports.add_int_column("reject_reason")
        reports.freeze()
        self._reports = reports

    def on_market_data(self, data: Schema) -> list[Intent]:
        exchange_id, security_id = data.exchange_id, data.security_id

        # Paced on exchange time, not the wall clock, so a backtest replay trades at the same rate.
        now = data.event_timestamp
        if self._first_ns is None:
            self._first_ns = now
        if self.exit_after_ns and now - self._first_ns >= self.exit_after_ns:
            print(f"RandomTrader: exiting (flatten={self.exit_flatten})", flush=True)
            self.request_exit(
                f"RandomTrader exit test after {self.exit_after_ns // 1_000_000_000}s",
                flatten=self.exit_flatten,
                timeout=self.exit_timeout,
            )
            return []
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
        self._record_decision(now, exchange_id, security_id, bid, ask, fields)
        return [Intent(exchange_id=exchange_id, security_id=security_id, **fields)]

    def _record_decision(self, now: int, exchange_id: int, security_id: int, bid: int, ask: int, fields: dict) -> None:
        if self._decisions is None:
            return
        stream = self._decisions
        stream.begin()
        stream.set_long(self._d_ts, now)
        stream.set_int(self._d_exchange, exchange_id)
        stream.set_int(self._d_security, security_id)
        stream.set_double(self._d_bid, bid / Scales.PRICE)
        stream.set_double(self._d_ask, ask / Scales.PRICE)
        stream.set_long(self._d_position, self.positions.get_effective_quantity(exchange_id, security_id))
        stream.set_int(self._d_take_side, _TAKE_SIDES.get(fields.get("take_side"), 0))
        stream.set_long(self._d_take_size, fields.get("take_size", 0))
        stream.set_double(self._d_quote_bid, fields["bid_price"] / Scales.PRICE if "bid_price" in fields else nan)
        stream.set_double(self._d_quote_ask, fields["ask_price"] / Scales.PRICE if "ask_price" in fields else nan)
        stream.commit()

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
        if self._reports is not None:
            stream = self._reports
            stream.begin()
            stream.set_long(self._r_ts, report.timestamp_recv)
            stream.set_int(self._r_exchange, report.exchange_id)
            stream.set_int(self._r_security, report.security_id)
            stream.set_long(self._r_client_oid, int(report.client_oid))
            stream.set_int(self._r_exec_type, _EXEC_TYPES[report.exec_type])
            stream.set_long(self._r_filled, report.filled_qty)
            stream.set_double(self._r_price, report.fill_price / Scales.PRICE)
            stream.set_long(self._r_leaves, report.leaves_qty)
            # Reports that aren't rejects carry RejectReason.UNKNOWN rather than nothing.
            rejected = report.exec_type in (ExecType.REJECT, ExecType.CANCEL_REJECT)
            stream.set_int(self._r_reject, _REJECT_REASONS[report.reject_reason] if rejected else -1)
            stream.commit()
        return []


def _to_price(dollars: float) -> int:
    return int(round(dollars * Scales.PRICE))


def _dollars(price: int) -> str:
    return f"${price / Scales.PRICE:.4f}"
