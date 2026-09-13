from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

from gnomepy import Intent, OrderType, Scales, Side, Strategy
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.enums import ExecType
from gnomepy.java.schemas import Schema

Listing = tuple[int, int]  # (exchange_id, security_id)


@dataclass
class TargetEntry:
    qty: int
    price: int = 0


Target = dict[Listing, TargetEntry]


class TargetPortfolioStrategy(Strategy):
    def _init_target_portfolio(
        self,
        tracked_listings: set[Listing],
        *,
        taker_order_type: OrderType = OrderType.MARKET,
        max_staleness_ns: int = 60_000_000_000,
        processing_time_ns: int = 5_000_000,
        imbalance_timeout_ns: int = 30_000_000_000,
        max_unhedged_qty: int = 0,
        debug: bool = False,
    ) -> None:
        self._taker_order_type = taker_order_type
        self._max_staleness_ns = max_staleness_ns
        self._processing_time_ns = processing_time_ns
        self._tracked_listings = tracked_listings
        self._ask_book: dict[Listing, list[tuple[int, int]]] = {}
        self._bid_book: dict[Listing, list[tuple[int, int]]] = {}
        self._last_update_ts: dict[Listing, int] = {}
        self._prev_target_listings: set[Listing] = set()
        self._imbalance_timeout_ns = imbalance_timeout_ns
        self._max_unhedged_qty = max_unhedged_qty
        self._imbalance_since: int | None = None
        self._unwinding: set[Listing] | None = None
        self._debug_mode = debug
        self._prev_intents: dict[Listing, Intent] = {}

    def _debug(self, timestamp: int, event: str, **kwargs) -> None:
        if not self._debug_mode:
            return
        parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"[{timestamp}] DEBUG {event} {parts}")

    def _on_book_updated(
        self, listing: Listing,
        bids: list[tuple[int, int]], asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None:
        pass

    def _on_fill(self, listing: Listing, report: ExecutionReport) -> None:
        pass

    @abstractmethod
    def compute_target(self, timestamp: int) -> Target:
        raise NotImplementedError

    def on_market_data(self, data: Schema) -> list[Intent]:
        listing: Listing = (data.exchange_id, data.security_id)
        timestamp = data.event_timestamp

        ask_levels: list[tuple[int, int]] = []
        bid_levels: list[tuple[int, int]] = []
        for i in range(10):
            ap, az = data.ask_price(i), data.ask_size(i)
            bp, bz = data.bid_price(i), data.bid_size(i)
            if ap > 0 and az > 0:
                ask_levels.append((ap, az))
            if bp > 0 and bz > 0:
                bid_levels.append((bp, bz))
        self._ask_book[listing] = ask_levels
        self._bid_book[listing] = bid_levels
        self._last_update_ts[listing] = timestamp
        self._on_book_updated(listing, bid_levels, ask_levels, timestamp)

        if listing not in self._tracked_listings:
            return []

        target = self.compute_target(timestamp)
        return self._generate_intents(target, timestamp)

    def _intent_key(self, intent: Intent | None) -> tuple:
        if intent is None:
            return ()
        take_size = intent.take_size
        return (
            intent.bid_price, intent.bid_size,
            intent.ask_price, intent.ask_size,
            intent.take_side if take_size > 0 else None, take_size,
            intent.take_order_type, intent.take_limit_price,
        )

    def _generate_intents(self, target: Target, timestamp: int) -> list[Intent]:
        target = self._apply_group_protections(target, timestamp)

        current_intents: dict[Listing, Intent] = {}
        active = set(target.keys()) | self._prev_target_listings

        for lst in active:
            entry = target.get(lst)
            if entry is None:
                self._debug(timestamp, "INTENT_CANCEL", eid=lst[0], sid=lst[1])
                current_intents[lst] = self._cancel_intent(lst)
                continue

            actual = self._net_qty(lst)
            delta = entry.qty - actual

            if delta == 0:
                self._debug(timestamp, "INTENT_CANCEL", eid=lst[0], sid=lst[1], reason="at_target")
                current_intents[lst] = self._cancel_intent(lst)
                continue

            if self._is_stale(lst, timestamp):
                self._debug(timestamp, "INTENT_CANCEL", eid=lst[0], sid=lst[1], reason="stale")
                current_intents[lst] = self._cancel_intent(lst)
                continue

            if entry.price > 0:
                size = self.positions.compliant_size(lst[0], lst[1], abs(delta), entry.price)
                if delta > 0:
                    self._debug(
                        timestamp, "INTENT_MAKER",
                        eid=lst[0], sid=lst[1],
                        bid_price=round(entry.price / Scales.PRICE, 4), bid_size=round(abs(delta) / Scales.SIZE, 4),
                    )
                    current_intents[lst] = Intent(
                        exchange_id=lst[0], security_id=lst[1],
                        bid_price=entry.price, bid_size=size,
                        ask_price=0, ask_size=0,
                    )
                else:
                    self._debug(
                        timestamp, "INTENT_MAKER",
                        eid=lst[0], sid=lst[1],
                        ask_price=round(entry.price / Scales.PRICE, 4), ask_size=round(abs(delta) / Scales.SIZE, 4),
                    )
                    current_intents[lst] = Intent(
                        exchange_id=lst[0], security_id=lst[1],
                        bid_price=0, bid_size=0,
                        ask_price=entry.price, ask_size=size,
                    )
            else:
                effective = self._effective_qty(lst)
                if effective != actual:
                    self._debug(timestamp, "INTENT_CANCEL", eid=lst[0], sid=lst[1], reason="pending_effective")
                    current_intents[lst] = self._cancel_intent(lst)
                    continue
                take_side = Side.BID if delta > 0 else Side.ASK
                price = self._estimate_fill_price(lst, abs(delta), buy=(delta > 0))
                size = self.positions.compliant_size(lst[0], lst[1], abs(delta), price)
                intent_kwargs: dict = dict(
                    exchange_id=lst[0], security_id=lst[1],
                    take_side=take_side, take_size=size,
                    take_order_type=self._taker_order_type,
                )
                if self._taker_order_type == OrderType.LIMIT:
                    intent_kwargs["take_limit_price"] = self._worst_fill_price(
                        lst, abs(delta), buy=(delta > 0)
                    )
                self._debug(
                    timestamp, "INTENT_TAKER",
                    eid=lst[0], sid=lst[1], side=take_side.name,
                    qty=round(abs(delta) / Scales.SIZE, 4),
                    price=round(price / Scales.PRICE, 4),
                    order_type=self._taker_order_type.name,
                )
                current_intents[lst] = Intent(**intent_kwargs)

        intents = [
            intent for lst, intent in current_intents.items()
            if self._intent_key(intent) != self._intent_key(self._prev_intents.get(lst))
        ]
        self._prev_intents = current_intents
        self._prev_target_listings = set(target.keys())
        return intents

    def _worst_fill_price(self, listing: Listing, qty: int, buy: bool) -> int:
        book = (self._ask_book if buy else self._bid_book).get(listing, [])
        if not book:
            return 0
        remaining = qty
        last_price = book[0][0]
        for price_raw, size in book:
            last_price = price_raw
            remaining -= size
            if remaining <= 0:
                break
        return last_price

    def _estimate_fill_price(self, listing: Listing, qty: int, buy: bool) -> int:
        book = (self._ask_book if buy else self._bid_book).get(listing, [])
        if not book:
            return 0
        total_cost = 0
        remaining = qty
        for price_raw, size in book:
            chunk = min(remaining, size)
            total_cost += price_raw * chunk
            remaining -= chunk
            if remaining <= 0:
                break
        filled = qty - remaining
        if filled <= 0:
            return book[0][0]
        return total_cost // filled

    def _is_unwinding(self) -> bool:
        return self._unwinding is not None

    def _check_unwind_complete(self) -> bool:
        if self._unwinding is None:
            return False
        if all(self._net_qty(lst) == 0 for lst in self._unwinding):
            self._unwinding = None
            return True
        return False

    def _apply_group_protections(self, target: Target, timestamp: int) -> Target:
        overrides: dict[Listing, TargetEntry] = {}

        # Resume existing unwind
        if self._unwinding is not None:
            if not self._check_unwind_complete():
                for lst in self._unwinding:
                    overrides[lst] = TargetEntry(qty=0)

        listings = [lst for lst in target if lst not in self._unwinding_listings()]
        if not listings:
            if not overrides:
                return target
            result = dict(target)
            result.update(overrides)
            return result

        actuals = {lst: self._net_qty(lst) for lst in listings}
        targets = {lst: target[lst].qty for lst in listings}
        deltas = {lst: targets[lst] - actuals[lst] for lst in listings}

        # Layer 4: Overfill detection
        if any(targets[lst] > 0 and actuals[lst] > targets[lst] + Scales.SIZE for lst in listings):
            overfill_lst = next(l for l in listings if targets[l] > 0 and actuals[l] > targets[l] + Scales.SIZE)
            self._debug(timestamp, "PROTECTION_OVERFILL",
                        eid=overfill_lst[0], sid=overfill_lst[1],
                        actual=actuals[overfill_lst], target=targets[overfill_lst])
            print(f"[{timestamp}] LEG_RISK overfill")
            unwind = self.on_imbalance_timeout(timestamp)
            self._unwinding = {lst for lst in listings if self._net_qty(lst) != 0}
            self._imbalance_since = None
            for lst in listings:
                overrides[lst] = TargetEntry(qty=0) if unwind is None else unwind.get(lst, TargetEntry(qty=0))
            result = dict(target)
            result.update(overrides)
            return result

        # Layer 3: Circuit breaker — only fires when there are no active entries, to avoid
        # false positives during EXIT_ENTER where one taker leg settles before the other.
        any_active_entries = any(targets[lst] > 0 for lst in listings)
        if self._max_unhedged_qty > 0 and not any_active_entries:
            exit_qtys = [abs(actuals[lst]) for lst in listings if targets[lst] == 0]
            imbalance = (max(exit_qtys) - min(exit_qtys)) if len(exit_qtys) >= 2 else 0
            if imbalance > self._max_unhedged_qty * Scales.SIZE:
                self._debug(timestamp, "PROTECTION_CIRCUIT_BREAKER",
                            imbalance=imbalance, max=self._max_unhedged_qty)
                print(f"[{timestamp}] LEG_RISK circuit_breaker imbalance={imbalance} max={self._max_unhedged_qty}")
                unwind = self.on_imbalance_timeout(timestamp)
                self._unwinding = {lst for lst in listings if self._net_qty(lst) != 0}
                self._imbalance_since = None
                for lst in listings:
                    overrides[lst] = TargetEntry(qty=0) if unwind is None else unwind.get(lst, TargetEntry(qty=0))
                result = dict(target)
                result.update(overrides)
                return result

        # Layer 2: Imbalance timeout
        any_at_target = any(deltas[lst] == 0 and targets[lst] > 0 for lst in listings)
        any_not_at_target = any(deltas[lst] != 0 and targets[lst] > 0 for lst in listings)
        any_has_progress = any(actuals[lst] > 0 and targets[lst] > 0 and deltas[lst] > 0 for lst in listings)
        any_no_progress = any(actuals[lst] == 0 and targets[lst] > 0 for lst in listings)
        group_imbalanced = (any_at_target and any_not_at_target) or (any_has_progress and any_no_progress)

        if group_imbalanced:
            if self._imbalance_since is None:
                self._imbalance_since = timestamp
            else:
                elapsed = timestamp - self._imbalance_since
                self._debug(timestamp, "PROTECTION_IMBALANCE",
                            elapsed_ns=elapsed, timeout_ns=self._imbalance_timeout_ns)
                if elapsed > self._imbalance_timeout_ns:
                    print(f"[{timestamp}] LEG_RISK imbalance_timeout")
                    unwind = self.on_imbalance_timeout(timestamp)
                    self._unwinding = {lst for lst in listings if self._net_qty(lst) != 0}
                    self._imbalance_since = None
                    for lst in listings:
                        overrides[lst] = TargetEntry(qty=0) if unwind is None else unwind.get(lst, TargetEntry(qty=0))
                    result = dict(target)
                    result.update(overrides)
                    return result
        else:
            self._imbalance_since = None

        # Layer 1: Depth gate
        has_pending_delta = any(deltas[lst] != 0 for lst in listings)
        if has_pending_delta:
            depth_ok = True
            for lst in listings:
                d = deltas[lst]
                if d > 0 and not self._ask_book.get(lst):
                    self._debug(timestamp, "PROTECTION_DEPTH_GATE",
                                eid=lst[0], sid=lst[1], side="ask", depth=0)
                    depth_ok = False
                    break
                if d < 0 and not self._bid_book.get(lst):
                    self._debug(timestamp, "PROTECTION_DEPTH_GATE",
                                eid=lst[0], sid=lst[1], side="bid", depth=0)
                    depth_ok = False
                    break
            if not depth_ok:
                for lst in listings:
                    overrides[lst] = TargetEntry(qty=actuals[lst])

        if not overrides:
            return target

        result = dict(target)
        result.update(overrides)
        return result

    def _unwinding_listings(self) -> set[Listing]:
        return self._unwinding if self._unwinding is not None else set()

    def on_imbalance_timeout(self, timestamp: int) -> Target | None:
        return None

    def _net_qty(self, listing: Listing) -> int:
        pos = self.positions.get_position(listing[0], listing[1])
        return pos.net_quantity if pos is not None else 0

    def _effective_qty(self, listing: Listing) -> int:
        return self.positions.get_effective_quantity(listing[0], listing[1])

    def _is_stale(self, listing: Listing, timestamp: int) -> bool:
        last = self._last_update_ts.get(listing, 0)
        return timestamp - last > self._max_staleness_ns

    def _cancel_intent(self, lst: Listing) -> Intent:
        return Intent(
            exchange_id=lst[0], security_id=lst[1],
            bid_price=0, bid_size=0, ask_price=0, ask_size=0,
        )

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        if report.exec_type not in (ExecType.FILL, ExecType.PARTIAL_FILL):
            return []
        if report.filled_qty <= 0:
            return []

        lst: Listing = (report.exchange_id, report.security_id)
        self._on_fill(lst, report)

        if self._unwinding is None:
            return []
        if self._check_unwind_complete():
            return []

        target: Target = {l: TargetEntry(qty=0) for l in self._unwinding}
        return self._generate_intents(target, report.timestamp_recv)
