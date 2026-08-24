from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

from gnomepy import Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema

Listing = tuple[int, int]  # (exchange_id, security_id)


@dataclass
class TargetEntry:
    qty: int
    use_taker: bool = False
    group_id: str | None = None


Target = dict[Listing, TargetEntry]


class TargetPortfolioStrategy(Strategy):
    def _init_target_portfolio(
        self,
        tracked_listings: set[Listing],
        *,
        maker_orders: bool = False,
        taker_order_type: OrderType = OrderType.MARKET,
        max_staleness_ns: int = 5_000_000_000,
        processing_time_ns: int = 5_000_000,
        size: int = 1_000_000,
        imbalance_timeout_ns: int = 30_000_000_000,
        max_unhedged_qty: int = 0,
    ) -> None:
        self._maker_orders = maker_orders
        self._taker_order_type = taker_order_type
        self._max_staleness_ns = max_staleness_ns
        self._processing_time_ns = processing_time_ns
        self._size = size
        self._tracked_listings = tracked_listings
        self._best_bid: dict[Listing, int] = {}
        self._best_ask: dict[Listing, int] = {}
        self._best_bid_size: dict[Listing, float] = {}
        self._best_ask_size: dict[Listing, float] = {}
        self._ask_book: dict[Listing, list[tuple[int, float]]] = {}
        self._bid_book: dict[Listing, list[tuple[int, float]]] = {}
        self._last_update_ts: dict[Listing, int] = {}
        self._prev_target_listings: set[Listing] = set()
        self._imbalance_timeout_ns = imbalance_timeout_ns
        self._max_unhedged_qty = max_unhedged_qty
        self._group_imbalance_since: dict[str, int] = {}
        self._unwinding_groups: dict[str, set[Listing]] = {}

    @abstractmethod
    def compute_target(self, timestamp: int) -> Target:
        raise NotImplementedError

    def on_market_data(self, data: Schema) -> list[Intent]:
        listing: Listing = (data.exchange_id, data.security_id)
        timestamp = data.event_timestamp

        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid > 0:
            self._best_bid[listing] = bid
            self._best_bid_size[listing] = data.bid_size(0) / self._size
        else:
            self._best_bid.pop(listing, None)
            self._best_bid_size.pop(listing, None)
        if ask > 0:
            self._best_ask[listing] = ask
            self._best_ask_size[listing] = data.ask_size(0) / self._size
        else:
            self._best_ask.pop(listing, None)
            self._best_ask_size.pop(listing, None)
        ask_levels: list[tuple[int, float]] = []
        bid_levels: list[tuple[int, float]] = []
        for i in range(10):
            ap, az = data.ask_price(i), data.ask_size(i)
            bp, bz = data.bid_price(i), data.bid_size(i)
            if ap > 0 and az > 0:
                ask_levels.append((ap, az / self._size))
            if bp > 0 and bz > 0:
                bid_levels.append((bp, bz / self._size))
        self._ask_book[listing] = ask_levels
        self._bid_book[listing] = bid_levels
        self._last_update_ts[listing] = timestamp

        if listing not in self._tracked_listings:
            return []

        target = self.compute_target(timestamp)
        return self._generate_intents(target, timestamp)

    def _generate_intents(self, target: Target, timestamp: int) -> list[Intent]:
        # Apply group-level protections before generating per-listing intents.
        target = self._apply_group_protections(target, timestamp)

        intents: list[Intent] = []
        active = set(target.keys()) | self._prev_target_listings

        for lst in active:
            entry = target.get(lst)
            if entry is None:
                intents.append(self._cancel_intent(lst))
                continue

            actual = self._net_qty(lst)
            delta = entry.qty - actual

            if delta == 0:
                intents.append(self._cancel_intent(lst))
                continue

            if self._is_stale(lst, timestamp):
                continue

            if entry.use_taker or not self._maker_orders:
                effective = self._effective_qty(lst)
                if effective != actual:
                    intents.append(self._cancel_intent(lst))
                    continue
                take_side = Side.BID if delta > 0 else Side.ASK
                qty_contracts = abs(delta) / self._size
                price = self._estimate_fill_price(lst, qty_contracts, buy=(delta > 0))
                size = self.positions.compliant_size(lst[0], lst[1], abs(delta), price)
                intent_kwargs: dict = dict(
                    exchange_id=lst[0],
                    security_id=lst[1],
                    take_side=take_side,
                    take_size=size,
                    take_order_type=self._taker_order_type,
                )
                if self._taker_order_type == OrderType.LIMIT:
                    intent_kwargs["take_limit_price"] = self._worst_fill_price(
                        lst, qty_contracts, buy=(delta > 0)
                    )
                intents.append(Intent(**intent_kwargs))
            else:
                if delta > 0:
                    price = self._best_bid.get(lst, 0)
                    if price > 0:
                        size = self.positions.compliant_size(lst[0], lst[1], abs(delta), price)
                        intents.append(Intent(
                            exchange_id=lst[0],
                            security_id=lst[1],
                            bid_price=price,
                            bid_size=size,
                            ask_price=0,
                            ask_size=0,
                        ))
                else:
                    price = self._best_ask.get(lst, 0)
                    if price > 0:
                        size = self.positions.compliant_size(lst[0], lst[1], abs(delta), price)
                        intents.append(Intent(
                            exchange_id=lst[0],
                            security_id=lst[1],
                            bid_price=0,
                            bid_size=0,
                            ask_price=price,
                            ask_size=size,
                        ))

        self._prev_target_listings = set(target.keys())
        return intents

    def _worst_fill_price(self, listing: Listing, qty_contracts: float, buy: bool) -> int:
        book = (self._ask_book if buy else self._bid_book).get(listing, [])
        if not book:
            return self._best_ask.get(listing, 0) if buy else self._best_bid.get(listing, 0)
        remaining = qty_contracts
        last_price = book[0][0]
        for price_raw, size in book:
            last_price = price_raw
            remaining -= size
            if remaining <= 0:
                break
        return last_price

    def _estimate_fill_price(self, listing: Listing, qty_contracts: float, buy: bool) -> int:
        book = (self._ask_book if buy else self._bid_book).get(listing, [])
        if not book:
            return self._best_ask.get(listing, 0) if buy else self._best_bid.get(listing, 0)
        total_cost = 0.0
        remaining = qty_contracts
        for price_raw, size in book:
            chunk = min(remaining, size)
            total_cost += (price_raw / 1_000_000_000) * chunk
            remaining -= chunk
            if remaining <= 0:
                break
        filled = qty_contracts - remaining
        if filled <= 0:
            return book[0][0]
        return int(total_cost / filled * 1_000_000_000)

    def _is_unwinding(self, group_id: str) -> bool:
        return group_id in self._unwinding_groups

    def _apply_group_protections(self, target: Target, timestamp: int) -> Target:
        overrides: dict[Listing, TargetEntry] = {}

        # Persist unwind intent for groups that have been force-unwound, overriding
        # whatever compute_target returned. Cleared only once all positions are flat.
        for group_id, listings in list(self._unwinding_groups.items()):
            if all(self._net_qty(lst) == 0 for lst in listings):
                del self._unwinding_groups[group_id]
            else:
                for lst in listings:
                    overrides[lst] = TargetEntry(qty=0, use_taker=True, group_id=group_id)

        # Build group -> listings map from current target, skipping groups already unwinding.
        groups: dict[str, list[Listing]] = {}
        for lst, entry in target.items():
            if entry.group_id is not None and entry.group_id not in self._unwinding_groups:
                groups.setdefault(entry.group_id, []).append(lst)

        if not groups:
            if not overrides:
                return target
            result = dict(target)
            result.update(overrides)
            return result

        for group_id, listings in groups.items():
            actuals = {lst: self._net_qty(lst) for lst in listings}
            targets = {lst: target[lst].qty for lst in listings}
            deltas = {lst: targets[lst] - actuals[lst] for lst in listings}

            # Layer 4: Overfill detection — any entry leg where actual > target by more
            # than 1 contract indicates a stuck partial fill from a pairing transition.
            # Trigger immediate unwind to prevent SCALE_UP from treating it as "at target".
            if any(targets[lst] > 0 and actuals[lst] > targets[lst] + self._size for lst in listings):
                print(f"[{timestamp}] LEG_RISK overfill group={group_id}")
                unwind = self.on_imbalance_timeout(group_id, timestamp)
                unwind_listings = {lst for lst in listings if self._net_qty(lst) != 0}
                self._unwinding_groups[group_id] = unwind_listings
                for lst in listings:
                    overrides[lst] = TargetEntry(
                        qty=0,
                        use_taker=True,
                        group_id=group_id,
                    ) if unwind is None else unwind.get(lst, TargetEntry(qty=0, use_taker=True, group_id=group_id))
                self._group_imbalance_since.pop(group_id, None)
                continue

            # Layer 3: Circuit breaker — hard cap on unhedged qty.
            # Only examine exit legs (target=0): entry imbalances are handled by
            # the timeout (Layer 2), and excluding entry legs avoids false positives
            # during pairing transitions where old-filled legs mix with new-empty legs.
            # Imbalance is in display units (divided by size) to match max_unhedged_qty.
            if self._max_unhedged_qty > 0:
                exit_qtys = [abs(actuals[lst]) for lst in listings if targets[lst] == 0]
                imbalance = (max(exit_qtys) - min(exit_qtys)) // self._size if len(exit_qtys) >= 2 else 0
                if imbalance > self._max_unhedged_qty:
                    print(f"[{timestamp}] LEG_RISK circuit_breaker group={group_id} imbalance={imbalance} max={self._max_unhedged_qty}")
                    unwind = self.on_imbalance_timeout(group_id, timestamp)
                    unwind_listings = {lst for lst in listings if self._net_qty(lst) != 0}
                    self._unwinding_groups[group_id] = unwind_listings
                    for lst in listings:
                        overrides[lst] = TargetEntry(
                            qty=0,
                            use_taker=True,
                            group_id=group_id,
                        ) if unwind is None else unwind.get(lst, TargetEntry(qty=0, use_taker=True, group_id=group_id))
                    self._group_imbalance_since.pop(group_id, None)
                    continue

            # Layer 2: Imbalance timeout — detect partial fill and unwind after timeout.
            any_at_target = any(deltas[lst] == 0 and targets[lst] > 0 for lst in listings)
            any_not_at_target = any(deltas[lst] != 0 and targets[lst] > 0 for lst in listings)
            group_imbalanced = any_at_target and any_not_at_target

            if group_imbalanced:
                if group_id not in self._group_imbalance_since:
                    self._group_imbalance_since[group_id] = timestamp
                elif timestamp - self._group_imbalance_since[group_id] > self._imbalance_timeout_ns:
                    print(f"[{timestamp}] LEG_RISK imbalance_timeout group={group_id}")
                    unwind = self.on_imbalance_timeout(group_id, timestamp)
                    unwind_listings = {lst for lst in listings if self._net_qty(lst) != 0}
                    self._unwinding_groups[group_id] = unwind_listings
                    for lst in listings:
                        overrides[lst] = TargetEntry(
                            qty=0,
                            use_taker=True,
                            group_id=group_id,
                        ) if unwind is None else unwind.get(lst, TargetEntry(qty=0, use_taker=True, group_id=group_id))
                    self._group_imbalance_since.pop(group_id, None)
                    continue
            else:
                self._group_imbalance_since.pop(group_id, None)

            # Layer 1: Depth gate — suppress the whole group if any leg lacks depth.
            has_pending_delta = any(deltas[lst] != 0 for lst in listings)
            if has_pending_delta:
                depth_ok = True
                for lst in listings:
                    d = deltas[lst]
                    if d > 0 and self._best_ask_size.get(lst, 0) <= 0:
                        depth_ok = False
                        break
                    if d < 0 and self._best_bid_size.get(lst, 0) <= 0:
                        depth_ok = False
                        break
                if not depth_ok:
                    for lst in listings:
                        overrides[lst] = TargetEntry(
                            qty=actuals[lst],
                            use_taker=False,
                            group_id=group_id,
                        )

        if not overrides:
            return target

        result = dict(target)
        result.update(overrides)
        return result

    def on_imbalance_timeout(self, group_id: str, timestamp: int) -> Target | None:
        """Called when a group has been imbalanced beyond imbalance_timeout_ns.

        Return a Target to override, or None to use the default (unwind all legs to 0).
        """
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

    def on_execution_report(self, report) -> list[Intent]:
        return []
