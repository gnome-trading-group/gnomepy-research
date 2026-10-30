from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import exp, nan

from gnomepy import Scales, Side
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.enums import OrderType
from gnomepy.registry import RegistryClient

from gnomepy_research.fees import registry_fee_rates
from gnomepy_research.arb import (
    AggressivePriceModel,
    ArbBudgetConstraint,
    ArbContext,
    ArbLeg,
    ArbPhase,
    ArbPortfolio,
    CompositeCostModel,
    CostModel,
    DepthCoverageCost,
    FeeCost,
    FillProbTargetModel,
    FillRiskCost,
    JoinBestBidModel,
    OptimalEVModel,
    OrderMode,
    PriceModel,
    VenuePairing,
    discover_group,
    walk_books,
)
from gnomepy_research.sessions.equivalent_event_arb.target_portfolio import (
    Listing,
    Target,
    TargetEntry,
    TargetPortfolioStrategy,
)


class Decision(IntEnum):
    ENTRY = 0
    SCALE_UP = 1
    EXIT = 2
    ENTRY_CANCEL = 3
    FLAT = 4
    LEG_RISK_UNWIND = 5


class EquivalentEventArb(TargetPortfolioStrategy):
    def __init__(
        self,
        event_ids: list[int],
        max_position: int = 0,
        min_pure_arb_bps: float = 5.0,
        taker_edge_threshold_bps: float = 20.0,
        use_taker_orders: bool = True,
        exit_edge_threshold_bps: float = 5.0,
        taker_order_type: str = "LIMIT",
        max_staleness_ns: int = 60_000_000_000,
        processing_time_ns: int = 5_000_000,
        allow_scaling: bool = False,
        allow_early_exit: bool = True,
        imbalance_timeout_ns: int = 30_000_000_000,
        max_unhedged_qty: int = 0,
        fill_risk_lambda: float = 0.0,
        unwind_spread_mult: float = 2.0,
        depth_coverage_mult: float = 0.0,
        price_window_cents: int = 2,
        price_model: str = "aggressive",
        price_improve_bps: float = 50.0,
        price_edge_share: float = 0.5,
        target_fill_prob: float = 0.7,
        optimal_ev_lambda: float = 30.0,
        maker_reprice_cooldown_ns: int = 0,
        cancel_grace_ns: int = 0,
        allow_flagged: bool = False,
        debug: bool = False,
    ):
        self.max_position = max_position * Scales.SIZE
        self.min_pure_arb_bps = min_pure_arb_bps
        self._taker_edge_threshold_bps = taker_edge_threshold_bps
        self._use_taker_orders = use_taker_orders
        self._exit_edge_threshold_bps = exit_edge_threshold_bps
        self._allow_scaling = allow_scaling
        self._allow_early_exit = allow_early_exit

        registry = RegistryClient()
        self._group = discover_group(event_ids, registry, allow_flagged=allow_flagged)

        # The rates each listing's fills are charged, from the registry at the strategy's time.
        self._fee_rates = registry_fee_rates(self)
        maker_components = [FeeCost(self._fee_rates, maker=True)]
        self._fill_risk: FillRiskCost | None = None
        if fill_risk_lambda > 0:
            self._fill_risk = FillRiskCost(fill_risk_lambda, unwind_spread_mult)
            maker_components.append(self._fill_risk)

        taker_components = [FeeCost(self._fee_rates, maker=False)]
        self._depth_coverage: DepthCoverageCost | None = None
        if depth_coverage_mult > 0:
            self._depth_coverage = DepthCoverageCost(depth_coverage_mult, price_window_cents)
            taker_components.append(self._depth_coverage)

        self._unwind_spread_mult = unwind_spread_mult
        self._cost_model: CostModel = CompositeCostModel(
            maker=maker_components,
            taker=taker_components,
        )

        self._stuck_cost_lambda: float | None = None
        if price_model == "join_best_bid":
            self._price_model: PriceModel = JoinBestBidModel(
                fill_risk_lambda=fill_risk_lambda if fill_risk_lambda > 0 else 10.0,
            )
        elif price_model == "fill_prob_target":
            self._price_model = FillProbTargetModel(
                target_fill_prob=target_fill_prob,
                fill_risk_lambda=fill_risk_lambda if fill_risk_lambda > 0 else 10.0,
            )
        elif price_model == "optimal_ev":
            if optimal_ev_lambda <= 0:
                raise ValueError("optimal_ev price model requires optimal_ev_lambda > 0")
            self._price_model = OptimalEVModel(
                fill_risk_lambda=optimal_ev_lambda,
            )
            self._stuck_cost_lambda = optimal_ev_lambda
        else:
            self._price_model = AggressivePriceModel(
                base_improve_bps=price_improve_bps,
                edge_share=price_edge_share,
                fill_risk_lambda=fill_risk_lambda if fill_risk_lambda > 0 else 10.0,
            )

        self._init_target_portfolio(
            tracked_listings=self._group.all_listings,
            taker_order_type=OrderType(taker_order_type),
            max_staleness_ns=max_staleness_ns,
            processing_time_ns=processing_time_ns,
            imbalance_timeout_ns=imbalance_timeout_ns,
            max_unhedged_qty=max_unhedged_qty,
            debug=debug,
        )

        self._maker_reprice_cooldown_ns = maker_reprice_cooldown_ns
        self._cancel_grace_ns = cancel_grace_ns
        self._portfolio: ArbPortfolio | None = None
        self._constraint: ArbBudgetConstraint | None = None
        self._logged_at_target: dict[Listing, bool] = {}
        self._last_maker_price: dict[Listing, int] = {}
        self._last_reprice_ts: dict[Listing, int] = {}
        self._decisions = None
        self._best_edges = None
        self._maker_prices = None
        self._last_best_edge: tuple[int, float] | None = None

    def register_metrics(self) -> None:
        decisions = self.metrics.create_stream("decisions")
        self._d_ts = decisions.add_long_column("timestamp")
        self._d_event = decisions.add_int_column("event")
        self._d_pairing = decisions.add_int_column("pairing")
        self._d_edge = decisions.add_double_column("edge_bps")
        self._d_qty = decisions.add_double_column("qty")
        self._d_mode = decisions.add_int_column("order_mode")
        decisions.freeze()
        self._decisions = decisions

        best_edges = self.metrics.create_stream("best_edge")
        self._e_ts = best_edges.add_long_column("timestamp")
        self._e_pairing = best_edges.add_int_column("pairing")
        self._e_edge = best_edges.add_double_column("maker_edge_bps")
        self._e_min = best_edges.add_double_column("maker_min_bps")
        best_edges.freeze()
        self._best_edges = best_edges

        maker_prices = self.metrics.create_stream("maker_prices")
        self._p_ts = maker_prices.add_long_column("timestamp")
        self._p_exchange = maker_prices.add_int_column("exchange_id")
        self._p_security = maker_prices.add_int_column("security_id")
        self._p_bid = maker_prices.add_double_column("best_bid")
        self._p_ask = maker_prices.add_double_column("best_ask")
        self._p_price = maker_prices.add_double_column("price")
        self._p_max_price = maker_prices.add_double_column("max_price")
        self._p_fill_prob = maker_prices.add_double_column("fill_prob")
        self._p_stuck_cost = maker_prices.add_double_column("stuck_cost")
        maker_prices.freeze()
        self._maker_prices = maker_prices

    def _record_decision(self, timestamp: int, event: Decision, edge_bps: float, qty: int) -> None:
        if self._decisions is None:
            return
        pairing = self._portfolio.pairing_index if self._portfolio is not None else -1
        mode = int(self._portfolio.order_mode) if self._portfolio is not None else -1
        self._decisions.begin()
        self._decisions.set_long(self._d_ts, timestamp)
        self._decisions.set_int(self._d_event, int(event))
        self._decisions.set_int(self._d_pairing, pairing)
        self._decisions.set_double(self._d_edge, edge_bps)
        self._decisions.set_double(self._d_qty, qty / Scales.SIZE)
        self._decisions.set_int(self._d_mode, mode)
        self._decisions.commit()

    def _record_best_edge(self, timestamp: int, pairing: VenuePairing | None, edge_bps: float) -> None:
        if self._best_edges is None:
            return
        best = (pairing.index if pairing is not None else -1, edge_bps)
        if best == self._last_best_edge:
            return
        self._last_best_edge = best
        self._best_edges.begin()
        self._best_edges.set_long(self._e_ts, timestamp)
        self._best_edges.set_int(self._e_pairing, best[0])
        self._best_edges.set_double(self._e_edge, edge_bps)
        self._best_edges.set_double(self._e_min, self.min_pure_arb_bps)
        self._best_edges.commit()

    def _record_maker_price(
        self, timestamp: int, listing: Listing, best_bid: int, best_ask: int, price: int,
        max_price: int | None, fill_prob: float, stuck_cost: float,
    ) -> None:
        if self._maker_prices is None:
            return
        self._maker_prices.begin()
        self._maker_prices.set_long(self._p_ts, timestamp)
        self._maker_prices.set_int(self._p_exchange, listing.exchange_id)
        self._maker_prices.set_int(self._p_security, listing.security_id)
        self._maker_prices.set_double(self._p_bid, best_bid / Scales.PRICE)
        self._maker_prices.set_double(self._p_ask, best_ask / Scales.PRICE)
        self._maker_prices.set_double(self._p_price, price / Scales.PRICE)
        self._maker_prices.set_double(self._p_max_price, max_price / Scales.PRICE if max_price is not None else nan)
        self._maker_prices.set_double(self._p_fill_prob, fill_prob)
        self._maker_prices.set_double(self._p_stuck_cost, stuck_cost)
        self._maker_prices.commit()

    def _on_book_updated(
        self, listing: Listing,
        bids: list[tuple[int, int]], asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None:
        self._cost_model.on_book_update(listing, bids, asks, timestamp)
        self._price_model.on_book_update(listing, bids, asks, timestamp)
        if bids and self._constraint is not None and self._portfolio is not None and listing in self._portfolio:
            self._constraint.update_bid(listing, bids[0][0])

    def _on_fill(self, listing: Listing, report: ExecutionReport) -> None:
        if self._portfolio is not None and self._portfolio.target_qty(listing) > 0:
            order_mode = self._portfolio.order_mode
        else:
            order_mode = OrderMode.TAKER
        self._cost_model.on_fill(listing, report, order_mode)

        if self._portfolio is None:
            return
        if self._constraint is not None and listing in self._portfolio:
            self._constraint.record_fill(listing, report.fill_price, report.filled_qty)
        if listing not in self._portfolio:
            self._debug(
                report.timestamp_recv, "FILL",
                eid=listing.exchange_id, sid=listing.security_id,
                price=round(report.fill_price / Scales.PRICE, 4),
                qty=round(report.filled_qty / Scales.SIZE, 4),
                phase=self._portfolio.phase.name,
                note="orphan_fill",
            )
            return

        was_entering = self._portfolio.phase == ArbPhase.ENTERING
        self._portfolio.record_fill(listing, report.fill_price, report.filled_qty)

        cum_qty = 0
        for leg in self._portfolio.legs:
            if leg.listing == listing:
                cum_qty = leg.filled_qty
                break
        self._debug(
            report.timestamp_recv, "FILL",
            eid=listing.exchange_id, sid=listing.security_id,
            price=round(report.fill_price / Scales.PRICE, 4),
            qty=round(report.filled_qty / Scales.SIZE, 4),
            cum_qty=round(cum_qty / Scales.SIZE, 4),
            leaves=round(report.leaves_qty / Scales.SIZE, 4),
            fee=round(report.fee, 6),
            phase=self._portfolio.phase.name,
            target_qty=round(self._portfolio.target_qty(listing) / Scales.SIZE, 4),
        )

        if was_entering and self._portfolio.phase == ArbPhase.FILLED:
            self._debug(report.timestamp_recv, "PHASE_CHANGE", old="ENTERING", new="FILLED")

    def compute_target(self, timestamp: int) -> Target:
        target: Target = {}
        group_target = self._compute_group_target(timestamp)

        all_at_zero = all(qty == 0 for qty in group_target.values())
        all_flat = all(self._net_qty(lst) == 0 for lst in group_target)
        if all_at_zero and all_flat and group_target:
            self._record_decision(timestamp, Decision.FLAT, nan, 0)
            self._clear_state()
            print(f"[{timestamp}] FLAT '{self._group.label}'")
            return target

        order_mode = self._portfolio.order_mode if self._portfolio is not None else OrderMode.TAKER
        phase = self._portfolio.phase if self._portfolio is not None else ArbPhase.IDLE

        for lst, qty in group_target.items():
            actual = self._net_qty(lst)
            delta = qty - actual
            at_target = (qty == 0 and actual == 0) or (qty > 0 and actual >= qty)
            if at_target and not self._logged_at_target.get(lst, False):
                print(f"[{timestamp}] AT_TARGET {lst}: pos={actual} target={qty}")
                self._logged_at_target[lst] = True

            if at_target or delta == 0 or qty == 0 or order_mode == OrderMode.TAKER:
                price = 0
            elif delta > 0:
                bid_book = self._bid_book.get(lst, [])
                ask_book = self._ask_book.get(lst, [])
                best_bid = bid_book[0][0] if bid_book else 0
                best_ask = ask_book[0][0] if ask_book else 0
                if best_bid <= 0 or best_ask <= 0:
                    price = 0
                else:
                    max_price = self._constraint.max_price(lst) if self._constraint is not None and phase == ArbPhase.ENTERING else None
                    context = ArbContext(stuck_cost=self._compute_stuck_cost(lst)) if self._portfolio is not None and self._stuck_cost_lambda is not None else None
                    result = self._price_model.compute_price(lst, best_bid, best_ask, max_price, context)
                    price = result.price
                    if price > 0 and self._maker_reprice_cooldown_ns > 0:
                        prev = self._last_maker_price.get(lst)
                        if prev is not None and prev > 0 and price != prev:
                            elapsed = timestamp - self._last_reprice_ts.get(lst, 0)
                            if elapsed < self._maker_reprice_cooldown_ns:
                                price = prev
                    if self._constraint is not None:
                        self._constraint.update_order_price(lst, price, is_crossing=price >= best_ask and best_ask > 0)
                    if price != self._last_maker_price.get(lst, 0):
                        self._last_maker_price[lst] = price
                        self._last_reprice_ts[lst] = timestamp
                        self._record_maker_price(
                            timestamp, lst, best_bid, best_ask, price, max_price, result.fill_prob,
                            context.stuck_cost if context is not None else nan,
                        )
                    self._debug(
                        timestamp, "PRICE_MODEL",
                        eid=lst.exchange_id, sid=lst.security_id,
                        best_bid=round(best_bid / Scales.PRICE, 4),
                        best_ask=round(best_ask / Scales.PRICE, 4),
                        price=round(price / Scales.PRICE, 4),
                        max_price=round(max_price / Scales.PRICE, 4) if max_price is not None else None,
                        fill_prob=round(result.fill_prob, 3),
                        spread=round(result.spread, 4),
                        stuck_cost=round(context.stuck_cost, 4) if context is not None else None,
                    )
            else:
                ask_book = self._ask_book.get(lst, [])
                price = ask_book[0][0] if ask_book else 0

            self._debug(
                timestamp, "TARGET_SET",
                eid=lst.exchange_id, sid=lst.security_id,
                qty=qty // Scales.SIZE,
                price=round(price / Scales.PRICE, 4) if price else 0,
                mode=order_mode.name,
                phase=phase.name,
            )
            target[lst] = TargetEntry(qty=qty, price=price)
        return target

    def on_imbalance_timeout(self, timestamp: int) -> Target | None:
        self._record_decision(timestamp, Decision.LEG_RISK_UNWIND, nan, 0)
        self._clear_state()
        return None

    def _compute_group_target(self, timestamp: int) -> dict[Listing, int]:
        if self._is_unwinding():
            return {lst: 0 for lst in self._group.all_listings if self._net_qty(lst) != 0}

        if self._portfolio is None:
            result = self._try_enter(timestamp)
            if result is None:
                return {}
            new_target, arb_edge = result
            self._record_decision(timestamp, Decision.ENTRY, arb_edge, next(iter(new_target.values())))
            print(
                f"[{timestamp}] ENTRY '{self._group.label}' "
                f"pairing={self._portfolio.pairing_index} edge={arb_edge:.1f}bps "
                f"mode={self._portfolio.order_mode.name} "
                f"qty={next(iter(new_target.values())) // Scales.SIZE}"
            )
            return new_target

        if all(leg.target_qty == 0 for leg in self._portfolio.legs):
            return self._portfolio.as_target_dict()

        entry_pairing = next(
            (p for p in self._group.pairings if p.index == self._portfolio.pairing_index), None
        )

        if self._portfolio.phase == ArbPhase.ENTERING and self._constraint is not None and self._portfolio.should_cancel_entry(self._constraint, timestamp):
            edge = self._constraint.edge_bps()
            edge_str = f"{edge:.1f}" if edge is not None else "?"
            held_qty = max(
                (self._net_qty(lst) for _, lst in entry_pairing.legs), default=0
            ) if entry_pairing is not None else 0
            has_position = held_qty > 0
            self._record_decision(
                timestamp, Decision.ENTRY_CANCEL, edge if edge is not None else nan, max(held_qty, 0),
            )
            if has_position:
                self._portfolio.begin_exit()
                print(f"[{timestamp}] ENTRY_CANCEL '{self._group.label}' maker_edge={edge_str}bps (unwinding)")
                return self._portfolio.as_target_dict()
            else:
                print(f"[{timestamp}] ENTRY_CANCEL '{self._group.label}' maker_edge={edge_str}bps")
                self._clear_state()
                return {}

        if self._allow_early_exit and entry_pairing is not None and self._portfolio.phase == ArbPhase.FILLED:
            current_qty = max(
                (abs(self._net_qty(lst)) for _, lst in entry_pairing.legs), default=0
            )
            exit_qty, exit_edge, _ = self._walk_pairing(
                entry_pairing, timestamp, buy=False, max_qty=current_qty,
                order_mode=OrderMode.TAKER,
            )
            if exit_edge > self._exit_edge_threshold_bps and exit_qty >= current_qty:
                self._record_decision(timestamp, Decision.EXIT, exit_edge, current_qty)
                self._portfolio.begin_exit()
                self._logged_at_target.clear()
                print(
                    f"[{timestamp}] EXIT '{self._group.label}' "
                    f"exit_edge={exit_edge:.1f}bps"
                )
                return self._portfolio.as_target_dict()

        if self._allow_scaling and entry_pairing is not None and self._portfolio.phase == ArbPhase.FILLED:
            current_qty = next((leg.target_qty for leg in self._portfolio.legs if leg.target_qty > 0), 0)
            remaining_qty = self.max_position - current_qty if self.max_position > 0 else 0
            if self.max_position > 0 and remaining_qty <= 0:
                return self._portfolio.as_target_dict()
            arb_qty, arb_edge, arb_legs = self._walk_pairing(
                entry_pairing, timestamp, buy=True, max_qty=remaining_qty,
                order_mode=OrderMode.MAKER,
            )
            if arb_qty > 0 and arb_edge > self.min_pure_arb_bps:
                for _, lst in arb_legs:
                    self._portfolio.add_target_qty(lst, arb_qty)
                self._portfolio.phase = ArbPhase.ENTERING
                for _, lst in arb_legs:
                    self._logged_at_target.pop(lst, None)
                self._record_decision(timestamp, Decision.SCALE_UP, arb_edge, arb_qty)
                print(
                    f"[{timestamp}] SCALE_UP '{self._group.label}' "
                    f"edge={arb_edge:.1f}bps qty={arb_qty // Scales.SIZE} "
                    f"new_total={self._portfolio.target_qty(arb_legs[0][1]) // Scales.SIZE}"
                )
                return self._portfolio.as_target_dict()

        return self._portfolio.as_target_dict()

    def _prepare_books(
        self, pairing: VenuePairing, timestamp: int, buy: bool = True,
        order_mode: OrderMode = OrderMode.TAKER,
    ) -> tuple[list[list[tuple[int, int]]], list[Callable[[int, int], float]]] | tuple[None, None]:
        books: list[list[tuple[int, int]]] = []
        cost_fns: list[Callable[[int, int], float]] = []
        side = Side.BID if buy else Side.ASK
        for _, lst in pairing.legs:
            if self._is_stale(lst, timestamp):
                return None, None
            levels = (self._ask_book if buy else self._bid_book).get(lst, [])
            if not levels or levels[0][0] <= 0:
                return None, None
            books.append(levels)
            cost_fns.append(
                lambda p, q, l=lst, s=side, m=order_mode: self._cost_model.expected_cost(l, p, s, m, q)
            )
        return books, cost_fns

    def _walk_pairing(
        self, pairing: VenuePairing, timestamp: int, buy: bool, max_qty: int = 0,
        order_mode: OrderMode = OrderMode.TAKER,
    ) -> tuple[int, float, list[tuple[int, Listing]]]:
        books, cost_fns = self._prepare_books(pairing, timestamp, buy=buy, order_mode=order_mode)
        if books is None:
            return 0, float("-inf"), []
        qty_raw, edge_bps = walk_books(books, cost_fns, max_qty=max_qty, buy=buy)
        if qty_raw <= 0:
            return 0, float("-inf"), []
        if max_qty > 0:
            qty_raw = min(qty_raw, max_qty)
        return qty_raw, edge_bps, pairing.legs

    def _select_order_mode(
        self, pairing: VenuePairing, timestamp: int, max_qty: int,
    ) -> tuple[OrderMode, int, float, list[tuple[int, Listing]]] | None:
        if self._use_taker_orders:
            qty, edge, legs = self._walk_pairing(
                pairing, timestamp, buy=True, max_qty=max_qty, order_mode=OrderMode.TAKER,
            )
            self._debug(timestamp, "EDGE_CHECK", pairing=pairing.index,
                        taker_edge=round(edge, 2), threshold=self._taker_edge_threshold_bps)
            if qty > 0 and edge >= self._taker_edge_threshold_bps:
                self._log_fill_risk(timestamp, pairing, "TAKER_ENTRY")
                self._log_slippage(timestamp, pairing, "TAKER_ENTRY")
                return OrderMode.TAKER, qty, edge, legs

        qty, edge, legs = self._walk_pairing(
            pairing, timestamp, buy=True, max_qty=max_qty, order_mode=OrderMode.MAKER,
        )
        self._debug(timestamp, "EDGE_CHECK", pairing=pairing.index,
                    maker_edge=round(edge, 2), min_bps=self.min_pure_arb_bps)
        if qty > 0 and edge >= self.min_pure_arb_bps:
            self._log_fill_risk(timestamp, pairing, "MAKER_ENTRY")
            return OrderMode.MAKER, qty, edge, legs

        if self._fill_risk is not None and edge > float("-inf"):
            self._log_fill_risk(timestamp, pairing, "FILL_RISK_REJECT")
        return None

    def _try_enter(self, timestamp: int) -> tuple[dict[Listing, int], float] | None:
        best_edge = float("-inf")
        best_pairing: VenuePairing | None = None
        for pairing in self._group.pairings:
            _, edge, _ = self._walk_pairing(pairing, timestamp, buy=True, max_qty=0, order_mode=OrderMode.MAKER)
            if edge > best_edge:
                best_edge = edge
                best_pairing = pairing
        self._record_best_edge(timestamp, best_pairing, best_edge)
        if best_pairing is None or best_edge <= self.min_pure_arb_bps:
            return None

        entry_max = self.max_position if self.max_position > 0 else 0
        result = self._select_order_mode(best_pairing, timestamp, entry_max)
        if result is None:
            return None

        mode, arb_qty, arb_edge, arb_legs = result
        legs = [ArbLeg(listing=lst, target_qty=arb_qty) for _, lst in arb_legs]
        self._portfolio = ArbPortfolio(
            legs, pairing_index=best_pairing.index, order_mode=mode, cancel_grace_ns=self._cancel_grace_ns,
        )
        self._constraint = ArbBudgetConstraint(legs, self.min_pure_arb_bps, self._fee_rates)
        for _, lst in arb_legs:
            self._logged_at_target.pop(lst, None)
        self._debug(timestamp, "PHASE_CHANGE", old="IDLE", new="ENTERING", mode=mode.name)
        return self._portfolio.as_target_dict(), arb_edge

    def _clear_state(self) -> None:
        self._portfolio = None
        self._constraint = None
        self._logged_at_target.clear()
        self._last_maker_price.clear()
        self._last_reprice_ts.clear()

    def _compute_stuck_cost(self, listing: Listing) -> float:
        other_legs = [leg for leg in self._portfolio.legs if leg.listing != listing]
        if not other_legs or all(leg.is_filled for leg in other_legs):
            return 0.0
        lam = self._stuck_cost_lambda
        if lam is None or lam <= 0:
            return 0.0
        bid_book = self._bid_book.get(listing, [])
        ask_book = self._ask_book.get(listing, [])
        if not bid_book or not ask_book or bid_book[0][0] <= 0 or ask_book[0][0] <= 0:
            return 0.0
        spread = (ask_book[0][0] - bid_book[0][0]) / Scales.PRICE
        unwind_cost = spread * self._unwind_spread_mult
        p_other = 1.0
        for leg in other_legs:
            if not leg.is_filled:
                ob = self._bid_book.get(leg.listing, [])
                oa = self._ask_book.get(leg.listing, [])
                if ob and oa and ob[0][0] > 0 and oa[0][0] > 0:
                    s = (oa[0][0] - ob[0][0]) / Scales.PRICE
                else:
                    s = 0.0
                p_other *= exp(-lam * s)
        return (1.0 - p_other) * unwind_cost

    def _log_slippage(self, timestamp: int, pairing: VenuePairing, event: str) -> None:
        if self._depth_coverage is None:
            return
        parts = []
        for _, lst in pairing.legs:
            depth = self._depth_coverage.get_ask_depth(lst)
            parts.append(f"({lst.exchange_id},{lst.security_id}):cheap_depth={depth:.2f}")
        print(f"[{timestamp}] {event}_COVERAGE pairing={pairing.index} {' | '.join(parts)}")

    def _log_fill_risk(self, timestamp: int, pairing: VenuePairing, event: str) -> None:
        if self._fill_risk is None:
            return
        parts = []
        for _, lst in pairing.legs:
            spread = self._fill_risk.get_spread(lst)
            p_fill = self._fill_risk.get_fill_prob(lst)
            penalty = self._fill_risk.cost(lst, 0, Side.BID, 0)
            parts.append(
                f"({lst.exchange_id},{lst.security_id}):spread={spread:.4f},p_fill={p_fill:.3f},penalty={penalty:.4f}"
            )
        print(f"[{timestamp}] {event} pairing={pairing.index} {' | '.join(parts)}")
