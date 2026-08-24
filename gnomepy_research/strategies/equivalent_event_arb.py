from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

from gnomepy.registry import RegistryClient, RelationshipGraph

from gnomepy_research.strategies.target_portfolio import (
    Listing,
    Target,
    TargetEntry,
    TargetPortfolioStrategy,
)

PRICE_SCALE = 1_000_000_000


def walk_books(
    books: list[list[tuple[int, float]]],
    fee_fns: list[Callable[[float], float]],
    max_qty: float = float("inf"),
    buy: bool = True,
) -> tuple[float, float, list[float]]:
    """Walk order books for multiple legs simultaneously.

    For buy: walks ask books, accumulates until sum(prices) + sum(fees) >= 1.0.
    For sell: walks bid books, accumulates until sum(prices) - sum(fees) <= 1.0.

    Returns (total_qty_contracts, edge_bps, vwap_per_leg).
    """
    num_legs = len(books)
    level_idx = [0] * num_legs
    remaining = [books[i][0][1] for i in range(num_legs)]

    total_qty = 0.0
    total_value = [0.0] * num_legs
    total_fees = [0.0] * num_legs

    while total_qty < max_qty:
        prices_f = [books[i][level_idx[i]][0] / PRICE_SCALE for i in range(num_legs)]
        fees_f = [fee_fns[i](prices_f[i]) for i in range(num_legs)]
        if buy and sum(prices_f) + sum(fees_f) >= 1.0:
            break
        if not buy and sum(prices_f) - sum(fees_f) <= 1.0:
            break

        chunk = min(remaining)
        if chunk <= 0:
            break
        chunk = min(chunk, max_qty - total_qty)

        total_qty += chunk
        for i in range(num_legs):
            total_value[i] += prices_f[i] * chunk
            total_fees[i] += fees_f[i] * chunk
            remaining[i] -= chunk
            if remaining[i] <= 0:
                level_idx[i] += 1
                if level_idx[i] < len(books[i]):
                    remaining[i] = books[i][level_idx[i]][1]
                else:
                    remaining[i] = 0

        if any(remaining[i] <= 0 and level_idx[i] >= len(books[i]) for i in range(num_legs)):
            break

    if total_qty <= 0:
        return 0.0, float("-inf"), []

    if buy:
        edge_bps = (total_qty - sum(total_value) - sum(total_fees)) / total_qty * 10_000
    else:
        edge_bps = (sum(total_value) - total_qty - sum(total_fees)) / total_qty * 10_000
    vwap = [total_value[i] / total_qty for i in range(num_legs)]
    return total_qty, edge_bps, vwap


@dataclass
class OutcomeLeg:
    outcome_label: str
    listings: list[Listing]


@dataclass
class VenuePairing:
    index: int
    legs: list[tuple[int, Listing]]  # [(outcome_idx, listing), ...] one per outcome


def _enumerate_pairings(outcomes: list[OutcomeLeg]) -> list[VenuePairing]:
    outcome_listings = [[(i, lst) for lst in o.listings] for i, o in enumerate(outcomes)]
    pairings = []
    for combo in product(*outcome_listings):
        pairings.append(VenuePairing(index=len(pairings), legs=list(combo)))
    return pairings


@dataclass
class ContractGroup:
    label: str
    index: int
    outcomes: list[OutcomeLeg]
    all_listings: set[Listing] = field(default_factory=set)
    pairings: list[VenuePairing] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.all_listings = {lst for o in self.outcomes for lst in o.listings}
        self.pairings = _enumerate_pairings(self.outcomes)


def _discover_groups(event_ids: list[int], registry: RegistryClient) -> list[ContractGroup]:
    rels = registry.get_contract_relationships(relationship_type="EQUIVALENT")
    graph = RelationshipGraph(rels)

    groups: list[ContractGroup] = []
    seen_listing_sets: set[frozenset] = set()

    for event_id in event_ids:
        contracts = registry.get_event_contracts(event_id=event_id)
        events = registry.get_event(event_id=event_id)
        if not events:
            continue
        event = events[0]

        outcomes: list[OutcomeLeg] = []
        for contract in contracts:
            sid = contract.security_id
            source_listings = registry.get_listing(security_id=sid)
            equivalent_sids = set(graph.get_equivalents(sid)) - {sid}
            equiv_listings = [
                lst
                for eq_sid in equivalent_sids
                for lst in registry.get_listing(security_id=eq_sid)
            ]
            all_listings = list(
                {(lst.exchange_id, lst.security_id) for lst in source_listings + equiv_listings}
            )
            if len(all_listings) >= 2:
                outcomes.append(OutcomeLeg(contract.outcome_label, all_listings))

        if len(outcomes) < 2:
            continue

        key = frozenset(lst for o in outcomes for lst in o.listings)
        if key in seen_listing_sets:
            continue
        seen_listing_sets.add(key)

        groups.append(ContractGroup(event.title, len(groups), outcomes))

    return groups


class EquivalentEventArb(TargetPortfolioStrategy):
    def __init__(
        self,
        event_ids: list[int],
        max_position: int = 0,
        taker_fee_rate: float = 0.07,
        min_pure_arb_bps: float = 5.0,
        max_staleness_ns: int = 5_000_000_000,
        processing_time_ns: int = 5_000_000,
        allow_scaling: bool = False,
        imbalance_timeout_ns: int = 30_000_000_000,
        max_unhedged_qty: int = 0,
    ):
        self.max_position = max_position
        self.min_pure_arb_bps = min_pure_arb_bps
        self._allow_scaling = allow_scaling

        registry = RegistryClient()
        self._groups = _discover_groups(event_ids, registry)

        all_exchanges = registry.get_exchange()
        exchange_by_name = {e.exchange_name.lower(): e for e in all_exchanges}

        poly = exchange_by_name.get("polymarket")
        kalshi = exchange_by_name.get("kalshi")
        if poly is None:
            raise ValueError("Exchange 'Polymarket' not found in registry")
        if kalshi is None:
            raise ValueError("Exchange 'Kalshi' not found in registry")

        self._eid_taker_fee_rate: dict[int, float] = {
            poly.exchange_id: taker_fee_rate,
            kalshi.exchange_id: taker_fee_rate,
        }

        all_eids = {lst[0] for g in self._groups for lst in g.all_listings}
        unknown = all_eids - set(self._eid_taker_fee_rate)
        if unknown:
            raise ValueError(f"No fee model implemented for exchange_id(s): {unknown}")

        all_listings = {lst for g in self._groups for lst in g.all_listings}
        self._init_target_portfolio(
            tracked_listings=all_listings,
            maker_orders=False,
            max_staleness_ns=max_staleness_ns,
            processing_time_ns=processing_time_ns,
            size=1_000_000,
            imbalance_timeout_ns=imbalance_timeout_ns,
            max_unhedged_qty=max_unhedged_qty,
        )

        self._group_target: dict[int, dict[Listing, int]] = {}
        self._logged_at_target: dict[Listing, bool] = {}
        self._group_entry_edge: dict[int, float] = {}
        self._group_entry_pairing: dict[int, int] = {}
        self._group_max_edge: dict[int, float] = {}

    def compute_target(self, timestamp: int) -> Target:
        target: Target = {}
        for group in self._groups:
            group_target = self._compute_group_target(group, timestamp)
            all_at_zero = all(qty == 0 for qty in group_target.values())
            all_flat = all(self._net_qty(lst) == 0 for lst in group_target)
            if all_at_zero and all_flat and group_target:
                self._clear_group_state(group.index)
                for lst in group.all_listings:
                    self._logged_at_target.pop(lst, None)
                print(f"[{timestamp}] FLAT group={group.index} '{group.label}'")
                continue
            for lst, qty in group_target.items():
                actual = self._net_qty(lst)
                at_target = (qty == 0 and actual == 0) or (qty > 0 and actual >= qty)
                if at_target and not self._logged_at_target.get(lst, False):
                    print(f"[{timestamp}] AT_TARGET {lst}: pos={actual} target={qty}")
                    self._logged_at_target[lst] = True
                target[lst] = TargetEntry(qty=qty, use_taker=not at_target, group_id=str(group.index))
        return target

    def on_imbalance_timeout(self, group_id: str, timestamp: int) -> Target | None:
        self._clear_group_state(int(group_id))
        return None

    def _compute_group_target(self, group: ContractGroup, timestamp: int) -> dict[Listing, int]:
        if self._is_unwinding(str(group.index)):
            return {lst: 0 for lst in group.all_listings if self._net_qty(lst) != 0}

        existing = self._group_target.get(group.index, {})

        if not existing:
            result = self._try_enter(group, timestamp, {})
            if result is None:
                return existing
            new_target, best_pairing, arb_qty, arb_edge, arb_legs = result
            prices = {str(lst): round(p / PRICE_SCALE, 4) for _, lst, p in arb_legs}
            print(
                f"[{timestamp}] ENTRY group={group.index} '{group.label}' "
                f"pairing={best_pairing.index} edge={arb_edge:.1f}bps "
                f"qty={arb_qty // self._size} prices={prices}"
            )
            return new_target

        if all(qty == 0 for qty in existing.values()):
            exiting_pairing_idx = self._group_entry_pairing.get(group.index, -1)
            result = self._try_enter(group, timestamp, dict(existing), exclude_pairing_idx=exiting_pairing_idx)
            if result is not None:
                new_target, best_pairing, arb_qty, arb_edge, arb_legs = result
                entry_prices = {str(lst): round(p / PRICE_SCALE, 4) for _, lst, p in arb_legs}
                print(
                    f"[{timestamp}] ENTER_WHILE_EXITING group={group.index} '{group.label}' "
                    f"pairing={best_pairing.index} edge={arb_edge:.1f}bps "
                    f"qty={arb_qty // self._size} prices={entry_prices}"
                )
                return new_target
            return existing

        entry_pairing_idx = self._group_entry_pairing.get(group.index, -1)
        entry_pairing = next(
            (p for p in group.pairings if p.index == entry_pairing_idx), None
        )

        if entry_pairing is not None:
            exit_qty, _, exit_edge, exit_legs = self._compute_exit_edge(group, timestamp, entry_pairing)
            current_qty = max(
                (abs(self._net_qty(lst)) for _, lst in entry_pairing.legs), default=0
            )
            if exit_edge > self.min_pure_arb_bps and exit_qty >= current_qty:
                exit_prices = {str(lst): round(p / PRICE_SCALE, 4) for _, lst, p in exit_legs}
                zero_target = {lst: 0 for lst in existing}
                result = self._try_enter(group, timestamp, zero_target, exclude_pairing_idx=entry_pairing_idx)
                if result is not None:
                    new_target, best_pairing, arb_qty, arb_edge, arb_legs = result
                    entry_prices = {str(lst): round(p / PRICE_SCALE, 4) for _, lst, p in arb_legs}
                    print(
                        f"[{timestamp}] EXIT_ENTER group={group.index} '{group.label}' "
                        f"exit_edge={exit_edge:.1f}bps entry_edge={arb_edge:.1f}bps "
                        f"qty={arb_qty // self._size} "
                        f"exit_prices={exit_prices} entry_prices={entry_prices}"
                    )
                    return new_target
                self._group_target[group.index] = zero_target
                for lst in zero_target:
                    self._logged_at_target.pop(lst, None)
                print(
                    f"[{timestamp}] EXIT group={group.index} '{group.label}' "
                    f"exit_edge={exit_edge:.1f}bps prices={exit_prices}"
                )
                return zero_target

        if self._allow_scaling and entry_pairing is not None:
            all_filled = all(
                self._net_qty(lst) >= qty
                for lst, qty in existing.items() if qty > 0
            )
            if all_filled:
                current_qty = next((qty for qty in existing.values() if qty > 0), 0)
                remaining_qty = self.max_position * self._size - current_qty if self.max_position > 0 else 0
                if self.max_position > 0 and remaining_qty <= 0:
                    return existing
                arb_qty, arb_edge, arb_legs = self._walk_pairing(entry_pairing, timestamp, buy=True, max_qty=remaining_qty)
                if arb_qty > 0 and arb_edge > self.min_pure_arb_bps:
                    new_target = dict(existing)
                    for _, lst, _ in arb_legs:
                        new_target[lst] = existing.get(lst, 0) + arb_qty
                    self._group_target[group.index] = new_target
                    for lst in new_target:
                        self._logged_at_target.pop(lst, None)
                    prices = {str(lst): round(p / PRICE_SCALE, 4) for _, lst, p in arb_legs}
                    print(
                        f"[{timestamp}] SCALE_UP group={group.index} '{group.label}' "
                        f"edge={arb_edge:.1f}bps qty={arb_qty // self._size} prices={prices} "
                        f"new_total={new_target.get(arb_legs[0][1], 0) // self._size}"
                    )
                    return new_target

        best_edge = self.min_pure_arb_bps
        for pairing in group.pairings:
            _, edge, _ = self._walk_pairing(pairing, timestamp, buy=True, max_qty=0)
            if edge > best_edge:
                best_edge = edge

        if best_edge > self._group_max_edge.get(group.index, 0.0):
            self._group_max_edge[group.index] = best_edge
            print(
                f"[{timestamp}] EDGE_WIDENED group={group.index} '{group.label}' "
                f"entry={self._group_entry_edge[group.index]:.1f}bps "
                f"now={best_edge:.1f}bps"
            )

        return existing

    def _compute_fee(self, listing: Listing, price_f: float) -> float:
        rate = self._eid_taker_fee_rate.get(listing[0])
        if rate is None:
            raise ValueError(f"No fee model for exchange_id: {listing[0]}")
        if rate == 0.0:
            return 0.0
        return rate * price_f * (1.0 - price_f)

    def _prepare_books(
        self, pairing: VenuePairing, timestamp: int, buy: bool = True
    ) -> tuple[list[list[tuple[int, float]]], list[Callable[[float], float]]] | tuple[None, None]:
        books: list[list[tuple[int, float]]] = []
        fee_fns: list[Callable[[float], float]] = []
        for _, lst in pairing.legs:
            if self._is_stale(lst, timestamp):
                return None, None
            levels = (self._ask_book if buy else self._bid_book).get(lst, [])
            if not levels or levels[0][0] <= 0:
                return None, None
            books.append(levels)
            fee_fns.append(lambda p, l=lst: self._compute_fee(l, p))
        return books, fee_fns

    def _walk_pairing(
        self, pairing: VenuePairing, timestamp: int, buy: bool, max_qty: int = 0
    ) -> tuple[int, float, list[tuple[int, Listing, int]]]:
        books, fee_fns = self._prepare_books(pairing, timestamp, buy=buy)
        if books is None:
            return 0, float("-inf"), []
        max_q = max_qty / self._size if max_qty > 0 else float("inf")
        total_qty, edge_bps, vwap = walk_books(books, fee_fns, max_qty=max_q, buy=buy)
        if total_qty <= 0:
            return 0, float("-inf"), []
        legs = [
            (pairing.legs[i][0], pairing.legs[i][1], int(vwap[i] * PRICE_SCALE))
            for i in range(len(pairing.legs))
        ]
        qty_scaled = int(total_qty * self._size)
        if max_qty > 0:
            qty_scaled = min(qty_scaled, max_qty)
        return qty_scaled, edge_bps, legs

    def _try_enter(
        self,
        group: ContractGroup,
        timestamp: int,
        base_target: dict[Listing, int],
        exclude_pairing_idx: int = -1,
    ) -> tuple[dict[Listing, int], VenuePairing, int, float, list[tuple[int, Listing, int]]] | None:
        best_edge = self.min_pure_arb_bps
        best_pairing: VenuePairing | None = None
        for pairing in group.pairings:
            if pairing.index == exclude_pairing_idx:
                continue
            _, edge, _ = self._walk_pairing(pairing, timestamp, buy=True, max_qty=0)
            if edge > best_edge:
                best_edge = edge
                best_pairing = pairing
        if best_pairing is None:
            return None
        entry_max = self.max_position * self._size if self.max_position > 0 else 0
        arb_qty, arb_edge, arb_legs = self._walk_pairing(best_pairing, timestamp, buy=True, max_qty=entry_max)
        if arb_qty <= 0:
            return None
        new_target = dict(base_target)
        for _, lst, _ in arb_legs:
            new_target[lst] = arb_qty
        self._group_target[group.index] = new_target
        self._group_entry_edge[group.index] = arb_edge
        self._group_entry_pairing[group.index] = best_pairing.index
        self._group_max_edge[group.index] = arb_edge
        for lst in new_target:
            self._logged_at_target.pop(lst, None)
        return new_target, best_pairing, arb_qty, arb_edge, arb_legs

    def _clear_group_state(self, group_index: int) -> None:
        self._group_target.pop(group_index, None)
        self._group_entry_edge.pop(group_index, None)
        self._group_entry_pairing.pop(group_index, None)
        self._group_max_edge.pop(group_index, None)

    def _compute_exit_edge(
        self, group: ContractGroup, timestamp: int, pairing: VenuePairing
    ) -> tuple[int, float, float, list[tuple[int, Listing, int]]]:
        current_qty = max(
            (abs(self._net_qty(lst)) for _, lst in pairing.legs), default=0
        )
        qty, exit_edge, legs = self._walk_pairing(pairing, timestamp, buy=False, max_qty=current_qty)
        if qty <= 0:
            return 0, 0.0, float("-inf"), []
        total = sum(p / PRICE_SCALE for _, _, p in legs)
        return qty, total, exit_edge, legs
