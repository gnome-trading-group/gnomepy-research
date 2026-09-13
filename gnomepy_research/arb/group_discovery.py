from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from gnomepy.registry import RegistryClient, RelationshipGraph

from gnomepy_research.arb.types import Listing


@dataclass
class OutcomeLeg:
    outcome_label: str
    listings: list[Listing]


@dataclass
class VenuePairing:
    index: int
    legs: list[tuple[int, Listing]]  # [(outcome_idx, listing), ...] one per outcome


def enumerate_pairings(outcomes: list[OutcomeLeg]) -> list[VenuePairing]:
    if not outcomes:
        return []
    outcome_listings = [[(i, lst) for lst in o.listings] for i, o in enumerate(outcomes)]
    pairings = []
    for combo in product(*outcome_listings):
        if len({lst[0] for _, lst in combo}) < 2:
            continue
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
        self.pairings = enumerate_pairings(self.outcomes)


def discover_group(event_ids: list[int], registry: RegistryClient) -> ContractGroup:
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

        groups.append(ContractGroup(event.title, 0, outcomes))

    if len(groups) == 0:
        raise ValueError(f"No contract groups found for event_ids: {event_ids}")
    if len(groups) > 1:
        raise ValueError(
            f"Expected 1 contract group, found {len(groups)}. "
            "Use separate strategy instances for distinct arbs."
        )
    return groups[0]
