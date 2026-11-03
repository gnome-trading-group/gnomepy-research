from __future__ import annotations

import pytest

from gnomepy_research.arb.group_discovery import (
    ContractGroup,
    OutcomeLeg,
    VenuePairing,
    enumerate_pairings,
)
from gnomepy_research.listing import Listing


def L(exchange_id: int, security_id: int) -> Listing:
    return Listing(exchange_id * 1000 + security_id, exchange_id, security_id)


def make_leg(label: str, listings: list[Listing]) -> OutcomeLeg:
    return OutcomeLeg(outcome_label=label, listings=listings)


class TestEnumeratePairings:
    def test_two_outcomes_two_listings_each(self):
        # 4 combos, 2 are same-exchange (exch 1+1, exch 2+2), 2 are cross-exchange
        outcomes = [
            make_leg("Yes", [L(1, 10), L(2, 20)]),
            make_leg("No", [L(1, 11), L(2, 21)]),
        ]
        pairings = enumerate_pairings(outcomes)
        assert len(pairings) == 2

    def test_same_exchange_pairings_excluded(self):
        outcomes = [
            make_leg("Yes", [L(1, 10), L(2, 20)]),
            make_leg("No", [L(1, 11), L(2, 21)]),
        ]
        pairings = enumerate_pairings(outcomes)
        for p in pairings:
            exchanges = {lst.exchange_id for _, lst in p.legs}
            assert len(exchanges) >= 2, f"Same-exchange pairing not filtered: {p}"

    def test_one_outcome_one_listing(self):
        # Single outcome on single exchange — can't form a cross-exchange pairing
        outcomes = [make_leg("Yes", [L(1, 10)])]
        pairings = enumerate_pairings(outcomes)
        assert len(pairings) == 0

    def test_three_outcomes_two_listings_each(self):
        # 2^3=8 combos; same-exchange combos are those where all 3 legs share exch 1 or all share exch 2
        # (1,1,1) and (2,2,2) are filtered → 8-2=6
        outcomes = [
            make_leg("A", [L(1, 1), L(2, 2)]),
            make_leg("B", [L(1, 3), L(2, 4)]),
            make_leg("C", [L(1, 5), L(2, 6)]),
        ]
        pairings = enumerate_pairings(outcomes)
        assert len(pairings) == 6

    def test_indices_are_sequential(self):
        outcomes = [
            make_leg("Yes", [L(1, 10), L(2, 20)]),
            make_leg("No", [L(1, 11), L(2, 21)]),
        ]
        pairings = enumerate_pairings(outcomes)
        assert [p.index for p in pairings] == list(range(len(pairings)))

    def test_each_pairing_has_one_leg_per_outcome(self):
        outcomes = [
            make_leg("Yes", [L(1, 10), L(2, 20)]),
            make_leg("No", [L(1, 11), L(2, 21)]),
            make_leg("Draw", [L(1, 12), L(2, 22)]),
        ]
        pairings = enumerate_pairings(outcomes)
        for pairing in pairings:
            assert len(pairing.legs) == len(outcomes)

    def test_legs_cover_all_outcome_indices(self):
        outcomes = [
            make_leg("Yes", [L(1, 10), L(2, 20)]),
            make_leg("No", [L(1, 11), L(2, 21)]),
        ]
        pairings = enumerate_pairings(outcomes)
        for pairing in pairings:
            outcome_indices = {leg[0] for leg in pairing.legs}
            assert outcome_indices == set(range(len(outcomes)))

    def test_all_cross_exchange_combinations_covered(self):
        listings_a = [L(1, 10), L(2, 20)]
        listings_b = [L(1, 11), L(2, 21)]
        outcomes = [make_leg("Yes", listings_a), make_leg("No", listings_b)]
        pairings = enumerate_pairings(outcomes)
        pairing_sets = {frozenset(lst for _, lst in p.legs) for p in pairings}
        expected = {
            frozenset([a, b])
            for a in listings_a
            for b in listings_b
            if a.exchange_id != b.exchange_id
        }
        assert pairing_sets == expected

    def test_empty_outcomes(self):
        pairings = enumerate_pairings([])
        assert pairings == []


class TestContractGroupPostInit:
    def test_all_listings_union(self):
        outcomes = [
            make_leg("Yes", [L(1, 10), L(2, 20)]),
            make_leg("No", [L(1, 11), L(2, 21)]),
        ]
        group = ContractGroup(label="Test", index=0, outcomes=outcomes)
        assert group.all_listings == {L(1, 10), L(2, 20), L(1, 11), L(2, 21)}

    def test_pairings_computed(self):
        outcomes = [
            make_leg("Yes", [L(1, 10), L(2, 20)]),
            make_leg("No", [L(1, 11), L(2, 21)]),
        ]
        group = ContractGroup(label="Test", index=0, outcomes=outcomes)
        assert len(group.pairings) == 2

    def test_duplicate_listing_deduplicated(self):
        shared: Listing = L(1, 10)
        outcomes = [
            make_leg("Yes", [shared, L(2, 20)]),
            make_leg("No", [shared, L(2, 21)]),
        ]
        group = ContractGroup(label="Test", index=0, outcomes=outcomes)
        assert shared in group.all_listings
        assert len(group.all_listings) == 3

    def test_single_outcome(self):
        outcomes = [make_leg("Yes", [L(1, 10)])]
        group = ContractGroup(label="Test", index=0, outcomes=outcomes)
        assert group.all_listings == {L(1, 10)}
        assert len(group.pairings) == 0

    def test_label_and_index_preserved(self):
        outcomes = [make_leg("Yes", [L(1, 10)]), make_leg("No", [L(2, 20)])]
        group = ContractGroup(label="My Market", index=7, outcomes=outcomes)
        assert group.label == "My Market"
        assert group.index == 7
