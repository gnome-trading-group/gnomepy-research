from __future__ import annotations

import pytest

from gnomepy_research.arb.group_discovery import (
    ContractGroup,
    OutcomeLeg,
    VenuePairing,
    enumerate_pairings,
)

Listing = tuple[int, int]


def make_leg(label: str, listings: list[Listing]) -> OutcomeLeg:
    return OutcomeLeg(outcome_label=label, listings=listings)


class TestEnumeratePairings:
    def test_two_outcomes_two_listings_each(self):
        # 4 combos, 2 are same-exchange (exch 1+1, exch 2+2), 2 are cross-exchange
        outcomes = [
            make_leg("Yes", [(1, 10), (2, 20)]),
            make_leg("No", [(1, 11), (2, 21)]),
        ]
        pairings = enumerate_pairings(outcomes)
        assert len(pairings) == 2

    def test_same_exchange_pairings_excluded(self):
        outcomes = [
            make_leg("Yes", [(1, 10), (2, 20)]),
            make_leg("No", [(1, 11), (2, 21)]),
        ]
        pairings = enumerate_pairings(outcomes)
        for p in pairings:
            exchanges = {lst[0] for _, lst in p.legs}
            assert len(exchanges) >= 2, f"Same-exchange pairing not filtered: {p}"

    def test_one_outcome_one_listing(self):
        # Single outcome on single exchange — can't form a cross-exchange pairing
        outcomes = [make_leg("Yes", [(1, 10)])]
        pairings = enumerate_pairings(outcomes)
        assert len(pairings) == 0

    def test_three_outcomes_two_listings_each(self):
        # 2^3=8 combos; same-exchange combos are those where all 3 legs share exch 1 or all share exch 2
        # (1,1,1) and (2,2,2) are filtered → 8-2=6
        outcomes = [
            make_leg("A", [(1, 1), (2, 2)]),
            make_leg("B", [(1, 3), (2, 4)]),
            make_leg("C", [(1, 5), (2, 6)]),
        ]
        pairings = enumerate_pairings(outcomes)
        assert len(pairings) == 6

    def test_indices_are_sequential(self):
        outcomes = [
            make_leg("Yes", [(1, 10), (2, 20)]),
            make_leg("No", [(1, 11), (2, 21)]),
        ]
        pairings = enumerate_pairings(outcomes)
        assert [p.index for p in pairings] == list(range(len(pairings)))

    def test_each_pairing_has_one_leg_per_outcome(self):
        outcomes = [
            make_leg("Yes", [(1, 10), (2, 20)]),
            make_leg("No", [(1, 11), (2, 21)]),
            make_leg("Draw", [(1, 12), (2, 22)]),
        ]
        pairings = enumerate_pairings(outcomes)
        for pairing in pairings:
            assert len(pairing.legs) == len(outcomes)

    def test_legs_cover_all_outcome_indices(self):
        outcomes = [
            make_leg("Yes", [(1, 10), (2, 20)]),
            make_leg("No", [(1, 11), (2, 21)]),
        ]
        pairings = enumerate_pairings(outcomes)
        for pairing in pairings:
            outcome_indices = {leg[0] for leg in pairing.legs}
            assert outcome_indices == set(range(len(outcomes)))

    def test_all_cross_exchange_combinations_covered(self):
        listings_a = [(1, 10), (2, 20)]
        listings_b = [(1, 11), (2, 21)]
        outcomes = [make_leg("Yes", listings_a), make_leg("No", listings_b)]
        pairings = enumerate_pairings(outcomes)
        pairing_sets = {frozenset(lst for _, lst in p.legs) for p in pairings}
        expected = {
            frozenset([a, b])
            for a in listings_a
            for b in listings_b
            if a[0] != b[0]
        }
        assert pairing_sets == expected

    def test_empty_outcomes(self):
        pairings = enumerate_pairings([])
        assert pairings == []


class TestContractGroupPostInit:
    def test_all_listings_union(self):
        outcomes = [
            make_leg("Yes", [(1, 10), (2, 20)]),
            make_leg("No", [(1, 11), (2, 21)]),
        ]
        group = ContractGroup(label="Test", index=0, outcomes=outcomes)
        assert group.all_listings == {(1, 10), (2, 20), (1, 11), (2, 21)}

    def test_pairings_computed(self):
        outcomes = [
            make_leg("Yes", [(1, 10), (2, 20)]),
            make_leg("No", [(1, 11), (2, 21)]),
        ]
        group = ContractGroup(label="Test", index=0, outcomes=outcomes)
        assert len(group.pairings) == 2

    def test_duplicate_listing_deduplicated(self):
        shared: Listing = (1, 10)
        outcomes = [
            make_leg("Yes", [shared, (2, 20)]),
            make_leg("No", [shared, (2, 21)]),
        ]
        group = ContractGroup(label="Test", index=0, outcomes=outcomes)
        assert shared in group.all_listings
        assert len(group.all_listings) == 3

    def test_single_outcome(self):
        outcomes = [make_leg("Yes", [(1, 10)])]
        group = ContractGroup(label="Test", index=0, outcomes=outcomes)
        assert group.all_listings == {(1, 10)}
        assert len(group.pairings) == 0

    def test_label_and_index_preserved(self):
        outcomes = [make_leg("Yes", [(1, 10)]), make_leg("No", [(2, 20)])]
        group = ContractGroup(label="My Market", index=7, outcomes=outcomes)
        assert group.label == "My Market"
        assert group.index == 7
