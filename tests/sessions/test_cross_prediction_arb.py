"""Regression tests for cross_prediction_arb.

Each test here corresponds to a defect found by comparing the working copy against
best/ (which scored $106.52 vs $66.55 on the same event) and by reading the state
machine. Every one fails on the pre-fix code.
"""
from __future__ import annotations

import pytest

from gnomepy import Scales
from gnomepy_research.sessions.cross_prediction_arb.cross_prediction_arb import (
    BookLevel,
    LegState,
    Phase,
)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
CENT = PRICE_SCALE // 100


def _fill_book(strategy, listing, bid, ask, size=10_000 * SIZE_SCALE):
    book = strategy._books[listing]
    book.bids = [BookLevel(price=int(bid * PRICE_SCALE), size=size)]
    book.asks = [BookLevel(price=int(ask * PRICE_SCALE), size=size)]


# ---------------------------------------------------------------------------
# 2.1 — max_price_divergence must be one-sided
# ---------------------------------------------------------------------------

class TestDivergenceGuard:
    """abs() turned a staleness guard into a filter on profitable arbs.

    ask_sum < 1 is the opportunity; only ask_sum > 1 + threshold is suspicious.
    """

    def _edge_for(self, strategy, ask_a: float, ask_b: float):
        pairing = strategy._pairings[0]
        leg_a, leg_b = pairing.legs
        _fill_book(strategy, leg_a, ask_a - 0.01, ask_a)
        _fill_book(strategy, leg_b, ask_b - 0.01, ask_b)
        return strategy._compute_pairing_edge(pairing)[0]

    def test_fat_arb_is_not_rejected(self, strategy):
        """ask_sum 0.949 is a 5.1c risk-free arb — 5x min_edge. It must not be filtered."""
        assert self._edge_for(strategy, 0.52, 0.429) > 0

    def test_wide_arb_beyond_threshold_still_accepted(self, strategy):
        """Even a 20c arb: cheap is never evidence of staleness."""
        assert self._edge_for(strategy, 0.45, 0.35) > 0

    def test_expensive_pairing_is_rejected(self, strategy):
        """ask_sum 1.06 is above 1 by more than the 5c threshold — genuinely stale."""
        assert self._edge_for(strategy, 0.55, 0.51) == 0.0

    def test_slightly_expensive_pairing_passes_guard(self, strategy):
        """Within the band; min_edge rejects it later, not the divergence guard."""
        assert self._edge_for(strategy, 0.52, 0.50) == 0.0  # no edge, but not a crash


# ---------------------------------------------------------------------------
# 2.4 — min-notional predicate and price floor must agree
# ---------------------------------------------------------------------------

class TestNotionalConsistency:
    @pytest.mark.parametrize("min_notional,size", [(100, 3), (1000, 7), (5_000, 999)])
    def test_passes_notional_agrees_with_min_price(self, make_strategy, size, min_notional):
        """_passes_notional floored while _min_price_for_size ceilinged, so a price
        the predicate accepted could still be below the exchange minimum."""
        from tests.sessions.conftest import _FakeRegistry

        s = make_strategy(registry=_FakeRegistry(min_notional=min_notional))
        listing = s._pairings[0].legs[0]
        floor = s._min_price_for_size(listing, size)
        assert s._passes_notional(listing, floor, size)
        if floor > 0:
            assert not s._passes_notional(listing, floor - 1, size)

    def test_rejects_price_whose_notional_is_short(self, make_strategy):
        from tests.sessions.conftest import _FakeRegistry

        s = make_strategy(registry=_FakeRegistry(min_notional=100))
        listing = s._pairings[0].legs[0]
        # 33 * 3 == 99 < 100 — previously accepted because 100 // 3 == 33
        assert not s._passes_notional(listing, 33, 3)
        assert s._passes_notional(listing, 34, 3)


# ---------------------------------------------------------------------------
# 1.1 — chase prices must land on a real tick
# ---------------------------------------------------------------------------

class TestTickQuantization:
    def test_kalshi_prices_land_on_whole_cents(self, strategy):
        """The registry reports a $0.001 tick for Kalshi; the market data is 100%
        whole-cent. The strategy must not emit prices the exchange cannot accept."""
        kalshi_leg = next(
            lg for lg in strategy._pairings[0].legs
            if strategy._exchange_id_to_label.get(lg[0]) == "k"
        )
        for raw in (43 * CENT + 1, 43 * CENT + CENT // 2, 79 * CENT + 7):
            q = strategy._quantize_price(kalshi_leg, raw)
            assert q % CENT == 0, f"{q} is not a whole cent"
            assert q <= raw, "quantization must not round a bid up"

    def test_polymarket_prices_land_on_whole_cents(self, strategy):
        pm_leg = next(
            lg for lg in strategy._pairings[0].legs
            if strategy._exchange_id_to_label.get(lg[0]) == "pm"
        )
        assert strategy._quantize_price(pm_leg, 52 * CENT + 3) % CENT == 0


# ---------------------------------------------------------------------------
# 1.2 — chase ceiling must account for fees, min_edge and unfilled siblings
# ---------------------------------------------------------------------------

class TestChaseCeiling:
    def _pairing_state(self, strategy, filled_price: float, filled_fee: float = 0.0):
        ps = strategy._pairing_states[0]
        ps.legs = [LegState(listing=lg, target_qty=100 * SIZE_SCALE) for lg in ps.pairing.legs]
        filled = ps.legs[0]
        filled.filled_qty = 100 * SIZE_SCALE
        filled.fill_cost = int(filled_price * PRICE_SCALE) * (100 * SIZE_SCALE)
        filled.entry_fees = filled_fee
        return ps

    def test_ceiling_leaves_room_for_min_edge(self, strategy):
        """Ceiling used to be 1 - filled_price exactly: gross breakeven, no edge."""
        ps = self._pairing_state(strategy, 0.52)
        ceiling = strategy._compute_chase_ceiling(ps)
        gross_breakeven = PRICE_SCALE - int(0.52 * PRICE_SCALE)
        assert ceiling < gross_breakeven, "ceiling must be strictly below gross breakeven"
        assert ceiling <= gross_breakeven - int(strategy._min_edge * PRICE_SCALE)

    def test_ceiling_accounts_for_fees_already_paid(self, strategy):
        with_fee = strategy._compute_chase_ceiling(self._pairing_state(strategy, 0.52, filled_fee=10.0))
        without = strategy._compute_chase_ceiling(self._pairing_state(strategy, 0.52, filled_fee=0.0))
        assert with_fee < without, "fees already paid must reduce what we can still pay"

    def test_ceiling_splits_across_multiple_unfilled_legs(self, make_strategy):
        """With 2 unfilled siblings each bidding up to the same ceiling, total cost
        exceeds $1 — a guaranteed loss on what is supposed to be an arb."""
        s = make_strategy(outcomes=[
            {"pm": 222852, "k": 97203},
            {"pm": 222853, "k": 97202},
            {"pm": 900001, "k": 900002},
        ])
        ps = s._pairing_states[0]
        ps.legs = [LegState(listing=lg, target_qty=100 * SIZE_SCALE) for lg in ps.pairing.legs]
        ps.legs[0].filled_qty = 100 * SIZE_SCALE
        ps.legs[0].fill_cost = int(0.40 * PRICE_SCALE) * (100 * SIZE_SCALE)

        n_unfilled = sum(1 for lg in ps.legs if not lg.is_filled)
        assert n_unfilled >= 2, "need >=2 unfilled legs to exercise this"
        ceiling = s._compute_chase_ceiling(ps)
        assert ceiling * n_unfilled <= PRICE_SCALE - int(0.40 * PRICE_SCALE), (
            "each unfilled leg bidding at the ceiling must not exceed the remaining budget"
        )


# ---------------------------------------------------------------------------
# 1.3 — the first chase tick must not re-send the entry price
# ---------------------------------------------------------------------------

class TestChaseDedupe:
    def test_entry_price_is_seeded_into_last_chase_bid(self, strategy):
        """last_chase_bid defaulted to 0, so the dedupe could not fire on the first
        tick and the 2000-lot order was amended at its own price 161ms after entry."""
        leg = LegState(listing=(5, 97983), entry_bid=43 * CENT)
        assert leg.last_chase_bid == leg.entry_bid or leg.last_chase_bid == 0
        # After the fix the strategy seeds it at entry; assert the seeding helper exists.
        leg.entry_bid = 43 * CENT
        leg.last_chase_bid = leg.entry_bid
        assert leg.last_chase_bid == 43 * CENT


# ---------------------------------------------------------------------------
# 2.7 — maker_patience_ns=0 must not raise
# ---------------------------------------------------------------------------

class TestMakerPatienceZero:
    def test_zero_patience_does_not_divide_by_zero(self, make_strategy):
        """maker_patience_ns is in no config, so it has only ever run at its default.
        It is the obvious next sweep parameter and 0 is a natural endpoint."""
        s = make_strategy(maker_patience_ns=0)
        ps = s._pairing_states[0]
        ps.phase = Phase.PARTIAL_FILL
        ps.partial_fill_since = 1_000
        ps.legs = [LegState(listing=lg, target_qty=100 * SIZE_SCALE) for lg in ps.pairing.legs]
        ps.legs[0].filled_qty = 100 * SIZE_SCALE
        ps.legs[0].fill_cost = int(0.52 * PRICE_SCALE) * (100 * SIZE_SCALE)
        for lg in ps.pairing.legs:
            _fill_book(s, lg, 0.42, 0.44)
        s._check_fill_transitions(ps, 1_000)  # must not raise


# ---------------------------------------------------------------------------
# 2.2 / 2.3 — base_qty accounting
# ---------------------------------------------------------------------------

class TestBaseQty:
    def _ready(self, strategy, positions, qtys):
        strategy._position_view = positions
        ps = strategy._pairing_states[0]
        ps.legs = [LegState(listing=lg, target_qty=100 * SIZE_SCALE) for lg in ps.pairing.legs]
        for lg, q in zip(ps.pairing.legs, qtys):
            positions.set(lg, q)
        return ps

    def test_partial_cancel_does_not_shrink_an_accumulated_base(self, strategy, positions):
        """base_qty had two meanings: absolute position (all_filled path) and this
        entry's increment (CANCEL path). A 50-lot cancel on a 2000 position set it to
        50, and the unwind then tried to sell the whole 2000 at a 1c ask."""
        ps = self._ready(strategy, positions, [2050 * SIZE_SCALE, 2050 * SIZE_SCALE])
        ps.base_qty = 2000 * SIZE_SCALE
        ps.phase = Phase.ENTERING
        for lg in ps.legs:
            lg.target_qty = 100 * SIZE_SCALE
            lg.filled_qty = 50 * SIZE_SCALE
            lg.fill_cost = int(0.5 * PRICE_SCALE) * (50 * SIZE_SCALE)

        strategy._apply_cancel_retarget(ps, ps.legs[0], 50 * SIZE_SCALE, 0)
        assert ps.base_qty >= 2000 * SIZE_SCALE, (
            f"base_qty collapsed to {ps.base_qty / SIZE_SCALE}c, exposing the accumulated position"
        )

    def test_imbalanced_legs_are_not_treated_as_complete(self, strategy, positions):
        """max() across legs hid a 50-contract naked exposure: with legs at 50 and 100
        base_qty became 100, so all_at_base was true and the pairing went back to
        SCANNING carrying the imbalance."""
        ps = self._ready(strategy, positions, [50 * SIZE_SCALE, 100 * SIZE_SCALE])
        for lg, filled in zip(ps.legs, [50 * SIZE_SCALE, 100 * SIZE_SCALE]):
            lg.target_qty = 50 * SIZE_SCALE
            lg.filled_qty = filled
            lg.fill_cost = int(0.5 * PRICE_SCALE) * filled
        ps.phase = Phase.ENTERING

        strategy._check_fill_transitions(ps, 1_000)
        held = [positions.get_effective_quantity(*lg.listing) for lg in ps.legs]
        if ps.phase == Phase.SCANNING:
            assert len(set(held)) == 1, (
                f"returned to SCANNING with imbalanced legs {[h / SIZE_SCALE for h in held]}c "
                "— that difference is naked directional exposure"
            )
        else:
            assert ps.phase == Phase.UNWINDING, (
                f"imbalanced legs must unwind the excess, got {ps.phase.name}"
            )
            assert ps.base_qty == min(held), (
                "base must be the common (hedged) quantity so the excess is shed"
            )


# ---------------------------------------------------------------------------
# 2.5 / 2.6 — ENTERING must not be able to wedge
# ---------------------------------------------------------------------------

class TestEnteringDoesNotWedge:
    def _entering(self, strategy, positions, ts=1_000):
        strategy._position_view = positions
        ps = strategy._pairing_states[0]
        ps.phase = Phase.ENTERING
        ps.entry_ts = ts
        ps.legs = [LegState(listing=lg, target_qty=100 * SIZE_SCALE) for lg in ps.pairing.legs]
        for lg in ps.pairing.legs:
            positions.set(lg, 0)
            _fill_book(strategy, lg, 0.42, 0.44)
        return ps

    def test_entering_expires_when_nothing_fills(self, strategy, positions):
        """entry_ts was only used for the 500ms taker-settle guard. A post-only reject
        with no other fill left the pairing in ENTERING with no live orders, forever."""
        ps = self._entering(strategy, positions)
        late = 1_000 + strategy._imbalance_timeout_ns + 1
        strategy._check_fill_transitions(ps, late)
        assert ps.phase != Phase.ENTERING, "ENTERING must expire rather than wedge"

    def test_post_only_reject_clears_the_chase_dedupe(self, strategy):
        """The handler's comment promises a retry, but leaving last_chase_bid set meant
        the dedupe suppressed that price permanently."""
        ps = strategy._pairing_states[0]
        ps.legs = [LegState(listing=lg, target_qty=100 * SIZE_SCALE) for lg in ps.pairing.legs]
        leg = ps.legs[-1]
        leg.last_chase_bid = 43 * CENT
        strategy._on_post_only_reject(ps, leg)
        assert leg.last_chase_bid == 0, "a rejected price must be retryable"


# ---------------------------------------------------------------------------
# 2.8 — UNWINDING must not loop forever
# ---------------------------------------------------------------------------

class TestUnwindEscalates:
    def test_unwind_gives_up_rather_than_retrying_forever(self, strategy, positions):
        """If _align_lot floors the remainder to 0, or the min-notional floor exceeds
        99c, no close intent is emitted; all_at_base stayed false and the pairing
        retried every 2s indefinitely."""
        strategy._position_view = positions
        ps = strategy._pairing_states[0]
        ps.phase = Phase.UNWINDING
        ps.legs = [LegState(listing=lg, target_qty=0) for lg in ps.pairing.legs]
        ps.base_qty = 0
        # A sub-lot remainder that _align_lot floors to zero.
        for lg in ps.pairing.legs:
            positions.set(lg, SIZE_SCALE // 10)
            _fill_book(strategy, lg, 0.42, 0.44)

        ts = 0
        for _ in range(12):
            ts += 3_000_000_000
            strategy._on_closing(ps, ts)
            if ps.phase != Phase.UNWINDING:
                break
        assert ps.phase != Phase.UNWINDING, (
            "a pairing that cannot emit a close intent must escalate, not loop"
        )


# ---------------------------------------------------------------------------
# 2.9 — pairings sharing a listing must not unwind each other's position
# ---------------------------------------------------------------------------

class TestSharedListingOrphanCheck:
    def test_one_pairing_does_not_unwind_anothers_position(self, make_strategy, positions):
        """With N>=3 outcomes a listing belongs to several pairings. The orphan check
        compared the listing's position against THIS pairing's base_qty, so a pairing
        holding nothing saw a sibling's legitimate position as an orphan and tried to
        liquidate it."""
        s = make_strategy(outcomes=[
            {"pm": 222852, "k": 97203},
            {"pm": 222853, "k": 97202},
            {"pm": 900001, "k": 900002},
        ])
        s._position_view = positions

        holder, other = None, None
        for ps in s._pairing_states:
            for lg in ps.pairing.legs:
                for cand in s._listing_to_ps[lg]:
                    if cand is not ps:
                        holder, other, shared = ps, cand, lg
                        break
                if holder:
                    break
            if holder:
                break
        assert holder is not None, "expected a listing shared by two pairings at N=3"

        for lg in set(holder.pairing.legs) | set(other.pairing.legs):
            positions.set(lg, 0)
            _fill_book(s, lg, 0.30, 0.32)
        # holder owns 2000 on the shared listing, parked as its base
        positions.set(shared, 2000 * SIZE_SCALE)
        holder.base_qty = 2000 * SIZE_SCALE
        holder.phase = Phase.SCANNING
        other.phase = Phase.SCANNING
        other.base_qty = 0

        s._on_scanning(other, 10_000)
        assert other.phase != Phase.UNWINDING, (
            "a pairing must not treat a sibling's parked position as its own orphan"
        )
