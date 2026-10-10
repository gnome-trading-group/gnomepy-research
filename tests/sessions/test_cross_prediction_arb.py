"""Regression tests for cross_prediction_arb.

Each test here corresponds to a defect found by comparing the working copy against
best/ (which scored $106.52 vs $66.55 on the same event) and by reading the state
machine. Every one fails on the pre-fix code.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from gnomepy import Scales
from gnomepy.java.enums import Action
from gnomepy_research.sessions.cross_prediction_arb.cross_prediction_arb import (
    BookLevel,
    LegState,
    Phase,
)
from gnomepy_research.sessions.cross_prediction_arb__multi import strategy as multi
from tests.sessions.conftest import FakePositions, _FakeRegistry

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
        s = make_strategy(registry=_FakeRegistry(min_notional=min_notional))
        listing = s._pairings[0].legs[0]
        floor = s._min_price_for_size(listing, size)
        assert s._passes_notional(listing, floor, size)
        if floor > 0:
            assert not s._passes_notional(listing, floor - 1, size)

    def test_rejects_price_whose_notional_is_short(self, make_strategy):
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


# ---------------------------------------------------------------------------
# Queue-drain fill probability
# ---------------------------------------------------------------------------

SEC = 1_000_000_000


class _Tick:
    """Minimal MBP stand-in for _update_book: one level per side plus an optional print."""

    def __init__(self, bid, ask, size=500, action=Action.ADD.value, price=0.0, qty=0):
        self._bid, self._ask, self._size = int(bid * PRICE_SCALE), int(ask * PRICE_SCALE), size * SIZE_SCALE
        self.action = action
        self.price = int(price * PRICE_SCALE)
        self.size = qty * SIZE_SCALE

    def bid_price(self, i):
        return self._bid if i == 0 else 0

    def bid_size(self, i):
        return self._size if i == 0 else 0

    def ask_price(self, i):
        return self._ask if i == 0 else 0

    def ask_size(self, i):
        return self._size if i == 0 else 0


def _multi(**kwargs):
    with patch.object(multi, "RegistryClient", lambda *a, **k: _FakeRegistry()):
        return multi.CrossPredictionArbMulti(
            outcomes=[{"pm": 222852}, {"pm": 222853}],
            dutch_book_labels=["pm"], taker_labels=[], **kwargs,
        )


def _sell_into_bid(s, lg, ts, qty, queue=500):
    s._update_book(lg, _Tick(0.40, 0.42, size=queue), ts)
    s._update_book(lg, _Tick(0.40, 0.42, size=queue, action=Action.TRADE.value, price=0.40, qty=qty), ts + 1)


class TestFillProbability:
    """P(fill) = 1 - exp(-lambda*T/Q), lambda measured from trades that hit the bid."""

    def test_no_observed_flow_means_no_fill(self):
        s = _multi(fill_risk_horizon_ns=60 * SEC)
        lg = s._pairings[0].legs[0]
        s._update_book(lg, _Tick(0.40, 0.42), SEC)
        assert s._fill_prob(lg, 60 * SEC) == 0.0, "never-traded queue cannot be assumed to fill"

    def test_queue_shrinking_without_trades_is_not_flow(self):
        """Cancellations drain the queue but never fill a resting bid."""
        s = _multi(fill_risk_horizon_ns=60 * SEC)
        lg = s._pairings[0].legs[0]
        for k, q in enumerate(range(5000, 0, -100)):
            s._update_book(lg, _Tick(0.40, 0.42, size=q), (k + 1) * 1_000_000)
        assert s._sell_rate(lg) == 0.0

    def test_buys_lifting_the_ask_are_not_flow(self):
        s = _multi(fill_risk_horizon_ns=60 * SEC)
        lg = s._pairings[0].legs[0]
        s._update_book(lg, _Tick(0.40, 0.42), SEC)
        s._update_book(lg, _Tick(0.40, 0.42, action=Action.TRADE.value, price=0.42, qty=1000), SEC + 1)
        assert s._sell_rate(lg) == 0.0

    def test_rate_matches_traded_volume_over_window(self):
        s = _multi(fill_risk_horizon_ns=60 * SEC, flow_window_ns=600 * SEC)
        lg = s._pairings[0].legs[0]
        _sell_into_bid(s, lg, SEC, 600)
        assert s._sell_rate(lg) == pytest.approx(1.0, rel=1e-6)

    def test_flow_decays_when_trading_stops(self):
        s = _multi(fill_risk_horizon_ns=60 * SEC, flow_window_ns=600 * SEC)
        lg = s._pairings[0].legs[0]
        _sell_into_bid(s, lg, SEC, 600)
        fresh = s._sell_rate(lg)
        s._update_book(lg, _Tick(0.40, 0.42), 1200 * SEC)
        assert s._sell_rate(lg) < 0.2 * fresh

    def test_deeper_queue_is_less_likely_to_fill(self):
        s = _multi(fill_risk_horizon_ns=60 * SEC)
        lg = s._pairings[0].legs[0]
        _sell_into_bid(s, lg, SEC, 600, queue=100)
        shallow = s._fill_prob(lg, 60 * SEC)
        s._update_book(lg, _Tick(0.40, 0.42, size=100_000), SEC + 2)
        deep = s._fill_prob(lg, 60 * SEC)
        assert 0.0 <= deep < shallow <= 1.0

    def test_longer_horizon_raises_fill_probability(self):
        s = _multi(fill_risk_horizon_ns=60 * SEC)
        lg = s._pairings[0].legs[0]
        _sell_into_bid(s, lg, SEC, 600)
        assert s._fill_prob(lg, 300 * SEC) > s._fill_prob(lg, 30 * SEC)

    def test_horizon_zero_disables_the_discount(self):
        """Default must preserve prior behaviour exactly."""
        assert _multi()._fill_risk_horizon_ns == 0


# ---------------------------------------------------------------------------
# Taker completion of the last leg
# ---------------------------------------------------------------------------

def _three_way(**kwargs):
    with patch.object(multi, "RegistryClient", lambda *a, **k: _FakeRegistry()):
        return multi.CrossPredictionArbMulti(
            outcomes=[{"pm": 222852}, {"pm": 222853}, {"pm": 900001}],
            dutch_book_labels=["pm"], taker_labels=[], **kwargs,
        )


def _stuck_on_last_leg(s, paid=(0.50, 0.30), stuck_bid=0.12, stuck_ask=0.14, ask_depth=1000):
    """Two legs fully filled at `paid`, the third resting unfilled at stuck_bid."""
    ps = s._pairing_states[0]
    ps.phase = Phase.PARTIAL_FILL
    ps.partial_fill_since = SEC
    qty = 100 * SIZE_SCALE
    ps.legs = [multi.LegState(listing=lg, target_qty=qty) for lg in ps.pairing.legs]
    for lg, px in zip(ps.legs[:2], paid):
        lg.record_fill(int(px * PRICE_SCALE), qty)
    stuck = ps.legs[2]
    stuck.entry_bid = int(stuck_bid * PRICE_SCALE)
    for lg, px in zip(ps.pairing.legs[:2], paid):
        _fill_book(s, lg, px - 0.01, px + 0.01)
    book = s._books[stuck.listing]
    book.bids = [BookLevel(price=stuck.entry_bid, size=10_000 * SIZE_SCALE)]
    book.asks = [BookLevel(price=int(stuck_ask * PRICE_SCALE), size=ask_depth * SIZE_SCALE)]
    return ps, stuck


class TestTakerCompletion:
    """Cross the last leg when the set still completes below $1 - edge."""

    def test_disabled_by_default(self):
        s = _three_way()
        ps, _ = _stuck_on_last_leg(s)
        assert s._taker_completion(ps, 2 * SEC) == []

    def test_crosses_when_set_stays_profitable(self):
        # 0.50 + 0.30 + 0.14 + fee(0.07*0.14*0.86 ~ 0.84c) ~ 94.8c < 99c
        s = _three_way(taker_complete_edge_cents=1.0)
        ps, stuck = _stuck_on_last_leg(s)
        out = s._taker_completion(ps, 2 * SEC)
        assert len(out) == 1
        it = out[0]
        assert (it.exchange_id, it.security_id) == stuck.listing
        assert it.take_size == 100 * SIZE_SCALE
        assert it.take_limit_price == int(0.14 * PRICE_SCALE)
        assert it.bid_size == 0, "a fully-covered take leaves nothing resting"

    def test_holds_when_crossing_would_lose(self):
        s = _three_way(taker_complete_edge_cents=1.0)
        ps, _ = _stuck_on_last_leg(s, paid=(0.55, 0.35), stuck_ask=0.10)
        assert s._taker_completion(ps, 2 * SEC) == []

    def test_fees_already_paid_count_against_the_set(self):
        s = _three_way(taker_complete_edge_cents=1.0)
        ps, _ = _stuck_on_last_leg(s, paid=(0.50, 0.30), stuck_ask=0.17)
        assert s._taker_completion(ps, 2 * SEC), "precondition: crossable without fees"
        ps.completion_ts = 0
        ps.legs[0].entry_fees = 2.0   # $2 on 100 contracts = 2c per set
        assert s._taker_completion(ps, 2 * SEC) == []

    def test_thin_ask_takes_touch_and_rests_the_remainder(self):
        s = _three_way(taker_complete_edge_cents=1.0)
        ps, stuck = _stuck_on_last_leg(s, ask_depth=30)
        it = s._taker_completion(ps, 2 * SEC)[0]
        assert it.take_size == 30 * SIZE_SCALE
        assert it.bid_size == 70 * SIZE_SCALE
        assert it.bid_price == stuck.entry_bid
        assert it.post_only

    def test_no_resend_inside_settle_window(self):
        s = _three_way(taker_complete_edge_cents=1.0)
        ps, _ = _stuck_on_last_leg(s, ask_depth=30)
        assert s._taker_completion(ps, 2 * SEC)
        assert s._taker_completion(ps, 2 * SEC + 100_000_000) == []
        assert s._taker_completion(ps, 3 * SEC)

    def test_two_unfilled_legs_take_equal_size(self):
        """Unequal takes would themselves unbalance the set."""
        s = _three_way(taker_complete_edge_cents=1.0)
        ps, _ = _stuck_on_last_leg(s, paid=(0.50, 0.30), stuck_ask=0.14, ask_depth=40)
        ps.legs[1].filled_qty = 0
        ps.legs[1].fill_cost = 0
        ps.legs[1].entry_bid = int(0.29 * PRICE_SCALE)
        s._books[ps.legs[1].listing].asks = [BookLevel(price=int(0.31 * PRICE_SCALE), size=25 * SIZE_SCALE)]
        out = s._taker_completion(ps, 2 * SEC)
        assert len(out) == 2
        assert {i.take_size for i in out} == {25 * SIZE_SCALE}

    def test_nothing_filled_means_nothing_to_complete(self):
        s = _three_way(taker_complete_edge_cents=1.0)
        ps, _ = _stuck_on_last_leg(s)
        for lg in ps.legs:
            lg.filled_qty = 0
            lg.fill_cost = 0
        assert s._taker_completion(ps, 2 * SEC) == []

    def test_rest_never_crosses_a_fallen_ask(self):
        """Resting at entry_bid after the ask drops through it is post-only rejected,
        and that reject unwound the whole pairing in the soccer backtest."""
        s = _three_way(taker_complete_edge_cents=1.0)
        ps, stuck = _stuck_on_last_leg(s, stuck_bid=0.20, stuck_ask=0.17, ask_depth=30)
        it = s._taker_completion(ps, 2 * SEC)[0]
        assert it.bid_size == 70 * SIZE_SCALE
        assert it.bid_price < int(0.17 * PRICE_SCALE)


class TestClosingRetryKeepsHedgedSets:
    def test_retry_does_not_sell_complete_sets(self):
        """First close kept 1792 complete sets; the 2s retry used base_qty=0 and sold them."""
        s = _three_way()
        positions = FakePositions()
        s._position_view = positions
        ps = s._pairing_states[0]
        ps.phase = Phase.UNWINDING
        ps.base_qty = 0
        ps.legs = [multi.LegState(listing=lg, target_qty=2506 * SIZE_SCALE) for lg in ps.pairing.legs]
        for lg in ps.pairing.legs:
            positions.set(lg, 1792 * SIZE_SCALE)
            _fill_book(s, lg, 0.20, 0.22)
        assert s._on_closing(ps, 10 * SEC) == []
        assert ps.phase == Phase.SCANNING

    def test_retry_sheds_only_the_naked_excess(self):
        s = _three_way()
        positions = FakePositions()
        s._position_view = positions
        ps = s._pairing_states[0]
        ps.phase = Phase.UNWINDING
        ps.legs = [multi.LegState(listing=lg, target_qty=2506 * SIZE_SCALE) for lg in ps.pairing.legs]
        for lg, q in zip(ps.pairing.legs, (2506, 1792, 2506)):
            positions.set(lg, q * SIZE_SCALE)
            _fill_book(s, lg, 0.20, 0.22)
        out = s._on_closing(ps, 10 * SEC)
        sold = {(i.exchange_id, i.security_id): i.take_size for i in out if i.take_size}
        assert sold == {ps.pairing.legs[0]: 714 * SIZE_SCALE, ps.pairing.legs[2]: 714 * SIZE_SCALE}


# ---------------------------------------------------------------------------
# Lead with the illiquid leg
# ---------------------------------------------------------------------------

def _scannable(s, bids=(0.50, 0.30, 0.10), ts=SEC):
    positions = FakePositions()
    s._position_view = positions
    for lg, b in zip(s._pairings[0].legs, bids):
        positions.set(lg, 0)
        _fill_book(s, lg, b, b + 0.02)
        s._books[lg].last_update_ts = ts
    s._now_ns = ts
    return s._pairing_states[0]


class TestLeadWithIlliquid:
    def test_entry_posts_only_the_least_fillable_leg(self):
        s = _three_way(lead_with_illiquid=True, min_contract_price=0.03)
        ps = _scannable(s)
        legs = s._pairings[0].legs
        for lg in legs[:2]:
            s._sell_flow[lg] = (6_000.0, SEC)   # 10 c/s over the 600s window
        out = s._on_scanning(ps, SEC)
        assert [(i.exchange_id, i.security_id) for i in out] == [legs[2]]
        assert [lg.posted for lg in ps.legs] == [False, False, True]

    def test_ties_go_to_the_cheapest_leg(self):
        s = _three_way(lead_with_illiquid=True, min_contract_price=0.03)
        ps = _scannable(s, bids=(0.30, 0.10, 0.50))
        out = s._on_scanning(ps, SEC)
        assert [(i.exchange_id, i.security_id) for i in out] == [s._pairings[0].legs[1]]

    def test_default_posts_every_leg(self):
        s = _three_way(min_contract_price=0.03)
        ps = _scannable(s)
        assert len(s._on_scanning(ps, SEC)) == 3

    def _lead_filled(self, s, lead_px=0.10, filled=100):
        ps = s._pairing_states[0]
        ps.phase = Phase.PARTIAL_FILL
        ps.partial_fill_since = SEC
        ps.legs = [multi.LegState(listing=lg, target_qty=100 * SIZE_SCALE, posted=False)
                   for lg in ps.pairing.legs]
        lead = ps.legs[2]
        lead.posted = True
        lead.record_fill(int(lead_px * PRICE_SCALE), filled * SIZE_SCALE)
        return ps

    def test_held_legs_post_one_at_a_time_least_fillable_first(self):
        s = _three_way(lead_with_illiquid=True)
        _scannable(s)
        ps = self._lead_filled(s)
        first = s._post_deferred_legs(ps, 2 * SEC)
        assert [(i.exchange_id, i.security_id) for i in first] == [ps.pairing.legs[1]], (
            "no flow anywhere, so the cheaper of the two held legs goes next")
        assert first[0].bid_size == 100 * SIZE_SCALE and first[0].post_only
        assert s._post_deferred_legs(ps, 3 * SEC) == [], "wait for it to fill"
        ps.legs[1].record_fill(int(0.30 * PRICE_SCALE), 100 * SIZE_SCALE)
        second = s._post_deferred_legs(ps, 4 * SEC)
        assert [(i.exchange_id, i.security_id) for i in second] == [ps.pairing.legs[0]]
        assert all(lg.posted for lg in ps.legs)

    def test_partial_lead_holds_the_rest_back(self):
        s = _three_way(lead_with_illiquid=True)
        _scannable(s)
        ps = self._lead_filled(s, filled=40)
        assert s._post_deferred_legs(ps, 2 * SEC) == []

    def test_unprofitable_after_lead_posts_nothing(self):
        s = _three_way(lead_with_illiquid=True)
        _scannable(s, bids=(0.55, 0.35, 0.10))
        ps = self._lead_filled(s, lead_px=0.10)
        assert s._post_deferred_legs(ps, 2 * SEC) == []
        assert not ps.legs[0].posted

    def test_completion_crosses_deferred_legs_when_affordable(self):
        s = _three_way(lead_with_illiquid=True, taker_complete_edge_cents=1.0)
        _scannable(s)
        ps = self._lead_filled(s)
        out = s._taker_completion(ps, 2 * SEC)
        assert len(out) == 2 and all(i.take_size == 100 * SIZE_SCALE for i in out)

    def test_completion_waits_for_the_lead(self):
        s = _three_way(lead_with_illiquid=True, taker_complete_edge_cents=1.0)
        _scannable(s)
        ps = self._lead_filled(s, filled=40)
        assert s._taker_completion(ps, 2 * SEC) == []
