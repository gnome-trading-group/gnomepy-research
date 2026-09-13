from __future__ import annotations

import math

from gnomepy_research.arb.book_walking import OrderMode
from gnomepy_research.arb.portfolio import ArbLeg, ArbPhase, ArbPortfolio

# Scale helpers
PRICE_SCALE = 1_000_000_000
SIZE_SCALE = 1_000_000


def p(frac: float) -> int:
    return round(frac * PRICE_SCALE)


def q(lots: float) -> int:
    return round(lots * SIZE_SCALE)


POLY = (4, 100)
KALS = (5, 200)
KALS2 = (5, 201)

_DEFAULT_PAIRING = 0
_DEFAULT_MODE = OrderMode.MAKER


def _portfolio(*legs: ArbLeg) -> ArbPortfolio:
    return ArbPortfolio(list(legs), pairing_index=_DEFAULT_PAIRING, order_mode=_DEFAULT_MODE)


# ---------------------------------------------------------------------------
# ArbLeg
# ---------------------------------------------------------------------------

class TestArbLeg:
    def test_record_fill_tracks_qty_and_cost(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        leg.record_fill(fill_price=p(0.37), fill_qty=q(5))
        assert leg.filled_qty == q(5)
        assert math.isclose(leg.fill_cost, 0.37 * 5, rel_tol=1e-9)

    def test_multiple_fills_accumulate(self):
        leg = ArbLeg(listing=KALS, target_qty=q(10))
        leg.record_fill(fill_price=p(0.37), fill_qty=q(4))
        leg.record_fill(fill_price=p(0.38), fill_qty=q(6))
        assert leg.filled_qty == q(10)
        expected_cost = 0.37 * 4 + 0.38 * 6
        assert math.isclose(leg.fill_cost, expected_cost, rel_tol=1e-9)

    def test_avg_fill_price_single_fill(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        leg.record_fill(fill_price=p(0.37), fill_qty=q(5))
        assert math.isclose(leg.avg_fill_price, 0.37, rel_tol=1e-9)

    def test_avg_fill_price_weighted_average(self):
        leg = ArbLeg(listing=KALS, target_qty=q(10))
        leg.record_fill(fill_price=p(0.36), fill_qty=q(4))
        leg.record_fill(fill_price=p(0.38), fill_qty=q(6))
        expected = (0.36 * 4 + 0.38 * 6) / 10
        assert math.isclose(leg.avg_fill_price, expected, rel_tol=1e-9)

    def test_avg_fill_price_zero_when_no_fills(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        assert leg.avg_fill_price == 0.0

    def test_is_filled_when_qty_met(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        leg.record_fill(fill_price=p(0.37), fill_qty=q(5))
        assert leg.is_filled

    def test_is_filled_when_overfilled(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        leg.record_fill(fill_price=p(0.37), fill_qty=q(6))
        assert leg.is_filled

    def test_not_filled_when_partial(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        leg.record_fill(fill_price=p(0.37), fill_qty=q(3))
        assert not leg.is_filled

    def test_not_filled_when_no_fills(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        assert not leg.is_filled

    def test_not_filled_when_target_zero(self):
        leg = ArbLeg(listing=KALS, target_qty=0)
        leg.record_fill(fill_price=p(0.37), fill_qty=q(5))
        assert not leg.is_filled

    def test_reset_clears_state(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        leg.record_fill(fill_price=p(0.37), fill_qty=q(5))
        leg.reset()
        assert leg.filled_qty == 0
        assert leg.fill_cost == 0.0
        assert leg.target_qty == 0


# ---------------------------------------------------------------------------
# ArbPortfolio — containment and state
# ---------------------------------------------------------------------------

class TestArbPortfolioState:
    def test_contains_listed_leg(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        assert KALS in portfolio

    def test_does_not_contain_unlisted(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        assert KALS2 not in portfolio

    def test_record_fill_propagates_to_leg(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        portfolio.record_fill(KALS, fill_price=p(0.37), fill_qty=q(5))
        assert leg.is_filled
        assert math.isclose(leg.avg_fill_price, 0.37, rel_tol=1e-9)

    def test_reset_clears_all_fills(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        portfolio = _portfolio(leg_a, leg_b)
        portfolio.record_fill(KALS, fill_price=p(0.37), fill_qty=q(5))
        portfolio.reset()
        assert not leg_a.is_filled


# ---------------------------------------------------------------------------
# ArbPortfolio — target quantity management and phase
# ---------------------------------------------------------------------------

class TestArbPortfolioTargetAndPhase:
    def test_phase_starts_entering(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        assert portfolio.phase == ArbPhase.ENTERING

    def test_phase_transitions(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        portfolio.phase = ArbPhase.FILLED
        assert portfolio.phase == ArbPhase.FILLED
        portfolio.phase = ArbPhase.EXITING
        assert portfolio.phase == ArbPhase.EXITING

    def test_pairing_index_and_order_mode_stored(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = ArbPortfolio([leg], pairing_index=3, order_mode=OrderMode.TAKER)
        assert portfolio.pairing_index == 3
        assert portfolio.order_mode == OrderMode.TAKER

    def test_target_qty_returns_leg_value(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        assert portfolio.target_qty(KALS) == q(5)

    def test_target_qty_returns_zero_for_unknown(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        assert portfolio.target_qty(KALS2) == 0

    def test_as_target_dict(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(3))
        portfolio = _portfolio(leg_a, leg_b)
        d = portfolio.as_target_dict()
        assert d == {KALS: q(5), KALS2: q(3)}

    def test_set_all_zero(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(3))
        portfolio = _portfolio(leg_a, leg_b)
        portfolio.set_all_zero()
        assert portfolio.target_qty(KALS) == 0
        assert portfolio.target_qty(KALS2) == 0
        assert portfolio.as_target_dict() == {KALS: 0, KALS2: 0}

    def test_begin_exit_zeros_targets_and_sets_exiting(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        portfolio = _portfolio(leg_a, leg_b)
        portfolio.begin_exit()
        assert portfolio.phase == ArbPhase.EXITING
        assert portfolio.target_qty(KALS) == 0
        assert portfolio.target_qty(KALS2) == 0

    def test_add_target_qty_increments_leg(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        portfolio.add_target_qty(KALS, q(3))
        assert portfolio.target_qty(KALS) == q(8)

    def test_add_target_qty_unknown_listing_noop(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        portfolio.add_target_qty(KALS2, q(3))
        assert portfolio.target_qty(KALS) == q(5)

    def test_record_fill_auto_transitions_to_filled(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        portfolio = _portfolio(leg_a, leg_b)
        assert portfolio.phase == ArbPhase.ENTERING
        portfolio.record_fill(KALS, fill_price=p(0.37), fill_qty=q(5))
        assert portfolio.phase == ArbPhase.ENTERING
        portfolio.record_fill(KALS2, fill_price=p(0.60), fill_qty=q(5))
        assert portfolio.phase == ArbPhase.FILLED

    def test_record_fill_no_transition_when_not_entering(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        portfolio = _portfolio(leg)
        portfolio.phase = ArbPhase.FILLED
        portfolio.record_fill(KALS, fill_price=p(0.37), fill_qty=q(5))
        assert portfolio.phase == ArbPhase.FILLED
