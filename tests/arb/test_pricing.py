from __future__ import annotations

import math
from math import exp, log

import pytest

from gnomepy_research.arb.book_walking import OrderMode
from gnomepy_research.arb.portfolio import ArbLeg, ArbPortfolio
from gnomepy_research.arb.pricing import (
    AggressivePriceModel,
    ArbBudgetConstraint,
    ArbContext,
    FillProbTargetModel,
    JoinBestBidModel,
    OptimalEVModel,
)

PRICE_SCALE = 1_000_000_000
SIZE_SCALE = 1_000_000
ONE_CENT = PRICE_SCALE // 100


def p(frac: float) -> int:
    return round(frac * PRICE_SCALE)


def q(lots: float) -> int:
    return round(lots * SIZE_SCALE)


POLY = (4, 100)
KALS = (5, 200)
KALS2 = (5, 201)
KALS3 = (5, 202)

POLY_RATES = {4: 0.0}
KALS_RATES = {5: 0.0175}
BOTH_RATES = {4: 0.0, 5: 0.0175}


def _fee(rate: float, price: float) -> float:
    return rate * price * (1.0 - price)


def _constraint(*legs: ArbLeg, rates: dict, bps: float = 5.0) -> ArbBudgetConstraint:
    return ArbBudgetConstraint(list(legs), min_edge_bps=bps, maker_fee_rates=rates)


# ---------------------------------------------------------------------------
# ArbBudgetConstraint
# ---------------------------------------------------------------------------

class TestArbBudgetConstraintLegCost:
    def test_leg_cost_unfilled_uses_order_price_when_set(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        c = _constraint(leg, rates=KALS_RATES)
        c.update_bid(KALS, p(0.50))
        c.update_order_price(KALS, p(0.37))
        cost = c.leg_cost(KALS)
        expected = 0.37 + _fee(0.0175, 0.37)
        assert math.isclose(cost, expected, rel_tol=1e-9)

    def test_leg_cost_unfilled_falls_back_to_bid_when_no_order_price(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        c = _constraint(leg, rates=KALS_RATES)
        c.update_bid(KALS, p(0.37))
        cost = c.leg_cost(KALS)
        expected = 0.37 + _fee(0.0175, 0.37)
        assert math.isclose(cost, expected, rel_tol=1e-9)

    def test_leg_cost_unfilled_falls_back_to_bid_when_order_price_zero(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        c = _constraint(leg, rates=KALS_RATES)
        c.update_bid(KALS, p(0.37))
        c.update_order_price(KALS, 0)
        cost = c.leg_cost(KALS)
        expected = 0.37 + _fee(0.0175, 0.37)
        assert math.isclose(cost, expected, rel_tol=1e-9)

    def test_leg_cost_zero_fee_rate(self):
        leg = ArbLeg(listing=POLY, target_qty=q(5))
        c = _constraint(leg, rates=POLY_RATES)
        c.update_bid(POLY, p(0.37))
        assert math.isclose(c.leg_cost(POLY), 0.37, rel_tol=1e-9)

    def test_leg_cost_no_data_returns_none(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        c = _constraint(leg, rates=KALS_RATES)
        assert c.leg_cost(KALS) is None

    def test_leg_cost_after_fill_uses_avg_fill_price(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        c = _constraint(leg, rates=KALS_RATES)
        c.record_fill(KALS, fill_price=p(0.37), fill_qty=q(5))
        c.update_bid(KALS, p(0.50))
        cost = c.leg_cost(KALS)
        expected = 0.37 + _fee(0.0175, 0.37)
        assert math.isclose(cost, expected, rel_tol=1e-9)

    def test_leg_cost_after_fill_ignores_missing_bid(self):
        leg = ArbLeg(listing=KALS, target_qty=q(5))
        c = _constraint(leg, rates=KALS_RATES)
        c.record_fill(KALS, fill_price=p(0.37), fill_qty=q(5))
        cost = c.leg_cost(KALS)
        assert cost is not None
        assert math.isclose(cost, 0.37 + _fee(0.0175, 0.37), rel_tol=1e-9)

    def test_record_fill_multiple_partial_fills(self):
        leg = ArbLeg(listing=KALS, target_qty=q(10))
        c = _constraint(leg, rates=KALS_RATES)
        c.record_fill(KALS, fill_price=p(0.36), fill_qty=q(4))
        c.record_fill(KALS, fill_price=p(0.38), fill_qty=q(6))
        avg = (0.36 * 4 + 0.38 * 6) / 10
        expected = avg + _fee(0.0175, avg)
        assert math.isclose(c.leg_cost(KALS), expected, rel_tol=1e-4)


class TestArbBudgetConstraintCombined:
    def test_combined_cost_sums_both_legs(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS, p(0.61))
        c.update_bid(KALS2, p(0.37))
        cost_a = 0.61 + _fee(0.0175, 0.61)
        cost_b = 0.37 + _fee(0.0175, 0.37)
        assert math.isclose(c.combined_cost(), cost_a + cost_b, rel_tol=1e-9)

    def test_combined_cost_none_if_any_leg_unknown(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS, p(0.61))
        assert c.combined_cost() is None

    def test_edge_bps_profitable(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS, p(0.61))
        c.update_bid(KALS2, p(0.35))
        assert c.edge_bps() is not None
        assert c.edge_bps() > 0

    def test_edge_bps_unprofitable(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS, p(0.63))
        c.update_bid(KALS2, p(0.37))
        assert c.edge_bps() is not None
        assert c.edge_bps() < 0

    def test_is_economically_valid_above_threshold(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS, p(0.61))
        c.update_bid(KALS2, p(0.35))
        assert c.is_economically_valid()

    def test_is_economically_valid_below_threshold(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS, p(0.63))
        c.update_bid(KALS2, p(0.37))
        assert not c.is_economically_valid()

    def test_is_economically_valid_true_when_data_missing(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS, p(0.63))
        assert c.is_economically_valid()


class TestArbBudgetConstraintMaxPrice:
    def test_max_price_zero_fee_rate(self):
        leg_a = ArbLeg(listing=POLY, target_qty=q(5))
        leg_b = ArbLeg(listing=(4, 101), target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates={4: 0.0})
        c.update_bid((4, 101), p(0.37))
        budget = 1.0 - 5.0 / 10000 - 0.37
        expected = int(budget * PRICE_SCALE)
        result = c.max_price(POLY)
        assert result is not None
        assert abs(result - expected) <= 1

    def test_max_price_nonzero_fee_rate(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS2, p(0.37))
        sibling_cost = 0.37 + _fee(0.0175, 0.37)
        budget = 1.0 - 5.0 / 10000 - sibling_cost
        r = 0.0175
        discriminant = (1 + r) ** 2 - 4 * r * budget
        expected_max = ((1 + r) - discriminant ** 0.5) / (2 * r)
        expected_int = int(expected_max * PRICE_SCALE)
        result = c.max_price(KALS)
        assert result is not None
        assert abs(result - expected_int) <= 1

    def test_max_price_budget_exhausted(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS2, p(1.0))
        assert c.max_price(KALS) == 0

    def test_max_price_sibling_unknown_returns_none(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        assert c.max_price(KALS) is None

    def test_max_price_uses_locked_fill_for_filled_sibling(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.record_fill(KALS2, fill_price=p(0.37), fill_qty=q(5))
        c.update_bid(KALS2, p(0.50))
        sibling_cost = 0.37 + _fee(0.0175, 0.37)
        budget = 1.0 - 5.0 / 10000 - sibling_cost
        r = 0.0175
        discriminant = (1 + r) ** 2 - 4 * r * budget
        expected_max = ((1 + r) - discriminant ** 0.5) / (2 * r)
        expected_int = int(expected_max * PRICE_SCALE)
        result = c.max_price(KALS)
        assert result is not None
        assert abs(result - expected_int) <= 1

    def test_max_price_three_legs(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        leg_c = ArbLeg(listing=KALS3, target_qty=q(5))
        c = _constraint(leg_a, leg_b, leg_c, rates=KALS_RATES)
        c.update_bid(KALS2, p(0.30))
        c.update_bid(KALS3, p(0.25))
        cost_b = 0.30 + _fee(0.0175, 0.30)
        cost_c = 0.25 + _fee(0.0175, 0.25)
        budget = 1.0 - 5.0 / 10000 - cost_b - cost_c
        r = 0.0175
        discriminant = (1 + r) ** 2 - 4 * r * budget
        expected_max = ((1 + r) - discriminant ** 0.5) / (2 * r)
        expected_int = int(expected_max * PRICE_SCALE)
        result = c.max_price(KALS)
        assert result is not None
        assert abs(result - expected_int) <= 1

    def test_max_price_caps_at_one(self):
        leg_a = ArbLeg(listing=POLY, target_qty=q(5))
        leg_b = ArbLeg(listing=(4, 101), target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates={4: 0.0})
        c.update_bid((4, 101), p(0.0))
        result = c.max_price(POLY)
        assert result is not None
        assert result <= PRICE_SCALE

    def test_max_price_is_binding(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.record_fill(KALS2, fill_price=p(0.37), fill_qty=q(5))
        max_price = c.max_price(KALS)
        assert max_price is not None
        p_max = max_price / PRICE_SCALE
        combined = (p_max + _fee(0.0175, p_max)) + (0.37 + _fee(0.0175, 0.37))
        assert combined <= 1.0 - 4.9 / 10000

    def test_budget_remaining_bps(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS2, p(0.37))
        c.update_bid(KALS, p(0.50))
        bps = c.budget_remaining_bps(KALS)
        assert bps is not None
        mp = c.max_price(KALS)
        expected = (mp - p(0.50)) / PRICE_SCALE * 10000
        assert math.isclose(bps, expected, rel_tol=1e-6)

    def test_budget_remaining_bps_none_when_bid_unknown(self):
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        c.update_bid(KALS2, p(0.37))
        assert c.budget_remaining_bps(KALS) is None


class TestArbBudgetConstraintShouldCancel:
    def _maker_portfolio_and_constraint(self) -> tuple[ArbPortfolio, ArbBudgetConstraint]:
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        portfolio = ArbPortfolio([leg_a, leg_b], pairing_index=0, order_mode=OrderMode.MAKER)
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        return portfolio, c

    def _taker_portfolio_and_constraint(self) -> tuple[ArbPortfolio, ArbBudgetConstraint]:
        leg_a = ArbLeg(listing=KALS, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS2, target_qty=q(5))
        portfolio = ArbPortfolio([leg_a, leg_b], pairing_index=0, order_mode=OrderMode.TAKER)
        c = _constraint(leg_a, leg_b, rates=KALS_RATES)
        return portfolio, c

    def test_taker_always_false_no_data(self):
        portfolio, c = self._taker_portfolio_and_constraint()
        assert not portfolio.should_cancel_entry(c)

    def test_taker_always_false_even_when_unprofitable(self):
        portfolio, c = self._taker_portfolio_and_constraint()
        c.update_bid(KALS, p(0.63))
        c.update_bid(KALS2, p(0.37))
        assert not c.is_economically_valid()
        assert not portfolio.should_cancel_entry(c)

    def test_maker_valid_does_not_cancel(self):
        portfolio, c = self._maker_portfolio_and_constraint()
        c.update_bid(KALS, p(0.61))
        c.update_bid(KALS2, p(0.35))
        assert c.is_economically_valid()
        assert not portfolio.should_cancel_entry(c)

    def test_maker_invalid_edge_cancels(self):
        portfolio, c = self._maker_portfolio_and_constraint()
        c.update_bid(KALS, p(0.63))
        c.update_bid(KALS2, p(0.37))
        assert not c.is_economically_valid()
        assert portfolio.should_cancel_entry(c)

    def test_maker_market_bid_exceeds_max_price_no_order_price_cancels(self):
        # When no order price is set, leg_cost falls back to market bid.
        # Market bid > max_price → combined cost exceeds budget → cancel.
        portfolio, c = self._maker_portfolio_and_constraint()
        c.update_bid(KALS2, p(0.35))
        max_price = c.max_price(KALS)
        assert max_price is not None
        c.update_bid(KALS, max_price + 1)
        assert portfolio.should_cancel_entry(c)

    def test_maker_market_bid_exceeds_max_price_but_order_price_below_does_not_cancel(self):
        # Market bid is above max_price, but our order is well within budget.
        # should_cancel_entry evaluates using order price → arb still viable → no cancel.
        portfolio, c = self._maker_portfolio_and_constraint()
        c.update_bid(KALS2, p(0.35))
        max_price = c.max_price(KALS)
        assert max_price is not None
        c.update_bid(KALS, max_price + 1)          # market bid above ceiling
        c.update_order_price(KALS, max_price - 1)  # our order well within budget
        assert not portfolio.should_cancel_entry(c)

    def test_maker_order_price_above_max_price_cancels(self):
        # Our own order price exceeds max_price → combined cost over budget → cancel.
        portfolio, c = self._maker_portfolio_and_constraint()
        c.update_bid(KALS2, p(0.35))
        max_price = c.max_price(KALS)
        assert max_price is not None
        c.update_order_price(KALS, max_price + 1)
        assert portfolio.should_cancel_entry(c)

    def test_maker_missing_bids_does_not_cancel(self):
        portfolio, c = self._maker_portfolio_and_constraint()
        assert not portfolio.should_cancel_entry(c)


# ---------------------------------------------------------------------------
# JoinBestBidModel
# ---------------------------------------------------------------------------

class TestJoinBestBidModel:
    def test_returns_best_bid_when_no_constraint(self):
        model = JoinBestBidModel()
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.42), max_price=None)
        assert result.price == p(0.40)

    def test_capped_by_max_price(self):
        model = JoinBestBidModel()
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.42), max_price=p(0.38))
        assert result.price == p(0.38)

    def test_returns_zero_when_best_bid_zero(self):
        model = JoinBestBidModel()
        result = model.compute_price(KALS, best_bid=0, best_ask=p(0.42), max_price=None)
        assert result.price == 0

    def test_never_crosses_ask(self):
        model = JoinBestBidModel()
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.41), max_price=p(0.45))
        assert result.price < p(0.41)

    def test_fill_prob_in_result(self):
        model = JoinBestBidModel(fill_risk_lambda=10.0)
        model.on_book_update(KALS, [(p(0.40), q(100))], [(p(0.42), q(100))], 0)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.42), max_price=None)
        assert 0.0 < result.fill_prob < 1.0
        assert result.spread > 0.0

    def test_max_price_none_does_not_cap(self):
        model = JoinBestBidModel()
        result = model.compute_price(KALS, best_bid=p(0.60), best_ask=p(0.62), max_price=None)
        assert result.price == p(0.60)


# ---------------------------------------------------------------------------
# AggressivePriceModel
# ---------------------------------------------------------------------------

class TestAggressivePriceModel:
    def _model(self, base_improve_bps: float = 50.0, edge_share: float = 0.5, **kwargs) -> AggressivePriceModel:
        return AggressivePriceModel(base_improve_bps=base_improve_bps, edge_share=edge_share, **kwargs)

    def _update(self, model, listing, bid, ask):
        model.on_book_update(listing, [(p(bid), q(100))], [(p(ask), q(100))], 0)

    def test_improves_on_best_bid(self):
        model = self._model()
        self._update(model, KALS, 0.40, 0.42)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.42), max_price=p(0.63))
        assert result.price > p(0.40)

    def test_capped_by_ask(self):
        model = self._model(base_improve_bps=10000.0)
        self._update(model, KALS, 0.40, 0.42)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.42), max_price=p(0.63))
        assert result.price <= p(0.42)

    def test_capped_by_max_price(self):
        model = self._model(base_improve_bps=10000.0)
        self._update(model, KALS, 0.40, 0.60)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.60), max_price=p(0.45))
        assert result.price <= p(0.45)

    def test_zero_improvement_returns_best_bid(self):
        model = self._model(base_improve_bps=0.0, edge_share=0.0)
        self._update(model, KALS, 0.40, 0.42)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.42), max_price=p(0.63))
        assert result.price == p(0.40)

    def test_no_improvement_when_no_edge_room(self):
        model = self._model()
        self._update(model, KALS, 0.40, 0.42)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.42), max_price=p(0.40))
        assert result.price == p(0.40)

    def test_edge_share_controls_aggressiveness(self):
        model_lo = AggressivePriceModel(base_improve_bps=100000.0, edge_share=0.1)
        model_hi = AggressivePriceModel(base_improve_bps=100000.0, edge_share=0.9)
        for m in (model_lo, model_hi):
            m.on_book_update(KALS, [(p(0.40), q(100))], [(p(0.60), q(100))], 0)
        r_lo = model_lo.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.60), max_price=p(0.55))
        r_hi = model_hi.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.60), max_price=p(0.55))
        assert r_hi.price > r_lo.price

    def test_fill_prob_higher_when_improved(self):
        model_passive = JoinBestBidModel(fill_risk_lambda=10.0)
        model_aggr = self._model(fill_risk_lambda=10.0)
        for m in (model_passive, model_aggr):
            m.on_book_update(KALS, [(p(0.40), q(100))], [(p(0.50), q(100))], 0)
        r_passive = model_passive.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.50), max_price=p(0.60))
        r_aggr = model_aggr.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.50), max_price=p(0.60))
        assert r_aggr.fill_prob > r_passive.fill_prob

    def test_zero_spread_returns_best_bid(self):
        model = self._model()
        result = model.compute_price(KALS, best_bid=0, best_ask=0, max_price=None)
        assert result.price == 0

    def test_never_exceeds_ask(self):
        model = self._model(base_improve_bps=100000.0)
        self._update(model, KALS, 0.40, 0.41)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.41), max_price=None)
        assert result.price <= p(0.41)


# ---------------------------------------------------------------------------
# FillProbTargetModel
# ---------------------------------------------------------------------------

class TestFillProbTargetModel:
    def test_achieves_target_fill_prob(self):
        target_p = 0.7
        lam = 10.0
        model = FillProbTargetModel(target_fill_prob=target_p, fill_risk_lambda=lam)
        model.on_book_update(KALS, [(p(0.40), q(100))], [(p(0.50), q(100))], 0)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.50), max_price=None)
        effective_spread = (p(0.50) - result.price) / PRICE_SCALE
        achieved_p = exp(-lam * effective_spread)
        assert math.isclose(achieved_p, target_p, rel_tol=0.05)

    def test_clamped_at_low_end(self):
        model = FillProbTargetModel(target_fill_prob=0.01, fill_risk_lambda=10.0)
        model.on_book_update(KALS, [(p(0.40), q(100))], [(p(0.42), q(100))], 0)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.42), max_price=None)
        assert result.price >= 0

    def test_clamped_by_max_price(self):
        model = FillProbTargetModel(target_fill_prob=0.99, fill_risk_lambda=10.0)
        model.on_book_update(KALS, [(p(0.40), q(100))], [(p(0.50), q(100))], 0)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.50), max_price=p(0.43))
        assert result.price <= p(0.43)

    def test_never_crosses_ask(self):
        model = FillProbTargetModel(target_fill_prob=0.999, fill_risk_lambda=10.0)
        model.on_book_update(KALS, [(p(0.40), q(100))], [(p(0.41), q(100))], 0)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.41), max_price=None)
        assert result.price < p(0.41)

    def test_invalid_prob_raises(self):
        import pytest
        with pytest.raises(ValueError):
            FillProbTargetModel(target_fill_prob=0.0)
        with pytest.raises(ValueError):
            FillProbTargetModel(target_fill_prob=1.0)

    def test_returns_zero_when_no_book(self):
        model = FillProbTargetModel()
        result = model.compute_price(KALS, best_bid=0, best_ask=0, max_price=None)
        assert result.price == 0


# ---------------------------------------------------------------------------
# OptimalEVModel
# ---------------------------------------------------------------------------

class TestOptimalEVModel:
    def _model(self, lam: float = 10.0) -> OptimalEVModel:
        return OptimalEVModel(fill_risk_lambda=lam)

    def _update(self, model, listing, bid, ask):
        model.on_book_update(listing, [(p(bid), q(100))], [(p(ask), q(100))], 0)

    def _ctx(self, stuck_cost: float = 0.0) -> ArbContext:
        return ArbContext(stuck_cost=stuck_cost)

    def test_optimal_price_below_ceiling(self):
        # p* = ceiling - 1/lambda when stuck_cost=0 and p* > best_bid
        # max_price=0.50 < ask-1c=0.89 → effective ceiling = 0.50
        # 1/lambda=0.10 → p*=0.40 > best_bid=0.20
        lam = 10.0
        model = self._model(lam)
        self._update(model, KALS, 0.20, 0.90)
        result = model.compute_price(KALS, best_bid=p(0.20), best_ask=p(0.90), max_price=p(0.50), context=self._ctx())
        expected = p(0.50) - int(1.0 / lam * PRICE_SCALE)
        assert abs(result.price - expected) <= 1

    def test_stuck_cost_shifts_price_down(self):
        # max_price=0.50 < ask-1c=0.89 → ceiling=0.50; stuck_cost shifts p* down
        lam = 10.0
        model = self._model(lam)
        self._update(model, KALS, 0.20, 0.90)
        r0 = model.compute_price(KALS, best_bid=p(0.20), best_ask=p(0.90), max_price=p(0.50), context=self._ctx(0.0))
        r1 = model.compute_price(KALS, best_bid=p(0.20), best_ask=p(0.90), max_price=p(0.50), context=self._ctx(0.05))
        assert r1.price < r0.price

    def test_infeasible_returns_zero(self):
        # ceiling=0.55, best_bid=0.50, stuck_cost=0.10 → net_value = 0.05 <= stuck_cost → infeasible
        model = self._model(10.0)
        self._update(model, KALS, 0.50, 0.60)
        result = model.compute_price(KALS, best_bid=p(0.50), best_ask=p(0.60), max_price=p(0.55), context=self._ctx(0.10))
        assert result.price == 0

    def test_bids_at_best_bid_when_optimal_below(self):
        # 1/lambda=0.10, ceiling=0.55 → p*=0.45 < best_bid=0.50
        # but net_value(0.50) = 0.55-0.50-0.0 = 0.05 > 0 → bid at best_bid
        model = self._model(10.0)
        self._update(model, KALS, 0.50, 0.60)
        result = model.compute_price(KALS, best_bid=p(0.50), best_ask=p(0.60), max_price=p(0.55), context=self._ctx(0.0))
        assert result.price == p(0.50)

    def test_never_exceeds_ask(self):
        model = self._model(1000.0)
        self._update(model, KALS, 0.40, 0.41)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.41), max_price=p(0.99))
        assert result.price <= p(0.41)

    def test_zero_book_returns_zero(self):
        model = self._model()
        result = model.compute_price(KALS, best_bid=0, best_ask=0, max_price=None)
        assert result.price == 0

    def test_fill_prob_in_result(self):
        lam = 10.0
        model = self._model(lam)
        self._update(model, KALS, 0.20, 0.60)
        result = model.compute_price(KALS, best_bid=p(0.20), best_ask=p(0.60), max_price=p(0.80))
        expected_fp = exp(-lam * (p(0.60) - result.price) / PRICE_SCALE)
        assert math.isclose(result.fill_prob, expected_fp, rel_tol=1e-6)

    def test_no_max_price_uses_ask_ceiling(self):
        # No constraint → uses ask as ceiling, never infeasible-rejected
        model = self._model(10.0)
        self._update(model, KALS, 0.40, 0.60)
        result = model.compute_price(KALS, best_bid=p(0.40), best_ask=p(0.60), max_price=None, context=self._ctx(0.99))
        assert result.price > 0

    def test_high_lambda_bids_near_ceiling(self):
        # lambda=100 → 1/lambda=0.01 → p* = ceiling - 0.01, very close to ceiling
        lam = 100.0
        model = self._model(lam)
        self._update(model, KALS, 0.20, 0.60)
        result = model.compute_price(KALS, best_bid=p(0.20), best_ask=p(0.60), max_price=p(0.80))
        ceiling = p(0.60)  # clamped by ask
        assert result.price >= ceiling - ONE_CENT

    def test_low_lambda_bids_far_below_ceiling(self):
        # lambda=2 → 1/lambda=0.50 → price pulled far down
        lam = 2.0
        model = self._model(lam)
        self._update(model, KALS, 0.10, 0.90)
        result = model.compute_price(KALS, best_bid=p(0.10), best_ask=p(0.90), max_price=p(0.85))
        high_lam_model = self._model(50.0)
        high_lam_model.on_book_update(KALS, [(p(0.10), q(100))], [(p(0.90), q(100))], 0)
        r_high = high_lam_model.compute_price(KALS, best_bid=p(0.10), best_ask=p(0.90), max_price=p(0.85))
        assert result.price < r_high.price

    def test_context_none_treats_stuck_cost_as_zero(self):
        lam = 10.0
        model = self._model(lam)
        self._update(model, KALS, 0.20, 0.60)
        r_none = model.compute_price(KALS, best_bid=p(0.20), best_ask=p(0.60), max_price=p(0.80), context=None)
        r_zero = model.compute_price(KALS, best_bid=p(0.20), best_ask=p(0.60), max_price=p(0.80), context=self._ctx(0.0))
        assert r_none.price == r_zero.price

    def test_max_price_above_ask_bids_at_or_inside_spread(self):
        # max_price=0.55 > ask=0.50; spread=2c; lambda=30 → 1/λ=0.033
        # value_ceiling=0.55, p*=0.55-0.033=0.517, clamped to ask=0.50
        lam = 30.0
        model = self._model(lam)
        self._update(model, KALS, 0.48, 0.50)
        result = model.compute_price(KALS, best_bid=p(0.48), best_ask=p(0.50), max_price=p(0.55), context=self._ctx())
        assert result.price > p(0.48), "should bid above best_bid"
        assert result.price <= p(0.50), "should not exceed ask"

    def test_max_price_above_ask_clamped_to_ask(self):
        # max_price=0.90 >> ask=0.50; p* gets clamped to placement ceiling (ask)
        lam = 30.0
        model = self._model(lam)
        self._update(model, KALS, 0.48, 0.50)
        result = model.compute_price(KALS, best_bid=p(0.48), best_ask=p(0.50), max_price=p(0.90), context=self._ctx())
        assert result.price == p(0.50)  # pinned to ask

    def test_max_price_below_ask_stays_inside_spread(self):
        # max_price=0.49 < ask=0.50; ceiling = min(ask, max_price) = 0.49
        # value_ceiling=0.49, p*=0.49-0.033=0.457, clamped to best_bid=0.48
        lam = 30.0
        model = self._model(lam)
        self._update(model, KALS, 0.48, 0.50)
        result = model.compute_price(KALS, best_bid=p(0.48), best_ask=p(0.50), max_price=p(0.49), context=self._ctx())
        assert result.price == p(0.48)


# ---------------------------------------------------------------------------
# ArbBudgetConstraint taker fee accounting
# ---------------------------------------------------------------------------

POLY_TAKER_RATES = {4: 0.07}
KALS_TAKER_RATES = {5: 0.07}


class TestArbBudgetConstraintTakerFees:
    def test_leg_cost_uses_taker_fee_when_crossing(self):
        leg_a = ArbLeg(listing=POLY, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS, target_qty=q(5))
        c = ArbBudgetConstraint([leg_a, leg_b], min_edge_bps=2.0, maker_fee_rates=POLY_RATES | KALS_RATES, taker_fee_rates=POLY_TAKER_RATES | KALS_TAKER_RATES)
        c.update_order_price(POLY, p(0.43), is_crossing=True)
        cost = c.leg_cost(POLY)
        assert cost is not None
        expected = 0.43 + _fee(0.07, 0.43)
        assert abs(cost - expected) < 1e-9

    def test_leg_cost_uses_maker_fee_when_not_crossing(self):
        leg_a = ArbLeg(listing=POLY, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS, target_qty=q(5))
        c = ArbBudgetConstraint([leg_a, leg_b], min_edge_bps=2.0, maker_fee_rates=POLY_RATES | KALS_RATES, taker_fee_rates=POLY_TAKER_RATES | KALS_TAKER_RATES)
        c.update_order_price(POLY, p(0.42), is_crossing=False)
        cost = c.leg_cost(POLY)
        assert cost is not None
        assert cost == pytest.approx(0.42 + _fee(0.0, 0.42))

    def test_crossing_flag_cleared_on_new_noncrossing_order(self):
        leg_a = ArbLeg(listing=POLY, target_qty=q(5))
        leg_b = ArbLeg(listing=KALS, target_qty=q(5))
        c = ArbBudgetConstraint([leg_a, leg_b], min_edge_bps=2.0, maker_fee_rates=POLY_RATES | KALS_RATES, taker_fee_rates=POLY_TAKER_RATES | KALS_TAKER_RATES)
        c.update_order_price(POLY, p(0.43), is_crossing=True)
        c.update_order_price(POLY, p(0.42), is_crossing=False)
        cost = c.leg_cost(POLY)
        assert cost == pytest.approx(0.42 + _fee(0.0, 0.42))
