from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Protocol

from gnomepy import Scales

from gnomepy_research.arb.portfolio import ArbLeg
from gnomepy_research.arb.types import Listing

_PRICE_SCALE = Scales.PRICE
_SIZE_SCALE = Scales.SIZE
_ONE_CENT = _PRICE_SCALE // 100


def _fee(rate: float, price_normalized: float) -> float:
    return rate * price_normalized * (1.0 - price_normalized)


# ---------------------------------------------------------------------------
# Constraint model
# ---------------------------------------------------------------------------

class LegConstraint(Protocol):
    def max_price(self, listing: Listing) -> int | None: ...


class ArbBudgetConstraint:
    """Budget constraint for a multi-leg arb position.

    Tracks per-leg costs (locked fill prices or last-known bids) and solves
    for the maximum bid price on any leg that keeps combined cost within
    1.0 - min_edge_bps/10000.
    """

    def __init__(
        self,
        legs: list[ArbLeg],
        min_edge_bps: float,
        maker_fee_rates: dict[int, float],
        taker_fee_rates: dict[int, float] | None = None,
    ) -> None:
        self._listings: list[Listing] = [leg.listing for leg in legs]
        self._min_edge_bps = min_edge_bps
        self._maker_fee_rates = maker_fee_rates
        self._taker_fee_rates: dict[int, float] = taker_fee_rates or {}
        self._last_known_bid: dict[Listing, int] = {}
        self._last_order_price: dict[Listing, int] = {}
        self._is_crossing: dict[Listing, bool] = {}
        self._fill_cost: dict[Listing, float] = {}
        self._fill_qty: dict[Listing, int] = {}

    def update_bid(self, listing: Listing, bid_price: int) -> None:
        self._last_known_bid[listing] = bid_price

    def update_order_price(self, listing: Listing, price: int, is_crossing: bool = False) -> None:
        self._last_order_price[listing] = price
        self._is_crossing[listing] = is_crossing

    def last_bid(self, listing: Listing) -> int | None:
        return self._last_known_bid.get(listing)

    def record_fill(self, listing: Listing, fill_price: int, fill_qty: int) -> None:
        self._fill_cost[listing] = self._fill_cost.get(listing, 0.0) + (fill_price / _PRICE_SCALE) * (fill_qty / _SIZE_SCALE)
        self._fill_qty[listing] = self._fill_qty.get(listing, 0) + fill_qty

    def _avg_fill_price(self, listing: Listing) -> float | None:
        qty = self._fill_qty.get(listing, 0)
        if qty <= 0:
            return None
        return self._fill_cost[listing] / (qty / _SIZE_SCALE)

    def leg_cost(self, listing: Listing) -> float | None:
        maker_rate = self._maker_fee_rates.get(listing[0], 0.0)
        avg = self._avg_fill_price(listing)
        if avg is not None:
            return avg + _fee(maker_rate, avg)
        # For unfilled legs, prefer our submitted order price over market best bid.
        # This keeps the economic validity check anchored to what WE would fill at,
        # not what others are bidding — preventing premature cancel when market bids
        # transiently spike above our budget ceiling.
        order_price = self._last_order_price.get(listing)
        if order_price is not None and order_price > 0:
            p = order_price / _PRICE_SCALE
            rate = self._taker_fee_rates.get(listing[0], maker_rate) if self._is_crossing.get(listing, False) else maker_rate
            return p + _fee(rate, p)
        bid = self._last_known_bid.get(listing)
        if bid is None:
            return None
        p = bid / _PRICE_SCALE
        return p + _fee(maker_rate, p)

    def combined_cost(self) -> float | None:
        total = 0.0
        for listing in self._listings:
            cost = self.leg_cost(listing)
            if cost is None:
                return None
            total += cost
        return total

    def edge_bps(self) -> float | None:
        cost = self.combined_cost()
        if cost is None:
            return None
        return (1.0 - cost) * 10000.0

    def is_economically_valid(self) -> bool:
        edge = self.edge_bps()
        if edge is None:
            return True
        return edge >= self._min_edge_bps

    def max_price(self, listing: Listing) -> int | None:
        """Maximum bid price for listing that keeps combined cost within budget.

        Returns None if a sibling leg's cost is unknown. Returns 0 if budget
        is already exhausted by other legs.
        """
        budget = 1.0 - self._min_edge_bps / 10000.0

        for lst in self._listings:
            if lst == listing:
                continue
            cost = self.leg_cost(lst)
            if cost is None:
                return None
            budget -= cost

        if budget <= 0.0:
            return 0

        maker_rate = self._maker_fee_rates.get(listing[0], 0.0)
        rate = self._taker_fee_rates.get(listing[0], maker_rate) if self._is_crossing.get(listing, False) else maker_rate
        if rate == 0.0:
            max_p = budget
        else:
            discriminant = (1.0 + rate) ** 2 - 4.0 * rate * budget
            if discriminant < 0.0:
                max_p = budget
            else:
                max_p = ((1.0 + rate) - sqrt(discriminant)) / (2.0 * rate)

        max_p = max(0.0, min(1.0, max_p))
        return int(max_p * _PRICE_SCALE)

    def budget_remaining_bps(self, listing: Listing) -> float | None:
        """Available budget for listing expressed in bps above current best bid."""
        mp = self.max_price(listing)
        if mp is None:
            return None
        bid = self._last_known_bid.get(listing)
        if bid is None:
            return None
        return (mp - bid) / _PRICE_SCALE * 10000.0


# ---------------------------------------------------------------------------
# Price model
# ---------------------------------------------------------------------------

@dataclass
class PriceResult:
    price: int
    max_price: int | None
    fill_prob: float
    spread: float


@dataclass
class ArbContext:
    stuck_cost: float


class PriceModel(Protocol):
    def on_book_update(
        self,
        listing: Listing,
        bids: list[tuple[int, int]],
        asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None: ...

    def compute_price(
        self,
        listing: Listing,
        best_bid: int,
        best_ask: int,
        max_price: int | None,
        context: ArbContext | None = None,
    ) -> PriceResult: ...


def _fill_prob(best_ask: int, price: int, fill_risk_lambda: float) -> float:
    effective_spread = max(0.0, (best_ask - price) / _PRICE_SCALE)
    return exp(-fill_risk_lambda * effective_spread)


def _clamp_price(price: int, best_ask: int, max_price: int | None) -> int:
    ceiling = best_ask
    if max_price is not None:
        ceiling = min(ceiling, max_price)
    return max(0, min(price, ceiling))


class JoinBestBidModel:
    """Baseline: post at best bid, capped by the constraint ceiling."""

    def __init__(self, fill_risk_lambda: float = 10.0) -> None:
        self._lambda = fill_risk_lambda
        self._spreads: dict[Listing, float] = {}

    def on_book_update(self, listing, bids, asks, timestamp) -> None:
        if bids and asks and bids[0][0] > 0 and asks[0][0] > 0:
            self._spreads[listing] = (asks[0][0] - bids[0][0]) / _PRICE_SCALE

    def compute_price(self, listing, best_bid, best_ask, max_price, context: ArbContext | None = None) -> PriceResult:
        if best_bid <= 0 or best_ask <= 0:
            return PriceResult(price=0, max_price=max_price, fill_prob=0.0, spread=0.0)
        price = _clamp_price(best_bid, best_ask, max_price)
        spread = self._spreads.get(listing, 0.0)
        return PriceResult(
            price=price,
            max_price=max_price,
            fill_prob=_fill_prob(best_ask, price, self._lambda),
            spread=spread,
        )


class AggressivePriceModel:
    """Improve on best bid using available edge headroom.

    Improvement = min(base_improve, edge_share * available_edge) where
    available_edge = max_price - best_bid. Capped by best_ask - 1 tick
    and the constraint ceiling.
    """

    def __init__(
        self,
        base_improve_bps: float = 50.0,
        edge_share: float = 0.5,
        fill_risk_lambda: float = 10.0,
    ) -> None:
        self._base_improve = base_improve_bps / 10000.0
        self._edge_share = edge_share
        self._lambda = fill_risk_lambda
        self._mids: dict[Listing, float] = {}
        self._spreads: dict[Listing, float] = {}

    def on_book_update(self, listing, bids, asks, timestamp) -> None:
        if bids and asks and bids[0][0] > 0 and asks[0][0] > 0:
            bid = bids[0][0] / _PRICE_SCALE
            ask = asks[0][0] / _PRICE_SCALE
            self._mids[listing] = (bid + ask) / 2.0
            self._spreads[listing] = ask - bid

    def compute_price(self, listing, best_bid, best_ask, max_price, context: ArbContext | None = None) -> PriceResult:
        if best_bid <= 0 or best_ask <= 0:
            return PriceResult(price=0, max_price=max_price, fill_prob=0.0, spread=0.0)

        mid = self._mids.get(listing, best_bid / _PRICE_SCALE)
        base_improve = int(self._base_improve * mid * _PRICE_SCALE)

        if max_price is not None:
            available_edge = max(0, max_price - best_bid)
            edge_improve = int(self._edge_share * available_edge)
            improvement = min(base_improve, edge_improve)
        else:
            improvement = base_improve

        raw_price = best_bid + improvement
        price = _clamp_price(raw_price, best_ask, max_price)
        spread = self._spreads.get(listing, 0.0)
        return PriceResult(
            price=price,
            max_price=max_price,
            fill_prob=_fill_prob(best_ask, price, self._lambda),
            spread=spread,
        )


class FillProbTargetModel:
    """Work backward from a desired fill probability to determine bid price.

    Given P(fill) = exp(-lambda * spread), solve for the spread at target P,
    then place at best_ask - spread. Clamped to [best_bid, best_ask - 1 tick]
    and the constraint ceiling.
    """

    def __init__(
        self,
        target_fill_prob: float = 0.7,
        fill_risk_lambda: float = 10.0,
    ) -> None:
        if not (0.0 < target_fill_prob < 1.0):
            raise ValueError("target_fill_prob must be in (0, 1)")
        self._target_spread = -log(target_fill_prob) / fill_risk_lambda
        self._lambda = fill_risk_lambda
        self._spreads: dict[Listing, float] = {}

    def on_book_update(self, listing, bids, asks, timestamp) -> None:
        if bids and asks and bids[0][0] > 0 and asks[0][0] > 0:
            self._spreads[listing] = (asks[0][0] - bids[0][0]) / _PRICE_SCALE

    def compute_price(self, listing, best_bid, best_ask, max_price, context: ArbContext | None = None) -> PriceResult:
        if best_bid <= 0 or best_ask <= 0:
            return PriceResult(price=0, max_price=max_price, fill_prob=0.0, spread=0.0)

        target_price_raw = best_ask - int(self._target_spread * _PRICE_SCALE)
        price = _clamp_price(target_price_raw, best_ask, max_price)
        spread = self._spreads.get(listing, 0.0)
        return PriceResult(
            price=price,
            max_price=max_price,
            fill_prob=_fill_prob(best_ask, price, self._lambda),
            spread=spread,
        )


class OptimalEVModel:
    """Optimal expected-value maker pricing.

    Maximizes E[value(p)] = P(fill|p) * (ceiling - p - stuck_cost_adj) where
    P(fill|p) = exp(-lambda * (ask - p)). Closed-form solution:
        p* = ceiling - 1/lambda - stuck_cost_adj

    stuck_cost_adj is passed via ArbContext at each compute_price call.
    If p* < best_bid, bids at best_bid when EV is still positive there,
    or returns price=0 (infeasible) when the constraint ceiling leaves no room.
    """

    def __init__(self, fill_risk_lambda: float = 10.0) -> None:
        self._lambda = fill_risk_lambda
        self._spreads: dict[Listing, float] = {}

    def on_book_update(
        self,
        listing: Listing,
        bids: list[tuple[int, int]],
        asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None:
        if bids and asks and bids[0][0] > 0 and asks[0][0] > 0:
            self._spreads[listing] = (asks[0][0] - bids[0][0]) / _PRICE_SCALE

    def compute_price(
        self,
        listing: Listing,
        best_bid: int,
        best_ask: int,
        max_price: int | None,
        context: ArbContext | None = None,
    ) -> PriceResult:
        if best_bid <= 0 or best_ask <= 0:
            return PriceResult(price=0, max_price=max_price, fill_prob=0.0, spread=0.0)

        spread = self._spreads.get(listing, 0.0)
        stuck_cost = context.stuck_cost if context is not None else 0.0

        placement_ceiling = best_ask / _PRICE_SCALE
        value_ceiling = max_price / _PRICE_SCALE if max_price is not None else placement_ceiling

        best_bid_norm = best_bid / _PRICE_SCALE
        net_value_at_bid = value_ceiling - best_bid_norm - stuck_cost

        if max_price is not None and net_value_at_bid <= 0.0:
            return PriceResult(price=0, max_price=max_price, fill_prob=0.0, spread=spread)

        optimal_offset = 1.0 / self._lambda + stuck_cost
        optimal_price_norm = value_ceiling - optimal_offset
        price_norm = max(best_bid_norm, min(optimal_price_norm, placement_ceiling))
        raw_price = int(price_norm * _PRICE_SCALE)
        price = _clamp_price(raw_price, best_ask, max_price)

        return PriceResult(
            price=price,
            max_price=max_price,
            fill_prob=_fill_prob(best_ask, price, self._lambda),
            spread=spread,
        )
