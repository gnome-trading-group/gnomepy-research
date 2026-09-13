from __future__ import annotations

import math

from gnomepy_research.arb.book_walking import OrderMode, walk_books

PRICE_SCALE = 1_000_000_000
SIZE_SCALE = 1_000_000


def p(frac: float) -> int:
    return int(frac * PRICE_SCALE)


def q(lots: int) -> int:
    return lots * SIZE_SCALE


def zero_cost(price: int, qty: int) -> float:
    return 0.0


class TestWalkBooksDegenerate:
    def test_no_books(self):
        qty, edge = walk_books([], [])
        assert qty == 0
        assert math.isinf(edge) and edge < 0

    def test_empty_levels(self):
        qty, edge = walk_books([[]], [zero_cost])
        assert qty == 0
        assert math.isinf(edge) and edge < 0


class TestWalkBooksBuy:
    def test_two_legs_profitable(self):
        books = [[(p(0.40), q(100))], [(p(0.40), q(100))]]
        cost_fns = [zero_cost, zero_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=True)
        assert total_qty == q(100)
        assert edge > 0

    def test_two_legs_not_profitable(self):
        books = [[(p(0.55), q(100))], [(p(0.55), q(100))]]
        cost_fns = [zero_cost, zero_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=True)
        assert total_qty == 0
        assert math.isinf(edge) and edge < 0

    def test_exact_breakeven_not_profitable(self):
        books = [[(p(0.50), q(100))], [(p(0.50), q(100))]]
        cost_fns = [zero_cost, zero_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=True)
        assert total_qty == 0

    def test_edge_calculation(self):
        # Two legs at 0.40 each, zero cost, 100 lots
        # Edge = (1.0 - 0.40 - 0.40) / 1.0 * 10000 = 2000 bps
        books = [[(p(0.40), q(100))], [(p(0.40), q(100))]]
        cost_fns = [zero_cost, zero_cost]
        _, edge = walk_books(books, cost_fns, buy=True)
        assert abs(edge - 2000.0) < 1.0


class TestWalkBooksSell:
    def test_two_legs_profitable(self):
        books = [[(p(0.55), q(100))], [(p(0.55), q(100))]]
        cost_fns = [zero_cost, zero_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=False)
        assert total_qty == q(100)
        assert edge > 0

    def test_two_legs_not_profitable(self):
        books = [[(p(0.40), q(100))], [(p(0.40), q(100))]]
        cost_fns = [zero_cost, zero_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=False)
        assert total_qty == 0
        assert math.isinf(edge) and edge < 0


class TestWalkBooksMultiLevel:
    def test_accumulates_across_levels(self):
        # Both levels profitable for leg 1; leg 2 has enough depth
        books = [
            [(p(0.38), q(50)), (p(0.42), q(50))],
            [(p(0.40), q(100))],
        ]
        cost_fns = [zero_cost, zero_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=True)
        assert total_qty == q(100)
        assert edge > 0

    def test_stops_at_unprofitable_level(self):
        # First 50 lots: 0.38 + 0.40 = 0.78 (profitable)
        # Next 50 lots: 0.62 + 0.40 = 1.02 (not profitable)
        books = [
            [(p(0.38), q(50)), (p(0.62), q(50))],
            [(p(0.40), q(100))],
        ]
        cost_fns = [zero_cost, zero_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=True)
        assert total_qty == q(50)
        assert edge > 0

    def test_partial_first_level_only(self):
        # 0.40 + 0.40 = 0.80 profitable; 0.70 + 0.40 = 1.10 not
        books = [
            [(p(0.40), q(100)), (p(0.70), q(100))],
            [(p(0.40), q(200))],
        ]
        cost_fns = [zero_cost, zero_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=True)
        assert total_qty == q(100)


class TestWalkBooksMaxQty:
    def test_max_qty_caps_result(self):
        books = [[(p(0.40), q(100))], [(p(0.40), q(100))]]
        cost_fns = [zero_cost, zero_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=True, max_qty=q(50))
        assert total_qty == q(50)
        assert edge > 0

    def test_max_qty_zero_means_unlimited(self):
        books = [[(p(0.40), q(100))], [(p(0.40), q(100))]]
        cost_fns = [zero_cost, zero_cost]
        total_qty, _ = walk_books(books, cost_fns, buy=True, max_qty=0)
        assert total_qty == q(100)


class TestWalkBooksCostFunctions:
    def test_positive_cost_reduces_edge(self):
        books = [[(p(0.40), q(100))], [(p(0.40), q(100))]]
        zero_fns = [zero_cost, zero_cost]
        fee_fns = [lambda pr, qt: 0.05, lambda pr, qt: 0.05]
        _, edge_zero = walk_books(books, zero_fns, buy=True)
        _, edge_fee = walk_books(books, fee_fns, buy=True)
        assert edge_zero > edge_fee

    def test_high_cost_makes_unprofitable(self):
        # 0.40 + 0.40 + 0.15 + 0.15 = 1.10 >= 1.0
        books = [[(p(0.40), q(100))], [(p(0.40), q(100))]]
        fee_fns = [lambda pr, qt: 0.15, lambda pr, qt: 0.15]
        total_qty, edge = walk_books(books, fee_fns, buy=True)
        assert total_qty == 0
        assert math.isinf(edge) and edge < 0

    def test_nonlinear_cost_binary_search(self):
        # Cost increases linearly with qty: starts at 0, reaches 0.25 at 100 lots
        # At qty=0: 0.40 + 0.40 + 0 + 0 = 0.80 profitable
        # At qty=100: 0.40 + 0.40 + 0.25 + 0.25 = 1.30 not profitable
        # Binary search should find a crossing point < 100 lots
        def growing_cost(pr: int, qt: int) -> float:
            return 0.25 * qt / q(100)

        books = [[(p(0.40), q(100))], [(p(0.40), q(100))]]
        cost_fns = [growing_cost, growing_cost]
        total_qty, edge = walk_books(books, cost_fns, buy=True)
        assert 0 < total_qty < q(100)
        assert edge > 0
