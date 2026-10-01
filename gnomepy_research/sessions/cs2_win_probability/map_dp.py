"""
Layer 2: P(team_a wins the map) from any score state.

Replaces the negative-binomial roll-up, which assumed a single constant
per-round probability for the rest of the map, ignored the round-12 side swap
entirely, and treated 12-12 as one coin flip rather than an MR3 overtime.

Measured at halftime against held-out maps, this DP reaches AUC 0.8729 versus
0.8698 for an XGBoost model trained directly on the halftime state — and unlike
that model it is defined at every score, not just the two snapshots per map that
the data happens to contain.

Team strength enters as a side-neutral log-odds theta; the map's CT bias enters
as delta. Both are supplied by the round-rate model, then theta is shifted so the
DP evaluated from 0-0 reproduces the Layer-1 pre-map probability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from gnomepy_research.pipelines.hltv_cs2.sides import side_swaps_before_round

_EPS = 1e-12


@dataclass(frozen=True)
class MapFormat:
    rounds_to_win: int = 13
    half_length: int = 12
    overtime: bool = True
    ot_half_length: int = 3
    ot_rounds_to_win: int = 4
    draw_value: float = 0.5


DEFAULT_FORMAT = MapFormat()


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def side_probs(theta: float, delta: float) -> tuple[float, float]:
    """(p_CT, p_T) for team_a: strength shifted by the map's CT bias in each direction."""
    return sigmoid(theta + delta), sigmoid(theta - delta)


def _a_is_ct(round_number: int, team_a_started_ct: bool, ot_offset: int) -> bool:
    return team_a_started_ct != (side_swaps_before_round(round_number, ot_offset) % 2 == 1)


def _period_value(
    p_ct: float, p_t: float, team_a_started_ct: bool, ot_offset: int,
    continuation: float, fmt: MapFormat, start_x: int = 0, start_y: int = 0,
) -> float:
    """Value of a fresh overtime period given the value of reaching a further period."""
    need = fmt.ot_rounds_to_win
    half = fmt.ot_half_length

    @lru_cache(maxsize=None)
    def f(x: int, y: int) -> float:
        if x >= need:
            return 1.0
        if y >= need:
            return 0.0
        if x == need - 1 and y == need - 1:
            return continuation
        r = 2 * fmt.half_length + (x + y) + 1
        p = p_ct if _a_is_ct(r, team_a_started_ct, ot_offset) else p_t
        return p * f(x + 1, y) + (1.0 - p) * f(x, y + 1)

    value = f(start_x, start_y)
    f.cache_clear()
    return value


def overtime_value(
    p_ct: float, p_t: float, team_a_started_ct: bool,
    fmt: MapFormat = DEFAULT_FORMAT, ot_offset: int = 0,
) -> float:
    """
    Value of entering overtime, solved exactly rather than truncated.

    Side parity repeats every overtime period, so a period's value is affine in
    the value of reaching the next one: g(c) = A + B*c. Solving the fixed point
    g(W) = W gives W = A / (1 - B), where A is the chance of closing the period
    out and B the chance of reaching 3-3 and going again.
    """
    a = _period_value(p_ct, p_t, team_a_started_ct, ot_offset, 0.0, fmt)
    b = _period_value(p_ct, p_t, team_a_started_ct, ot_offset, 1.0, fmt) - a
    if b >= 1.0 - _EPS:
        return 0.5
    return a / (1.0 - b)


def _overtime_period_state(team_a_score: int, team_b_score: int, fmt: MapFormat) -> tuple[int, int]:
    """
    Map an absolute overtime scoreline onto (x, y) within the current period.

    Every drawn period adds ot_half_length to both scores, so completed periods
    subtract off symmetrically and the remainder is the live period.
    """
    drawn_periods = (min(team_a_score, team_b_score) - fmt.half_length) // fmt.ot_half_length
    base = fmt.half_length + fmt.ot_half_length * drawn_periods
    return team_a_score - base, team_b_score - base


def map_win_prob(
    p_ct: float, p_t: float,
    team_a_score: int, team_b_score: int,
    team_a_started_ct: bool,
    fmt: MapFormat = DEFAULT_FORMAT,
    ot_offset: int = 0,
) -> float:
    """P(team_a wins the map) from the given score, with sides swapping on schedule."""
    win = fmt.rounds_to_win
    in_overtime = fmt.overtime and team_a_score >= win - 1 and team_b_score >= win - 1

    if in_overtime:
        x, y = _overtime_period_state(team_a_score, team_b_score, fmt)
        tail = overtime_value(p_ct, p_t, team_a_started_ct, fmt, ot_offset)
        return _period_value(p_ct, p_t, team_a_started_ct, ot_offset, tail, fmt, x, y)

    if team_a_score >= win:
        return 1.0
    if team_b_score >= win:
        return 0.0
    if team_a_score == win - 1 and team_b_score == win - 1:
        return fmt.draw_value

    tail = overtime_value(p_ct, p_t, team_a_started_ct, fmt, ot_offset) if fmt.overtime \
        else fmt.draw_value

    @lru_cache(maxsize=None)
    def v(a: int, b: int) -> float:
        if a >= win:
            return 1.0
        if b >= win:
            return 0.0
        if a == win - 1 and b == win - 1:
            return tail
        r = a + b + 1
        p = p_ct if _a_is_ct(r, team_a_started_ct, ot_offset) else p_t
        return p * v(a + 1, b) + (1.0 - p) * v(a, b + 1)

    value = v(team_a_score, team_b_score)
    v.cache_clear()
    return value


def solve_theta(
    target_prob: float,
    delta: float,
    team_a_started_ct: bool | None,
    team_a_score: int = 0,
    team_b_score: int = 0,
    fmt: MapFormat = DEFAULT_FORMAT,
    ot_offset: int = 0,
    bracket: tuple[float, float] = (-8.0, 8.0),
    tol: float = 1e-9,
) -> float:
    """
    Find the team-strength log-odds whose DP value equals `target_prob`.

    This is the join between layers: Layer 1 sets the level, the round-rate model
    sets the CT/T split via `delta`, and inverting recovers the theta consistent
    with both. The DP is strictly increasing in theta, so bisection is safe.

    `team_a_started_ct=None` means sides are not yet known — the knife round has
    not happened — so the target is matched against the side-averaged value.
    """
    def value(theta: float) -> float:
        p_ct, p_t = side_probs(theta, delta)
        if team_a_started_ct is None:
            return 0.5 * (
                map_win_prob(p_ct, p_t, team_a_score, team_b_score, True, fmt, ot_offset)
                + map_win_prob(p_ct, p_t, team_a_score, team_b_score, False, fmt, ot_offset)
            )
        return map_win_prob(p_ct, p_t, team_a_score, team_b_score, team_a_started_ct, fmt, ot_offset)

    lo, hi = bracket
    if value(lo) > target_prob:
        return lo
    if value(hi) < target_prob:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if value(mid) < target_prob:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def map_win_prob_from_pre_map(
    pre_map_prob: float,
    delta: float,
    team_a_score: int,
    team_b_score: int,
    team_a_started_ct: bool | None,
    fmt: MapFormat = DEFAULT_FORMAT,
    ot_offset: int = 0,
) -> float:
    """
    Price a live score from a pre-map probability.

    theta is pinned so the DP reproduces `pre_map_prob` at 0-0, then the same
    theta is evaluated at the current score. When the starting side becomes known
    mid-map, theta stays fixed and only the side assignment changes — which is
    what produces the small, correct price move at knife-round resolution.
    """
    theta = solve_theta(pre_map_prob, delta, team_a_started_ct, 0, 0, fmt, ot_offset)
    p_ct, p_t = side_probs(theta, delta)
    if team_a_started_ct is None:
        return 0.5 * (
            map_win_prob(p_ct, p_t, team_a_score, team_b_score, True, fmt, ot_offset)
            + map_win_prob(p_ct, p_t, team_a_score, team_b_score, False, fmt, ot_offset)
        )
    return map_win_prob(p_ct, p_t, team_a_score, team_b_score, team_a_started_ct, fmt, ot_offset)


class MapValueTable:
    """
    Precomputed V(a, b) for one map's strength and side assignment.

    theta is fixed per map, so the whole value surface can be built once and then
    read per snapshot. The intra-round overlay needs V at two adjacent states for
    every snapshot, which would otherwise mean re-solving the DP tens of
    thousands of times for values that never change within a map.
    """

    def __init__(
        self,
        p_ct: float, p_t: float, team_a_started_ct: bool,
        fmt: MapFormat = DEFAULT_FORMAT, ot_offset: int = 0,
    ):
        self.p_ct = p_ct
        self.p_t = p_t
        self.started_ct = team_a_started_ct
        self.fmt = fmt
        self.ot_offset = ot_offset
        win = fmt.rounds_to_win
        self._tail = overtime_value(p_ct, p_t, team_a_started_ct, fmt, ot_offset) if fmt.overtime \
            else fmt.draw_value

        grid = np.zeros((win + 1, win + 1), dtype=float)
        for total in range(2 * win, -1, -1):
            for a in range(min(total, win), max(-1, total - win), -1):
                b = total - a
                if b > win or b < 0:
                    continue
                if a >= win:
                    grid[a, b] = 1.0
                elif b >= win:
                    grid[a, b] = 0.0
                elif a == win - 1 and b == win - 1:
                    grid[a, b] = self._tail
                else:
                    p = p_ct if _a_is_ct(a + b + 1, team_a_started_ct, ot_offset) else p_t
                    grid[a, b] = p * grid[a + 1, b] + (1.0 - p) * grid[a, b + 1]
        self._grid = grid

    @classmethod
    def from_pre_map(
        cls,
        pre_map_prob: float, delta: float, team_a_started_ct: bool,
        fmt: MapFormat = DEFAULT_FORMAT, ot_offset: int = 0,
    ) -> "MapValueTable":
        theta = solve_theta(pre_map_prob, delta, team_a_started_ct, 0, 0, fmt, ot_offset)
        p_ct, p_t = side_probs(theta, delta)
        return cls(p_ct, p_t, team_a_started_ct, fmt, ot_offset)

    def value(self, team_a_score: int, team_b_score: int) -> float:
        win = self.fmt.rounds_to_win
        if self.fmt.overtime and team_a_score >= win - 1 and team_b_score >= win - 1:
            return map_win_prob(
                self.p_ct, self.p_t, team_a_score, team_b_score,
                self.started_ct, self.fmt, self.ot_offset,
            )
        return float(self._grid[min(team_a_score, win), min(team_b_score, win)])

    def round_prob(self, round_number: int) -> float:
        return self.p_ct if _a_is_ct(round_number, self.started_ct, self.ot_offset) else self.p_t

    def overlay(self, team_a_score: int, team_b_score: int, p_round: float) -> float:
        """Blend one modelled round outcome into the map value; the DP handles the rest."""
        return p_round * self.value(team_a_score + 1, team_b_score) \
            + (1.0 - p_round) * self.value(team_a_score, team_b_score + 1)
