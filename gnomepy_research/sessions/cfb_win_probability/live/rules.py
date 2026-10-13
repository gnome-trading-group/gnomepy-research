"""NCAA football game structure shared by the drive model (fitting) and the simulator.

Clock is tracked as seconds remaining in the current half (0-1800); quarters only matter for the
2-minute and halftime boundaries, and drives that cross the 1st/3rd quarter break simply continue.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

OUTCOMES = ("TD", "FG", "PUNT", "TURNOVER", "DOWNS", "MISSED_FG", "DEF_TD", "SAFETY", "END_HALF")
TD, FG, PUNT, TURNOVER, DOWNS, MISSED_FG, DEF_TD, SAFETY, END_HALF = range(len(OUTCOMES))
N_OUTCOMES = len(OUTCOMES)
OUTCOME_INDEX = {name: i for i, name in enumerate(OUTCOMES)}

HALF_SECONDS = 1800
QUARTER_SECONDS = 900
OT_START_YTG = 25

# Bin upper bounds (inclusive). Finer near the goal line, where TD/FG odds move fastest.
YTG_UPPER = np.array([5, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90], dtype=np.int64)
N_YTG = len(YTG_UPPER) + 1
# Seconds left in the half: end-of-half behaviour (kneels, hurry-up, END_HALF) lives in the last 5 min.
TIME_UPPER = np.array([30, 60, 120, 300, 900], dtype=np.int64)
N_TIME = len(TIME_UPPER) + 1
# Offense score minus defense score, in possession-count steps.
DIFF_UPPER = np.array([-17, -9, -4, -1, 0, 3, 8, 16], dtype=np.int64)
N_DIFF = len(DIFF_UPPER) + 1

N_REG_SIT = 2 * N_TIME * N_DIFF
SIT_OT = N_REG_SIT
N_SIT = N_REG_SIT + 1


@njit(cache=True)
def ytg_bin(ytg: int) -> int:
    return np.searchsorted(YTG_UPPER, ytg, side="left")


@njit(cache=True)
def time_bin(half_clock: float) -> int:
    return np.searchsorted(TIME_UPPER, half_clock, side="left")


@njit(cache=True)
def diff_bin(diff: int) -> int:
    return np.searchsorted(DIFF_UPPER, diff, side="left")


@njit(cache=True)
def situation(half: int, half_clock: float, diff: int) -> int:
    if half > 2:
        return SIT_OT
    return (half - 1) * N_TIME * N_DIFF + time_bin(half_clock) * N_DIFF + diff_bin(diff)


def half_clock(period: int, clock_s: float) -> float:
    """Seconds left in the half from a (period, seconds-left-in-period) pair."""
    return clock_s + (QUARTER_SECONDS if period in (1, 3) else 0)


@dataclass(frozen=True)
class OvertimeRules:
    # First OT period in which a team that scores a TD must go for two.
    two_point_try_from: int
    # First OT period played as alternating 2-point attempts instead of possessions from the 25.
    shootout_from: int


def overtime_rules(season: int) -> OvertimeRules:
    """2019 added the 2-point shootout from the 5th OT; 2021 moved the forced 2-point try to the
    2nd OT and the shootout to the 3rd."""
    if season >= 2021:
        return OvertimeRules(two_point_try_from=2, shootout_from=3)
    if season >= 2019:
        return OvertimeRules(two_point_try_from=3, shootout_from=5)
    return OvertimeRules(two_point_try_from=3, shootout_from=10**6)
