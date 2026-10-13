"""Drive-level Monte Carlo simulation of the rest of a game.

Each simulated game draws one posterior sample of the two teams' net offensive strengths, so the
spread of results reflects both game randomness and parameter uncertainty (posterior predictive).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from gnomepy_research.sessions.cfb_win_probability.live.drive_model import DriveModel
from gnomepy_research.sessions.cfb_win_probability.live.rules import (
    DEF_TD,
    END_HALF,
    FG,
    HALF_SECONDS,
    N_OUTCOMES,
    OT_START_YTG,
    QUARTER_SECONDS,
    SAFETY,
    SIT_OT,
    TD,
    OvertimeRules,
    overtime_rules,
    situation,
    ytg_bin,
)

HOME, AWAY = 0, 1
KICKOFF = 0       # ytg value meaning "possession starts with a kickoff to this team"
COIN_TOSS = -1    # possession value meaning "pregame: opening kickoff receiver is random"
MAX_OT = 30       # a degenerate model could otherwise loop forever; real games never get close


@dataclass
class GameState:
    """State at the start of a possession, from the home team's perspective."""
    season: int
    half: int = 1                      # 1, 2, or 3+ for overtime period (half - 2)
    half_clock: float = HALF_SECONDS   # seconds left in the half
    home_score: int = 0
    away_score: int = 0
    possession: int = COIN_TOSS        # HOME, AWAY, or COIN_TOSS before the opening kickoff
    ytg: int = KICKOFF                 # offense yards to goal, or KICKOFF
    second_half_receiver: int = -1     # who receives the 2nd-half kickoff; -1 = decided by coin toss


@dataclass
class SimResult:
    p_home: float
    mc_se: float
    n_sims: int
    home_margin: np.ndarray        # final margin; overtime results are reported as +/-1
    home_points: np.ndarray        # end-of-regulation points
    away_points: np.ndarray
    n_drives: np.ndarray           # drives simulated (excludes overtime)


@njit(cache=True)
def _sample_cdf(cdf: np.ndarray) -> int:
    return np.searchsorted(cdf, np.random.random(), side="right") + 1


@njit(cache=True)
def _sample_outcome(base_logits, beta, ybin, sit, s, fg_to_td):
    logits = np.empty(N_OUTCOMES)
    m = -1e300
    for k in range(N_OUTCOMES):
        logits[k] = base_logits[k, ybin, sit] + beta[k] * s
        if logits[k] > m:
            m = logits[k]
    total = 0.0
    for k in range(N_OUTCOMES):
        logits[k] = np.exp(logits[k] - m)
        total += logits[k]
    if fg_to_td > 0.0:
        moved = logits[FG] * fg_to_td
        logits[TD] += moved
        logits[FG] -= moved
    u = np.random.random() * total
    acc = 0.0
    for k in range(N_OUTCOMES):
        acc += logits[k]
        if u < acc:
            return k
    return N_OUTCOMES - 1


@njit(cache=True)
def _sample_duration(q):
    u = np.random.random() * (q.shape[0] - 1)
    i = int(u)
    if i >= q.shape[0] - 1:
        return q[q.shape[0] - 1]
    return q[i] + (q[i + 1] - q[i]) * (u - i)


@njit(cache=True)
def _try_points(lead_after_td, half, half_clock, p_xp, p_two, two_point_diffs, forced_two):
    go_for_two = forced_two
    if not go_for_two and half == 2 and half_clock <= QUARTER_SECONDS:
        for d in two_point_diffs:
            if d == lead_after_td:
                go_for_two = True
                break
    if go_for_two:
        return 2 if np.random.random() < p_two else 0
    return 1 if np.random.random() < p_xp else 0


@njit(cache=True)
def _overtime(s_off, first, base_logits, beta, p_xp, p_two, ot_fg_to_td, two_from, shootout_from):
    ybin = ytg_bin(OT_START_YTG)
    for ot in range(1, MAX_OT + 1):
        if ot >= shootout_from:
            a = np.random.random() < p_two
            b = np.random.random() < p_two
            if a != b:
                return HOME if (a and first == HOME) or (b and first == AWAY) else AWAY
            continue
        forced = ot >= two_from
        pts = np.zeros(2, dtype=np.int64)
        for turn in range(2):
            team = first if turn == 0 else 1 - first
            other = 1 - team
            # The second team knows what it needs: a FG can't save it when trailing by a TD.
            fg_to_td = ot_fg_to_td if turn == 1 and pts[other] > 3 else 0.0
            k = _sample_outcome(base_logits, beta, ybin, SIT_OT, s_off[team], fg_to_td)
            if k == DEF_TD:
                return other
            if k == TD:
                pts[team] += 6
                if turn == 1 and pts[team] > pts[other]:
                    return team
                if forced or (turn == 1 and pts[other] - pts[team] == 2):
                    pts[team] += 2 if np.random.random() < p_two else 0
                else:
                    pts[team] += 1 if np.random.random() < p_xp else 0
            elif k == FG:
                pts[team] += 3
            if turn == 1 and pts[team] > pts[other]:
                return team
        if pts[HOME] != pts[AWAY]:
            return HOME if pts[HOME] > pts[AWAY] else AWAY
        first = 1 - first
    return HOME if np.random.random() < 0.5 else AWAY


@njit(cache=True)
def _simulate_one(half, clock, home_score, away_score, poss, ytg, sh_recv, s_home, s_away,
                  base_logits, beta, next_cdf, kick_cdf, safety_kick_cdf, duration_q,
                  p_xp, p_two, two_point_diffs, ot_fg_to_td, two_from, shootout_from):
    score = np.array([home_score, away_score], dtype=np.int64)
    s_off = np.array([s_home, s_away])
    if poss == COIN_TOSS:
        poss = HOME if np.random.random() < 0.5 else AWAY
        ytg = KICKOFF
        sh_recv = 1 - poss
    elif sh_recv < 0:
        sh_recv = HOME if np.random.random() < 0.5 else AWAY

    n_drives = 0
    if half <= 2:
        while True:
            if clock <= 0.0:
                if half == 1:
                    half = 2
                    clock = float(HALF_SECONDS)
                    poss = sh_recv
                    ytg = KICKOFF
                    continue
                break
            if ytg == KICKOFF:
                ytg = _sample_cdf(kick_cdf)
            off = poss
            de = 1 - poss
            sit = situation(half, clock, score[off] - score[de])
            ybin = ytg_bin(ytg)
            k = _sample_outcome(base_logits, beta, ybin, sit, s_off[off], 0.0)
            n_drives += 1
            clock -= _sample_duration(duration_q[k, sit])
            if clock < 0.0:
                clock = 0.0
            if k == TD:
                score[off] += 6
                score[off] += _try_points(score[off] - score[de], half, clock, p_xp, p_two, two_point_diffs, False)
                poss = de
                ytg = KICKOFF
            elif k == FG:
                score[off] += 3
                poss = de
                ytg = KICKOFF
            elif k == DEF_TD:
                score[de] += 6
                score[de] += _try_points(score[de] - score[off], half, clock, p_xp, p_two, two_point_diffs, False)
                poss = off
                ytg = KICKOFF
            elif k == SAFETY:
                score[de] += 2
                poss = de
                ytg = _sample_cdf(safety_kick_cdf)
            elif k == END_HALF:
                clock = 0.0
            else:
                poss = de
                ytg = _sample_cdf(next_cdf[k, ybin])
        if score[HOME] != score[AWAY]:
            return score[HOME] - score[AWAY], score[HOME], score[AWAY], n_drives
        first = HOME if np.random.random() < 0.5 else AWAY
    else:
        # Mid-overtime states are rare and short; restart the OT from the current period's first team.
        first = poss

    winner = _overtime(s_off, first, base_logits, beta, p_xp, p_two, ot_fg_to_td, two_from, shootout_from)
    # Overtime margins are small; report a nominal 1-point win so margin sign carries the result.
    return (1 if winner == HOME else -1), score[HOME], score[AWAY], n_drives


@njit(cache=True)
def _simulate_many(n_sims, seed, half, clock, home_score, away_score, poss, ytg, sh_recv,
                   s_home, s_away, base_logits, beta, next_cdf, kick_cdf, safety_kick_cdf, duration_q,
                   p_xp, p_two, two_point_diffs, ot_fg_to_td, two_from, shootout_from):
    np.random.seed(seed)
    out = np.empty((4, n_sims), dtype=np.int64)
    n_samples = s_home.shape[0]
    for i in range(n_sims):
        j = i % n_samples
        out[0, i], out[1, i], out[2, i], out[3, i] = _simulate_one(half, clock, home_score, away_score, poss, ytg, sh_recv,
                                   s_home[j], s_away[j], base_logits, beta, next_cdf, kick_cdf,
                                   safety_kick_cdf, duration_q, p_xp, p_two, two_point_diffs,
                                   ot_fg_to_td, two_from, shootout_from)
    return out


def simulate(state: GameState, model: DriveModel, s_home: np.ndarray | float = 0.0,
             s_away: np.ndarray | float = 0.0, n_sims: int = 10_000, seed: int = 0,
             ot: OvertimeRules | None = None) -> SimResult:
    """Simulate the rest of the game. s_home / s_away are posterior samples of each team's net
    offensive strength (cycled if shorter than n_sims)."""
    ot = ot or overtime_rules(state.season)
    s_home = np.atleast_1d(np.asarray(s_home, dtype=np.float64))
    s_away = np.atleast_1d(np.asarray(s_away, dtype=np.float64))
    if s_home.shape != s_away.shape:
        raise ValueError("s_home and s_away must be paired posterior samples")
    out = _simulate_many(
        n_sims, seed, state.half, float(state.half_clock), state.home_score, state.away_score,
        state.possession, state.ytg, state.second_half_receiver, s_home, s_away,
        model.base_logits, model.beta, model.next_cdf, model.kick_cdf, model.safety_kick_cdf,
        model.duration_q, model.p_xp, model.p_two, model.two_point_diffs, model.ot_fg_to_td,
        ot.two_point_try_from, ot.shootout_from,
    )
    p = float((out[0] > 0).mean())
    return SimResult(p_home=p, mc_se=float(np.sqrt(p * (1 - p) / n_sims)), n_sims=n_sims,
                     home_margin=out[0], home_points=out[1], away_points=out[2], n_drives=out[3])
