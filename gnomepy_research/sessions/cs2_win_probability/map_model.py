from __future__ import annotations

import math


def round_win_prob_to_map_win_prob(
    p_ct_wins_round: float,
    ct_score: int,
    t_score: int,
    rounds_to_win: int = 13,
) -> float:
    """
    Given P(CT wins next round), current scores, and the rounds needed to win
    the map, compute P(CT wins the map) via closed-form negative binomial formula.

    Standard CS2: first to 13 rounds wins, or whoever leads after 24 rounds
    (overtime handled by the caller via rounds_to_win adjustment).
    """
    p = float(p_ct_wins_round)
    p = max(1e-9, min(1 - 1e-9, p))
    q = 1.0 - p

    ct_need = rounds_to_win - ct_score
    t_need = rounds_to_win - t_score

    if ct_need <= 0:
        return 1.0
    if t_need <= 0:
        return 0.0

    total_rounds = ct_need + t_need - 1
    prob_ct_wins = 0.0
    for k in range(ct_need, total_rounds + 1):
        # CT wins on round k (0-indexed: k rounds played, CT wins last)
        # CT wins exactly (ct_need - 1) of the first (k-1) rounds, then wins round k
        n = k - 1
        r = ct_need - 1
        binom = _binom_coef(n, r)
        prob_ct_wins += binom * (p ** ct_need) * (q ** (k - ct_need))

    return float(min(max(prob_ct_wins, 0.0), 1.0))


def map_win_probs_to_series_win_prob(
    p_ct_wins_map: float,
    ct_maps_won: int,
    t_maps_won: int,
    maps_to_win: int = 2,
) -> float:
    """
    Given P(CT wins current/next map), map series score, compute P(CT wins series).
    Used for BO3 (maps_to_win=2) or BO5 (maps_to_win=3).
    """
    p = float(p_ct_wins_map)
    p = max(1e-9, min(1 - 1e-9, p))
    q = 1.0 - p

    ct_need = maps_to_win - ct_maps_won
    t_need = maps_to_win - t_maps_won

    if ct_need <= 0:
        return 1.0
    if t_need <= 0:
        return 0.0

    total_maps = ct_need + t_need - 1
    prob_ct_wins = 0.0
    for k in range(ct_need, total_maps + 1):
        n = k - 1
        r = ct_need - 1
        binom = _binom_coef(n, r)
        prob_ct_wins += binom * (p ** ct_need) * (q ** (k - ct_need))

    return float(min(max(prob_ct_wins, 0.0), 1.0))


def series_win_prob_per_map(
    current_map_prob: float,
    remaining_map_probs: list[float],
    team_a_maps_won: int,
    team_b_maps_won: int,
    maps_to_win: int = 2,
) -> float:
    """
    Compute P(team_a wins series) given heterogeneous per-map win probabilities.

    current_map_prob: P(team_a wins the current map)
    remaining_map_probs: P(team_a wins map_i) for each subsequent map in veto order
    team_a_maps_won / team_b_maps_won: maps already won before the current map
    """
    all_probs = [float(current_map_prob)] + [float(p) for p in remaining_map_probs]

    def _dp(a_wins: int, b_wins: int, map_idx: int) -> float:
        if a_wins >= maps_to_win:
            return 1.0
        if b_wins >= maps_to_win:
            return 0.0
        if map_idx >= len(all_probs):
            raise ValueError(
                f"veto list holds {len(all_probs)} map(s) but the series is undecided at "
                f"{a_wins}-{b_wins} needing {maps_to_win}; returning here would assign all "
                "remaining probability to team_b"
            )
        p = all_probs[map_idx]
        return p * _dp(a_wins + 1, b_wins, map_idx + 1) + (1 - p) * _dp(a_wins, b_wins + 1, map_idx + 1)

    return float(_dp(team_a_maps_won, team_b_maps_won, 0))


def _binom_coef(n: int, r: int) -> float:
    if r < 0 or r > n:
        return 0.0
    return math.exp(math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1))
