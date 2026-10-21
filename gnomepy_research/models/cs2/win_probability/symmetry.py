"""
A/B orientation symmetry for team-vs-team features.

HLTV lists the better-ranked team first 60% of the time, so P(team_a wins)=0.5471
in the training data. That prior is real in the dataset and worthless at inference:
on a prediction market the YES side is arbitrary, so a model that has learned
"team_a is usually favoured" is wrong half the time.

Two mechanisms, doing different jobs. Augmenting with mirrored rows stops model
capacity being spent on the ordering artifact. Averaging both orientations at
predict time makes order-invariance exact rather than approximate — without it,
measured residual asymmetry is 0.0144.

The mirror is declared per feature and validated at import, because a spec that
drifts from the feature list as features are added would silently mis-mirror.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SymOp(str, Enum):
    IDENTITY = "identity"
    NEGATE = "negate"
    COMPLEMENT = "complement"
    SWAP = "swap"


FEATURE_SYMMETRY: dict[str, tuple[SymOp, str | None]] = {
    "team_a_rating_diff": (SymOp.NEGATE, None),
    "rank_diff": (SymOp.NEGATE, None),
    "recent_form_diff": (SymOp.NEGATE, None),
    "h2h_win_rate": (SymOp.COMPLEMENT, None),
    "team_a_picked_map": (SymOp.COMPLEMENT, None),
    "team_a_map_winrate_long": (SymOp.SWAP, "team_b_map_winrate_long"),
    "team_b_map_winrate_long": (SymOp.SWAP, "team_a_map_winrate_long"),
    "team_a_map_winrate_short": (SymOp.SWAP, "team_b_map_winrate_short"),
    "team_b_map_winrate_short": (SymOp.SWAP, "team_a_map_winrate_short"),
    "team_a_overall_winrate": (SymOp.SWAP, "team_b_overall_winrate"),
    "team_b_overall_winrate": (SymOp.SWAP, "team_a_overall_winrate"),
    "team_a_rank": (SymOp.SWAP, "team_b_rank"),
    "team_b_rank": (SymOp.SWAP, "team_a_rank"),
    "team_a_series_score": (SymOp.SWAP, "team_b_series_score"),
    "team_b_series_score": (SymOp.SWAP, "team_a_series_score"),
    "is_decider": (SymOp.IDENTITY, None),
    "is_lan": (SymOp.IDENTITY, None),
    "bo_type": (SymOp.IDENTITY, None),
    "map_position_in_series": (SymOp.IDENTITY, None),
    "event_tier": (SymOp.IDENTITY, None),
    "ranking_age_days": (SymOp.IDENTITY, None),
    "elo_diff": (SymOp.NEGATE, None),
    "elo_map_diff": (SymOp.NEGATE, None),
    "elo_map_resid_diff": (SymOp.NEGATE, None),
    "elo_side_asym": (SymOp.NEGATE, None),
    "elo_games_min": (SymOp.IDENTITY, None),

    # --- harvested blocks (player skill, roster, schedule, veto, event, page h2h) ---
    "team_a_roster_standins": (SymOp.SWAP, "team_b_roster_standins"),
    "team_b_roster_standins": (SymOp.SWAP, "team_a_roster_standins"),
    "team_a_roster_standin_coach": (SymOp.SWAP, "team_b_roster_standin_coach"),
    "team_b_roster_standin_coach": (SymOp.SWAP, "team_a_roster_standin_coach"),
    "team_a_roster_core_present": (SymOp.SWAP, "team_b_roster_core_present"),
    "team_b_roster_core_present": (SymOp.SWAP, "team_a_roster_core_present"),
    "team_a_roster_core_missing": (SymOp.SWAP, "team_b_roster_core_missing"),
    "team_b_roster_core_missing": (SymOp.SWAP, "team_a_roster_core_missing"),
    "team_a_roster_new_faces": (SymOp.SWAP, "team_b_roster_new_faces"),
    "team_b_roster_new_faces": (SymOp.SWAP, "team_a_roster_new_faces"),
    "team_a_roster_usage_mean": (SymOp.SWAP, "team_b_roster_usage_mean"),
    "team_b_roster_usage_mean": (SymOp.SWAP, "team_a_roster_usage_mean"),
    "team_a_roster_usage_min": (SymOp.SWAP, "team_b_roster_usage_min"),
    "team_b_roster_usage_min": (SymOp.SWAP, "team_a_roster_usage_min"),
    "team_a_roster_rotation_30": (SymOp.SWAP, "team_b_roster_rotation_30"),
    "team_b_roster_rotation_30": (SymOp.SWAP, "team_a_roster_rotation_30"),
    "team_a_roster_days_since_move": (SymOp.SWAP, "team_b_roster_days_since_move"),
    "team_b_roster_days_since_move": (SymOp.SWAP, "team_a_roster_days_since_move"),
    "team_a_roster_moves_30": (SymOp.SWAP, "team_b_roster_moves_30"),
    "team_b_roster_moves_30": (SymOp.SWAP, "team_a_roster_moves_30"),
    "team_a_roster_moves_90": (SymOp.SWAP, "team_b_roster_moves_90"),
    "team_b_roster_moves_90": (SymOp.SWAP, "team_a_roster_moves_90"),
    "team_a_roster_recent_joiners": (SymOp.SWAP, "team_b_roster_recent_joiners"),
    "team_b_roster_recent_joiners": (SymOp.SWAP, "team_a_roster_recent_joiners"),
    "team_a_roster_core_lost_30": (SymOp.SWAP, "team_b_roster_core_lost_30"),
    "team_b_roster_core_lost_30": (SymOp.SWAP, "team_a_roster_core_lost_30"),
    "team_a_roster_coach_change_90": (SymOp.SWAP, "team_b_roster_coach_change_90"),
    "team_b_roster_coach_change_90": (SymOp.SWAP, "team_a_roster_coach_change_90"),
    "roster_standins_diff": (SymOp.NEGATE, None),
    "roster_core_present_diff": (SymOp.NEGATE, None),
    "roster_core_missing_diff": (SymOp.NEGATE, None),
    "roster_new_faces_diff": (SymOp.NEGATE, None),
    "roster_usage_mean_diff": (SymOp.NEGATE, None),
    "roster_usage_min_diff": (SymOp.NEGATE, None),
    "roster_rotation_30_diff": (SymOp.NEGATE, None),
    "roster_days_since_move_diff": (SymOp.NEGATE, None),
    "roster_moves_30_diff": (SymOp.NEGATE, None),
    "roster_recent_joiners_diff": (SymOp.NEGATE, None),
    "roster_core_lost_30_diff": (SymOp.NEGATE, None),
    "ps_p_map": (SymOp.COMPLEMENT, None),
    "ps_p_series": (SymOp.COMPLEMENT, None),
    "ps_strength_diff": (SymOp.NEGATE, None),
    "ps_gap_sd": (SymOp.IDENTITY, None),
    "team_a_ps_strength": (SymOp.SWAP, "team_b_ps_strength"),
    "team_b_ps_strength": (SymOp.SWAP, "team_a_ps_strength"),
    "team_a_ps_chem": (SymOp.SWAP, "team_b_ps_chem"),
    "team_b_ps_chem": (SymOp.SWAP, "team_a_ps_chem"),
    "team_a_ps_sd": (SymOp.SWAP, "team_b_ps_sd"),
    "team_b_ps_sd": (SymOp.SWAP, "team_a_ps_sd"),
    "team_a_ps_new_players": (SymOp.SWAP, "team_b_ps_new_players"),
    "team_b_ps_new_players": (SymOp.SWAP, "team_a_ps_new_players"),
    "team_a_ps_best": (SymOp.SWAP, "team_b_ps_best"),
    "team_b_ps_best": (SymOp.SWAP, "team_a_ps_best"),
    "team_a_ps_worst": (SymOp.SWAP, "team_b_ps_worst"),
    "team_b_ps_worst": (SymOp.SWAP, "team_a_ps_worst"),
    "team_a_ps_five_maps": (SymOp.SWAP, "team_b_ps_five_maps"),
    "team_b_ps_five_maps": (SymOp.SWAP, "team_a_ps_five_maps"),
    "team_a_rest_days": (SymOp.SWAP, "team_b_rest_days"),
    "team_b_rest_days": (SymOp.SWAP, "team_a_rest_days"),
    "team_a_maps_14d": (SymOp.SWAP, "team_b_maps_14d"),
    "team_b_maps_14d": (SymOp.SWAP, "team_a_maps_14d"),
    "team_a_bans_before_pick": (SymOp.SWAP, "team_b_bans_before_pick"),
    "team_b_bans_before_pick": (SymOp.SWAP, "team_a_bans_before_pick"),
    "rest_days_diff": (SymOp.NEGATE, None),
    "maps_14d_diff": (SymOp.NEGATE, None),
    "own_h2h_rate": (SymOp.COMPLEMENT, None),
    "page_h2h_rate": (SymOp.COMPLEMENT, None),
    "page_h2h_recent_rate": (SymOp.COMPLEMENT, None),
    "team_a_picked_map_exact": (SymOp.COMPLEMENT, None),
    "team_a_banned_first": (SymOp.COMPLEMENT, None),
    "own_h2h_maps": (SymOp.IDENTITY, None),
    "own_h2h_never_met": (SymOp.IDENTITY, None),
    "veto_known": (SymOp.IDENTITY, None),
    "map_pick_index": (SymOp.IDENTITY, None),
    "page_h2h_maps": (SymOp.IDENTITY, None),
    "page_h2h_never_met": (SymOp.IDENTITY, None),
    "page_h2h_days_since": (SymOp.IDENTITY, None),
    "is_elimination": (SymOp.IDENTITY, None),
    "is_qualifier": (SymOp.IDENTITY, None),
    "is_playoff": (SymOp.IDENTITY, None),
    "is_group": (SymOp.IDENTITY, None),
}


@dataclass(frozen=True)
class SwapPlan:
    """Mirror as an affine index map: x' = offset + sign * x[perm]."""
    perm: np.ndarray
    sign: np.ndarray
    offset: np.ndarray
    names: tuple[str, ...]


def build_swap_plan(feature_names: list[str]) -> SwapPlan:
    """
    Compile the mirror for `feature_names`, raising on anything unspecified.

    Map one-hots are IDENTITY implicitly — the map does not change when the two
    teams are exchanged.
    """
    index = {n: i for i, n in enumerate(feature_names)}
    n = len(feature_names)
    perm = np.arange(n)
    sign = np.ones(n, dtype=np.float64)
    offset = np.zeros(n, dtype=np.float64)

    for i, name in enumerate(feature_names):
        if name.startswith("map_de_"):
            continue
        if name not in FEATURE_SYMMETRY:
            raise KeyError(f"feature {name!r} has no FEATURE_SYMMETRY entry")
        op, partner = FEATURE_SYMMETRY[name]
        if op is SymOp.IDENTITY:
            continue
        if op is SymOp.NEGATE:
            sign[i] = -1.0
        elif op is SymOp.COMPLEMENT:
            sign[i] = -1.0
            offset[i] = 1.0
        elif op is SymOp.SWAP:
            if partner is None or partner not in index:
                raise KeyError(f"feature {name!r} declares SWAP partner {partner!r}, which is absent")
            perm[i] = index[partner]

    if not np.array_equal(perm[perm], np.arange(n)):
        raise ValueError("SWAP partners are not mutual — the mirror is not an involution")
    return SwapPlan(perm=perm, sign=sign, offset=offset, names=tuple(feature_names))


def swap_features(X: np.ndarray, plan: SwapPlan) -> np.ndarray:
    """Mirror a feature matrix, exchanging the roles of team_a and team_b."""
    return plan.offset + plan.sign * X[:, plan.perm]


def augment(X: np.ndarray, y: np.ndarray, plan: SwapPlan) -> tuple[np.ndarray, np.ndarray]:
    """Stack the mirrored copy with flipped labels beneath the originals."""
    return np.vstack([X, swap_features(X, plan)]), np.concatenate([y, 1 - y])


def symmetric_predict_proba(model, X: np.ndarray, plan: SwapPlan) -> np.ndarray:
    """
    Average both orientations so the output cannot depend on which team is 'a'.

    Guarantees p(A,B) + p(B,A) == 1 to floating-point precision.
    """
    p = model.predict_proba(X)[:, 1]
    p_mirror = model.predict_proba(swap_features(X, plan))[:, 1]
    return 0.5 * (p + (1.0 - p_mirror))
