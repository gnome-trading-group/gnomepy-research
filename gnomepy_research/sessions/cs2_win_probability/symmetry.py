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

    # --- harvested blocks (player form, schedule, rank, veto, event, page h2h) ---
    "team_a_form_rating": (SymOp.SWAP, "team_b_form_rating"),
    "team_b_form_rating": (SymOp.SWAP, "team_a_form_rating"),
    "team_a_form_worst": (SymOp.SWAP, "team_b_form_worst"),
    "team_b_form_worst": (SymOp.SWAP, "team_a_form_worst"),
    "team_a_form_spread": (SymOp.SWAP, "team_b_form_spread"),
    "team_b_form_spread": (SymOp.SWAP, "team_a_form_spread"),
    "team_a_form_swing": (SymOp.SWAP, "team_b_form_swing"),
    "team_b_form_swing": (SymOp.SWAP, "team_a_form_swing"),
    "team_a_form_adr_resid": (SymOp.SWAP, "team_b_form_adr_resid"),
    "team_b_form_adr_resid": (SymOp.SWAP, "team_a_form_adr_resid"),
    "team_a_form_kast_resid": (SymOp.SWAP, "team_b_form_kast_resid"),
    "team_b_form_kast_resid": (SymOp.SWAP, "team_a_form_kast_resid"),
    "team_a_form_ct": (SymOp.SWAP, "team_b_form_ct"),
    "team_b_form_ct": (SymOp.SWAP, "team_a_form_ct"),
    "team_a_form_t": (SymOp.SWAP, "team_b_form_t"),
    "team_b_form_t": (SymOp.SWAP, "team_a_form_t"),
    "team_a_form_maps_min": (SymOp.SWAP, "team_b_form_maps_min"),
    "team_b_form_maps_min": (SymOp.SWAP, "team_a_form_maps_min"),
    "team_a_form_known": (SymOp.SWAP, "team_b_form_known"),
    "team_b_form_known": (SymOp.SWAP, "team_a_form_known"),
    "team_a_rest_days": (SymOp.SWAP, "team_b_rest_days"),
    "team_b_rest_days": (SymOp.SWAP, "team_a_rest_days"),
    "team_a_maps_14d": (SymOp.SWAP, "team_b_maps_14d"),
    "team_b_maps_14d": (SymOp.SWAP, "team_a_maps_14d"),
    "team_a_rank_best": (SymOp.SWAP, "team_b_rank_best"),
    "team_b_rank_best": (SymOp.SWAP, "team_a_rank_best"),
    "team_a_log_rank": (SymOp.SWAP, "team_b_log_rank"),
    "team_b_log_rank": (SymOp.SWAP, "team_a_log_rank"),
    "team_a_bans_before_pick": (SymOp.SWAP, "team_b_bans_before_pick"),
    "team_b_bans_before_pick": (SymOp.SWAP, "team_a_bans_before_pick"),
    "form_rating_diff": (SymOp.NEGATE, None),
    "form_worst_diff": (SymOp.NEGATE, None),
    "form_swing_diff": (SymOp.NEGATE, None),
    "form_adr_resid_diff": (SymOp.NEGATE, None),
    "form_kast_resid_diff": (SymOp.NEGATE, None),
    "form_ct_diff": (SymOp.NEGATE, None),
    "form_t_diff": (SymOp.NEGATE, None),
    "rest_days_diff": (SymOp.NEGATE, None),
    "maps_14d_diff": (SymOp.NEGATE, None),
    "rank_best_diff": (SymOp.NEGATE, None),
    "log_rank_diff": (SymOp.NEGATE, None),
    "rank_source_disagreement": (SymOp.NEGATE, None),
    "own_h2h_rate": (SymOp.COMPLEMENT, None),
    "page_h2h_rate": (SymOp.COMPLEMENT, None),
    "page_h2h_recent_rate": (SymOp.COMPLEMENT, None),
    "team_a_picked_map_exact": (SymOp.COMPLEMENT, None),
    "team_a_banned_first": (SymOp.COMPLEMENT, None),
    "form_maps_min": (SymOp.IDENTITY, None),
    "form_known_min": (SymOp.IDENTITY, None),
    "own_h2h_maps": (SymOp.IDENTITY, None),
    "own_h2h_never_met": (SymOp.IDENTITY, None),
    "rank_known_both": (SymOp.IDENTITY, None),
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
