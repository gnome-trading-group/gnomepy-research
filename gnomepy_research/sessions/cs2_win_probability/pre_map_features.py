"""
Pre-map feature extraction.

One extractor serves both training and live inference. The two previously
diverged: the training variant read team_a_recent_form / team_b_recent_form,
which cs2_match_priors never emitted, so recent-form was NaN for every training
row while live inference received real values. A single code path plus an
explicit completeness check makes that class of skew structurally impossible.

Missing values are NaN by design — XGBoost handles them natively — but a key
being *absent* is a bug, not a missing value, so it raises.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gnomepy_research.pipelines.hltv_cs2.config import MAP_POOL
from gnomepy_research.pipelines.hltv_cs2.context_features import (
    EVENT_FEATURES,
    PAGE_H2H_FEATURES,
    RANK_FEATURES,
    SCHEDULE_FEATURES,
    VETO_FEATURES,
)
from gnomepy_research.pipelines.hltv_cs2.player_form import PLAYER_FORM_FEATURES

HARVEST_FEATURE_NAMES = (
    PLAYER_FORM_FEATURES + SCHEDULE_FEATURES + RANK_FEATURES
    + EVENT_FEATURES + VETO_FEATURES + PAGE_H2H_FEATURES
)

PRE_MAP_FEATURE_NAMES = [
    "team_a_rating_diff",
    "rank_diff",
    "team_a_map_winrate_long",
    "team_b_map_winrate_long",
    "team_a_map_winrate_short",
    "team_b_map_winrate_short",
    "team_a_overall_winrate",
    "team_b_overall_winrate",
    "h2h_win_rate",
    "recent_form_diff",
    "team_a_picked_map",
    "is_decider",
    "is_lan",
    "bo_type",
    "map_position_in_series",
    "event_tier",
    "team_a_series_score",
    "team_b_series_score",
    "elo_diff",
    "elo_map_diff",
    "elo_map_resid_diff",
    "elo_side_asym",
    "elo_games_min",
    "ranking_age_days",
] + HARVEST_FEATURE_NAMES + [f"map_{m}" for m in MAP_POOL]

_MAP_OHE_NAMES = frozenset(f"map_{m}" for m in MAP_POOL)
_SCALAR_KEYS = tuple(f for f in PRE_MAP_FEATURE_NAMES if f not in _MAP_OHE_NAMES)
_REQUIRED_KEYS = frozenset(_SCALAR_KEYS) | {"map_name"}


def assert_prior_keys_complete(priors: Mapping) -> None:
    """Raise when a model input is absent, rather than silently defaulting it to NaN."""
    missing = _REQUIRED_KEYS - set(priors)
    if missing:
        raise KeyError(f"pre-map priors missing required keys: {sorted(missing)}")


def _map_ohe(map_name: str) -> list[float]:
    return [1.0 if m == map_name else 0.0 for m in MAP_POOL]


def _f(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def extract_pre_map_features(priors: Mapping, *, strict: bool = True) -> np.ndarray:
    """
    Build the feature vector aligned with PRE_MAP_FEATURE_NAMES.

    `priors` is a cs2_match_priors row during training and the output of
    compute_live_priors at inference; both carry the same keys.
    """
    if strict:
        assert_prior_keys_complete(priors)
    scalars = [_f(priors.get(k)) for k in _SCALAR_KEYS]
    return np.array(scalars + _map_ohe(str(priors.get("map_name", ""))), dtype=np.float32)


def extract_pre_map_matrix(df, *, strict: bool = True) -> np.ndarray:
    """Vectorised equivalent of extract_pre_map_features over a priors DataFrame."""
    if strict:
        missing = _REQUIRED_KEYS - set(df.columns)
        if missing:
            raise KeyError(f"pre-map priors frame missing required columns: {sorted(missing)}")
    scalars = df[list(_SCALAR_KEYS)].astype(np.float32).to_numpy()
    maps = df["map_name"].to_numpy()
    ohe = np.stack([(maps == m).astype(np.float32) for m in MAP_POOL], axis=1)
    return np.hstack([scalars, ohe])
