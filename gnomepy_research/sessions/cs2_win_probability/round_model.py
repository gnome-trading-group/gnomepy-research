"""
Layer 3: P(team_a wins the round in progress), overlaid on the map DP.

    P(map) = p_round * V(a+1, b) + (1 - p_round) * V(a, b+1)

The round model prices only the live round; the DP prices everything after. That
decomposition is chosen for statistical efficiency, not elegance — the round
target has ~11k independent outcomes in train, against ~540 independent map
outcomes if a model predicted the map directly from a snapshot. The DP supplies
the map structure analytically instead of learning it from 540 examples.

Everything is framed team_a-relative. The previous model mixed frames: its state
features were CT/T-keyed against a ct_won label while its nine prior features
were team_a-keyed, with nothing linking them, so the priors were sign-scrambled
against the label roughly half the time.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.pipelines.hltv_cs2.config import MAP_POOL

logger = logging.getLogger(__name__)

_SIDE_PAIRS = {
    "equip": ("ct_equip_value", "t_equip_value"),
    "alive": ("ct_alive", "t_alive"),
    "hp": ("ct_hp", "t_hp"),
    "money": ("ct_money", "t_money"),
    "armor": ("ct_armor", "t_armor"),
    "armored": ("ct_armored", "t_armored"),
    "alive_equip": ("ct_alive_equip", "t_alive_equip"),
    "round_start_equip": ("ct_round_start_equip", "t_round_start_equip"),
    "consecutive_losses": ("ct_consecutive_losses", "t_consecutive_losses"),
}

STATE_FEATURES = (
    [f"d_{k}" for k in _SIDE_PAIRS]
    + [f"a_{k}" for k in ("alive", "hp", "equip", "alive_equip")]
    + [f"b_{k}" for k in ("alive", "hp", "equip", "alive_equip")]
    + ["team_a_is_ct", "bomb_planted", "bomb_favours_a", "bomb_time_remaining",
       "round_number", "current_half", "team_a_score_before", "team_b_score_before",
       "score_diff", "rounds_remaining_to_win"]
)

PRIOR_FEATURES = [
    "elo_diff", "elo_map_diff", "elo_side_asym", "team_a_rating_diff", "rank_diff",
    "team_a_map_winrate_long", "team_b_map_winrate_long",
    "team_a_overall_winrate", "team_b_overall_winrate",
    "h2h_win_rate", "recent_form_diff", "is_lan", "event_tier",
]

ROUND_FEATURE_NAMES = STATE_FEATURES + PRIOR_FEATURES + [f"map_{m}" for m in MAP_POOL]


def orient_to_team_a(rounds: pd.DataFrame) -> pd.DataFrame:
    """
    Re-express side-keyed snapshot columns in team_a's frame.

    Requires team_a_is_ct, which the cs2_round_features repair attaches.
    """
    if "team_a_is_ct" not in rounds.columns:
        raise KeyError("cs2_round_features is missing team_a_is_ct — run the sides repair first")

    df = rounds.copy()
    is_ct = df["team_a_is_ct"].to_numpy().astype(bool)
    for name, (ct_col, t_col) in _SIDE_PAIRS.items():
        ct_vals, t_vals = df[ct_col].to_numpy(float), df[t_col].to_numpy(float)
        a = np.where(is_ct, ct_vals, t_vals)
        b = np.where(is_ct, t_vals, ct_vals)
        df[f"a_{name}"], df[f"b_{name}"], df[f"d_{name}"] = a, b, a - b

    df["team_a_is_ct"] = is_ct.astype(float)
    # a planted bomb helps whoever is on T
    df["bomb_favours_a"] = df["bomb_planted"].to_numpy(float) * np.where(is_ct, -1.0, 1.0)
    df["score_diff"] = df["team_a_score_before"] - df["team_b_score_before"]
    df["rounds_remaining_to_win"] = 13 - df[["team_a_score_before", "team_b_score_before"]].max(axis=1)
    return df


def snapshot_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Down-weight rounds that produced many snapshots.

    Every snapshot in a round carries that round's single outcome, so a 20-kill
    round would otherwise count twenty times against a 3-kill round.
    """
    counts = df.groupby(["match_id", "map_name", "round_number"])["round_number"].transform("size")
    return (1.0 / counts).to_numpy(float)


def build_round_frame() -> pd.DataFrame:
    """Repaired snapshots joined to their map's priors and outcome, oriented to team_a."""
    ds = DatasetStore()
    rounds = ds.load("cs2_round_features")
    priors = ds.load("cs2_match_priors")
    history = ds.load("cs2_match_history")[
        ["match_id", "map_name", "team_a_won", "team_a_started_ct"]
    ]

    prior_cols = ["match_id", "map_name"] + [c for c in PRIOR_FEATURES if c in priors.columns]
    df = rounds.merge(priors[prior_cols], on=["match_id", "map_name"], how="inner")
    df = df.merge(history, on=["match_id", "map_name"], how="inner")
    return orient_to_team_a(df).sort_values(["match_date", "match_id", "round_number"]).reset_index(drop=True)


def round_matrix(df: pd.DataFrame, names: list[str] | None = None) -> np.ndarray:
    names = names or ROUND_FEATURE_NAMES
    ohe_names = {f"map_{m}" for m in MAP_POOL}
    scalars = [n for n in names if n not in ohe_names]
    cols = df.reindex(columns=scalars).astype(np.float32).to_numpy()
    wanted = [n for n in names if n in ohe_names]
    if not wanted:
        return cols
    maps = df["map_name"].to_numpy()
    ohe = np.stack([(maps == n.removeprefix("map_")).astype(np.float32) for n in wanted], axis=1)
    return np.hstack([cols, ohe])
