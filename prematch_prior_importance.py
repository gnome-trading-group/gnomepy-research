"""
Measures prior feature importance specifically at pre-match state
(round 1, score 0-0, 5v5) vs mid-match round starts.

Run: poetry run python prematch_prior_importance.py
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.sessions.cs2_win_probability.features import (
    FEATURE_NAMES,
    extract_features_from_demo_row,
)

PRIOR_FEATURE_NAMES = [
    "team_a_rating_diff",
    "team_a_map_winrate_long",
    "team_b_map_winrate_long",
    "team_a_map_winrate_short",
    "team_b_map_winrate_short",
    "h2h_win_rate",
    "recent_form_diff",
    "team_a_picked_map",
    "is_lan",
]
PRIOR_INDICES = [FEATURE_NAMES.index(f) for f in PRIOR_FEATURE_NAMES]


def _load_priors_lookup(priors_df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in priors_df.iterrows():
        mid = row.get("match_id")
        if mid is not None:
            lookup[(int(mid), str(row["map_name"]))] = {
                "team_a_avg_rating": float(row.get("team_a_rating_diff", float("nan"))),
                "team_b_avg_rating": 0.0,
                "team_a_map_winrate_long": float(row.get("team_a_map_winrate_long", float("nan"))),
                "team_b_map_winrate_long": float(row.get("team_b_map_winrate_long", float("nan"))),
                "team_a_map_winrate_short": float(row.get("team_a_map_winrate_short", float("nan"))),
                "team_b_map_winrate_short": float(row.get("team_b_map_winrate_short", float("nan"))),
                "h2h_win_rate": float(row.get("h2h_win_rate", float("nan"))),
                "team_a_recent_form": 0.5 + float(row.get("recent_form_diff", 0.0)) / 2,
                "team_b_recent_form": 0.5 - float(row.get("recent_form_diff", 0.0)) / 2,
                "team_a_picked_map": bool(row.get("team_a_picked_map", False)),
                "is_lan": int(row.get("is_lan", 1)),
            }
    return lookup


def build_matrix(subset: pd.DataFrame, priors_lookup: dict):
    vecs_with, vecs_no = [], []
    for _, row in subset.iterrows():
        priors = None
        if row.get("match_id") is not None:
            priors = priors_lookup.get((int(row["match_id"]), str(row.get("map_name", ""))))
        vec = extract_features_from_demo_row(row, priors)
        vec_no = vec.copy()
        vec_no[PRIOR_INDICES] = np.nan
        vecs_with.append(vec)
        vecs_no.append(vec_no)
    return np.vstack(vecs_with), np.vstack(vecs_no)


def report_segment(label: str, X_with: np.ndarray, X_no: np.ndarray, model, y: np.ndarray):
    p_with = model.predict_proba(X_with)[:, 1]
    p_no = model.predict_proba(X_no)[:, 1]
    delta = np.abs(p_with - p_no)

    print(f"\n{'='*60}")
    print(f"  {label}  (n={len(delta)})")
    print(f"{'='*60}")
    print(f"  Mean P(CT wins) with priors:    {p_with.mean():.4f}")
    print(f"  Mean P(CT wins) without priors: {p_no.mean():.4f}")
    print(f"  Mean |Δ|:  {delta.mean():.4f}")
    print(f"  Median |Δ|:{np.median(delta):.4f}")
    print(f"  p75 |Δ|:   {np.percentile(delta, 75):.4f}")
    print(f"  p95 |Δ|:   {np.percentile(delta, 95):.4f}")
    print(f"  Max |Δ|:   {delta.max():.4f}")
    print(f"  CT win rate (actual): {y.mean():.4f}")

    print(f"\n  Per-feature ablation:")
    individual = []
    for idx, name in zip(PRIOR_INDICES, PRIOR_FEATURE_NAMES):
        X_abl = X_with.copy()
        X_abl[:, idx] = np.nan
        p_abl = model.predict_proba(X_abl)[:, 1]
        individual.append((name, np.abs(p_with - p_abl).mean()))
    for name, d in sorted(individual, key=lambda x: x[1], reverse=True):
        bar = "█" * int(d * 400)
        print(f"    {name:<35} {d:.5f}  {bar}")


def main():
    store = DatasetStore()
    df = store.load("cs2_round_features")
    priors_df = store.load("cs2_match_priors")
    model = joblib.load("/tmp/cs2_round_model.xgb")

    priors_lookup = _load_priors_lookup(priors_df)

    demos = df["source_demo"].unique()
    np.random.seed(42)
    np.random.shuffle(demos)
    split = int(len(demos) * 0.8)
    test_demos = set(demos[split:])
    test_df = df[df["source_demo"].isin(test_demos)].copy()

    # Pre-match: round 1, score 0-0, 5v5 — the very first snapshot of each map
    prematch = test_df[
        (test_df["round_number"] == 1) &
        (test_df["ct_score"] == 0) &
        (test_df["t_score"] == 0) &
        (test_df["ct_alive"] == 5) &
        (test_df["t_alive"] == 5)
    ]

    # Round starts mid-match (5v5, any round, score not 0-0)
    round_starts_mid = test_df[
        (test_df["ct_alive"] == 5) &
        (test_df["t_alive"] == 5) &
        ~((test_df["ct_score"] == 0) & (test_df["t_score"] == 0))
    ]

    # Late-round snapshots (≤3 players per side alive)
    late_round = test_df[
        (test_df["ct_alive"] + test_df["t_alive"]) <= 4
    ]

    X_pre_with, X_pre_no = build_matrix(prematch, priors_lookup)
    X_mid_with, X_mid_no = build_matrix(round_starts_mid, priors_lookup)
    X_late_with, X_late_no = build_matrix(late_round, priors_lookup)

    report_segment("PRE-MATCH (round 1, 0-0, 5v5)", X_pre_with, X_pre_no, model, prematch["ct_won"].values)
    report_segment("MID-MATCH ROUND START (5v5, score ≠ 0-0)", X_mid_with, X_mid_no, model, round_starts_mid["ct_won"].values)
    report_segment("LATE ROUND (≤4 players total alive)", X_late_with, X_late_no, model, late_round["ct_won"].values)


if __name__ == "__main__":
    main()
