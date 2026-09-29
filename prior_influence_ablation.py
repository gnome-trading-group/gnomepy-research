"""
Measures how much prior features shift P(CT wins) vs mechanical features,
grouped by total players alive (proxy for round progression).

Run: poetry run python prior_influence_ablation.py
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


def main():
    store = DatasetStore()
    df = store.load("cs2_round_features")
    priors_df = store.load("cs2_match_priors")
    model = joblib.load("/tmp/cs2_round_model.xgb")

    priors_lookup = _load_priors_lookup(priors_df)

    # Same demo split as training (seed 42, 80/20)
    demos = df["source_demo"].unique()
    np.random.seed(42)
    np.random.shuffle(demos)
    split = int(len(demos) * 0.8)
    test_demos = set(demos[split:])
    test_df = df[df["source_demo"].isin(test_demos)].copy()

    print(f"Test set: {len(test_df)} rows, {len(test_demos)} demos\n")

    rows_with_priors = []
    rows_no_priors = []
    players_alive_total = []

    for _, row in test_df.iterrows():
        priors = None
        if row.get("match_id") is not None:
            priors = priors_lookup.get((int(row["match_id"]), str(row.get("map_name", ""))))

        vec_with = extract_features_from_demo_row(row, priors)
        vec_no = vec_with.copy()
        vec_no[PRIOR_INDICES] = np.nan

        rows_with_priors.append(vec_with)
        rows_no_priors.append(vec_no)
        players_alive_total.append(
            int(row.get("ct_alive", 5)) + int(row.get("t_alive", 5))
        )

    X_with = np.vstack(rows_with_priors)
    X_no = np.vstack(rows_no_priors)

    p_with = model.predict_proba(X_with)[:, 1]
    p_no = model.predict_proba(X_no)[:, 1]

    delta = np.abs(p_with - p_no)
    alive = np.array(players_alive_total)

    print(f"{'Players alive (ct+t)':<24} {'N':>6}  {'mean |Δ|':>10}  {'p95 |Δ|':>10}  {'mean P(CT) w/ priors':>22}  {'mean P(CT) no priors':>22}")
    print("-" * 100)
    for total in sorted(np.unique(alive), reverse=True):
        mask = alive == total
        d = delta[mask]
        pw = p_with[mask]
        pn = p_no[mask]
        print(f"{total:<24} {mask.sum():>6}  {d.mean():>10.4f}  {np.percentile(d, 95):>10.4f}  {pw.mean():>22.4f}  {pn.mean():>22.4f}")

    print()
    print(f"{'Overall':<24} {len(delta):>6}  {delta.mean():>10.4f}  {np.percentile(delta, 95):>10.4f}")

    # Top prior features by mean |delta| when ablated individually
    print("\n--- Per-feature ablation (mean |Δ P(CT)| when that feature alone → NaN) ---")
    individual_deltas = []
    for idx, name in zip(PRIOR_INDICES, PRIOR_FEATURE_NAMES):
        X_ablate = X_with.copy()
        X_ablate[:, idx] = np.nan
        p_ablate = model.predict_proba(X_ablate)[:, 1]
        mean_d = np.abs(p_with - p_ablate).mean()
        individual_deltas.append((name, mean_d))

    individual_deltas.sort(key=lambda x: x[1], reverse=True)
    for name, d in individual_deltas:
        print(f"  {name:<35} {d:.5f}")


if __name__ == "__main__":
    main()
