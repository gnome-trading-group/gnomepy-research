"""Dev sweep on Sep 2025 - Feb 2026 HLTV series with HLTV outcomes only (no market), one parameter at a time."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.e04.predict import predict, predict_detail
from gnomepy_research.sessions.cs2_comp.e04.ratings import Params

DEV_START = pd.Timestamp("2025-09-01")
DEV_END = pd.Timestamp("2026-03-01")


def dev_targets(data_dir: str) -> pd.DataFrame:
    h = pd.read_parquet(f"{data_dir}/hltv/cs2_match_history.parquet", columns=[
        "match_id", "match_date", "match_time", "bo_type", "team_a_id", "team_b_id",
        "team_a_series_score", "team_b_series_score", "map_position_in_series",
        "event_name", "format",
    ])
    h = h[~showmatch_mask(h)]
    h = h[(h.match_date >= DEV_START) & (h.match_date < DEV_END)]
    last = h.sort_values(["match_id", "map_position_in_series"]).groupby("match_id").tail(1)
    week = last.match_date.dt.normalize() - pd.to_timedelta(last.match_date.dt.weekday, unit="D")
    return pd.DataFrame({
        "match_id": last.match_id.to_numpy(),
        "match_time": last.match_time.to_numpy(),
        "match_date": last.match_date.to_numpy(),
        "team_a_id": last.team_a_id.to_numpy(),
        "team_b_id": last.team_b_id.to_numpy(),
        "bo_type": last.bo_type.fillna(3).astype(int).to_numpy(),
        "week_cutoff": week.to_numpy(),
        "y": (last.team_a_series_score > last.team_b_series_score).astype(float).to_numpy(),
    }).dropna(subset=["team_a_id", "team_b_id"]).reset_index(drop=True)


def nll(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def paired_row(data_dir: str, targets: pd.DataFrame, params: Params) -> dict:
    detail = predict_detail(data_dir, targets, faceit=True, params=params)
    nof = predict(data_dir, targets, faceit=False, params=params)
    merged = targets.merge(detail, on="match_id", how="left")
    merged["nof_q"] = targets.merge(nof, on="match_id", how="left").q.to_numpy()
    y = merged.y.to_numpy()
    ll_f = nll(merged.q.to_numpy(), y)
    ll_n = nll(merged.nof_q.to_numpy(), y)
    diff = ll_f - ll_n
    thin = np.minimum(merged.faceit_players_a, merged.faceit_players_b).to_numpy()
    buckets = {"0": thin == 0, "1-3": (thin >= 1) & (thin <= 3), "4-5": thin >= 4}
    cov = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            cov[name] = {"n": 0}
            continue
        cov[name] = {"n": int(mask.sum()), "faceit_minus_nofaceit": float(diff[mask].mean()),
                     "se": float(diff[mask].std(ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else None}
    return {
        "ll_faceit": float(ll_f.mean()),
        "ll_nofaceit": float(ll_n.mean()),
        "paired_diff": float(diff.mean()),
        "paired_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "coverage": cov,
        "q_faceit": merged.q.to_numpy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    args = ap.parse_args()
    targets = dev_targets(args.data)
    base = Params()
    grid = {
        "shrink0_kfm0.6": replace(base, shrink_games=0.0),
        "shrink4_kfm0.6": replace(base, shrink_games=4.0),
        "shrink12_kfm0.6": replace(base, shrink_games=12.0),
        "shrink0_kfm0.3": replace(base, shrink_games=0.0, k_faceit_mult=0.3),
        "shrink4_kfm0.3": replace(base, shrink_games=4.0, k_faceit_mult=0.3),
        "shrink12_kfm0.3": replace(base, shrink_games=12.0, k_faceit_mult=0.3),
        "shrink0_kfm1.0": replace(base, shrink_games=0.0, k_faceit_mult=1.0),
        "shrink0_kfm0.3_overlap1": replace(base, shrink_games=0.0, k_faceit_mult=0.3, overlap_power=1.0),
        "shrink0_kfm0.3_overlap4": replace(base, shrink_games=0.0, k_faceit_mult=0.3, overlap_power=4.0),
        "shrink0_kfm0.3_kteam24": replace(base, shrink_games=0.0, k_faceit_mult=0.3, k_team=24.0),
        "shrink0_kfm0.3_kteam48": replace(base, shrink_games=0.0, k_faceit_mult=0.3, k_team=48.0),
    }
    results = {"n_targets": int(len(targets)), "coinflip_ll": float(nll(np.full(len(targets), 0.5), targets.y.to_numpy()).mean())}
    for name, params in grid.items():
        row = paired_row(args.data, targets, params)
        row.pop("q_faceit")
        row["params"] = asdict(params)
        results[name] = row
        print(name, round(row["ll_nofaceit"], 5), round(row["ll_faceit"], 5), round(row["paired_diff"], 5),
              round(row["paired_se"], 5), flush=True)
    Path(os.environ["OUTPUT_DIR"], "dev_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("params",)} for k, v in results.items()
                      if isinstance(v, dict) and "paired_diff" in v}, indent=1, default=str))


if __name__ == "__main__":
    main()
