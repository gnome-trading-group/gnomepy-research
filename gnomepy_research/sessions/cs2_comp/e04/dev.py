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
from gnomepy_research.sessions.cs2_comp.e04.predict import predict
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


def logloss(q: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def score(targets: pd.DataFrame, q: np.ndarray) -> dict:
    y = targets.y.to_numpy()
    ll = np.array([-(yy * np.log(np.clip(qq, 1e-4, 1 - 1e-4)) + (1 - yy) * np.log(1 - np.clip(qq, 1e-4, 1 - 1e-4)))
                   for yy, qq in zip(y, q)])
    return {"n": int(len(y)), "logloss": float(ll.mean()), "se": float(ll.std(ddof=1) / np.sqrt(len(ll)))}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    args = ap.parse_args()
    targets = dev_targets(args.data)
    base = Params()
    results = {"n_targets": int(len(targets)), "base_coinflip": score(targets, np.full(len(targets), 0.5))}

    configs = {
        "no_faceit": (False, base),
        "faceit_default": (True, base),
    }
    for name, val in [("k_team", [16.0, 48.0]), ("overlap_power", [1.0, 4.0]), ("shrink_games", [0.0, 30.0]),
                      ("faceit_level_shift", [-100.0, 100.0]), ("k_faceit_mult", [0.3, 1.0]),
                      ("k_player", [6.0, 24.0])]:
        for v in val:
            configs[f"faceit_{name}={v}"] = (True, replace(base, **{name: v}))

    for name, (faceit, params) in configs.items():
        pred = predict(args.data, targets, faceit=faceit, params=params)
        merged = targets.merge(pred, on="match_id", how="left")
        results[name] = score(targets, merged.q.to_numpy())
        results[name]["params"] = asdict(params)
        results[name]["faceit"] = faceit
        print(name, results[name]["logloss"], flush=True)

    fit_base = predict(args.data, targets, faceit=True, params=base)
    merged = targets.merge(fit_base, on="match_id", how="left")
    q1 = merged.q.to_numpy()
    cal = {}
    for s in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        cal[str(s)] = score(targets, sigmoid(s * logit(q1)))["logloss"]
    results["calibration_scan_faceit_default"] = cal

    Path(os.environ["OUTPUT_DIR"], "dev_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps({k: v for k, v in results.items() if isinstance(v, dict) and "logloss" in v}, indent=1, default=str))


if __name__ == "__main__":
    main()
