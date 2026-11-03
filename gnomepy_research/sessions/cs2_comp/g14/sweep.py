"""
Rating-engine sweep for Batch: each task runs the full pipeline for a list of configs and scores each one on the
development targets with g14's own scorer, plus the raw series log loss on the year before the dev period (no layer,
no market), so a config can be judged outside dev too.

    python -m gnomepy_research.sessions.cs2_comp.g14.sweep --configs '[{"drift_player": 1e-4}, {...}]' [--data /work/in/comp]
        [--layer '{"l2_market": 2.0}'] [--tag name]

One JSON line per config goes to $OUTPUT_DIR/sweep_<tag>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.evaluate import ll, score
from gnomepy_research.sessions.cs2_comp.g14.layer import LayerConfig
from gnomepy_research.sessions.cs2_comp.g14.pipeline import run
from gnomepy_research.sessions.cs2_comp.g14.ratings import RatingConfig, bo3, map_win

PRE_DEV = (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-04-01"))


def pre_dev_ll(rows: pd.DataFrame) -> dict:
    t = pd.to_datetime(rows.time)
    r = rows[(t >= PRE_DEV[0]) & (t < PRE_DEV[1])]
    d = (r.d_player + r.d_team).to_numpy(float)
    q = bo3(map_win(d))
    qv = 1 / (1 + np.exp(-r.z_veto.to_numpy(float)))
    qe = 1 / (1 + np.exp(-r.z_elo.to_numpy(float)))
    return {"n": int(len(r)), "ll_series": float(ll(q, r.y.to_numpy(float)).mean()),
            "ll_veto": float(ll(qv, r.y.to_numpy(float)).mean()), "ll_elo": float(ll(qe, r.y.to_numpy(float)).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--layer", default="{}")
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--tag", default="0")
    a = ap.parse_args()
    data = Path(a.data)
    targets = pd.read_parquet(data / "targets_tune.parquet")
    lcfg = LayerConfig(**json.loads(a.layer))
    out_dir = Path(os.environ.get("OUTPUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"sweep_{a.tag}.jsonl", "a") as f:
        for over in json.loads(a.configs):
            t0 = time.time()
            rcfg = RatingConfig(**over)
            res = run(data, targets, rcfg, lcfg)
            rec = {"config": over, "layer": json.loads(a.layer), "pre_dev": pre_dev_ll(res.series_rows), "seconds": time.time() - t0}
            for col in ("q", "q_model"):
                s = score(res.predictions, data, None, col, quiet=True)
                rec[col] = {k: s[k] for k in ("q_minus_market", "pool_minus_market", "q_ll", "pool_ll", "market_ll")}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
