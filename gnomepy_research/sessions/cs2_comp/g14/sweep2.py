"""
Batch sweep over batch-fit configs: for each config, the monthly walk-forward rows (layer training), then the weekly
dev predictions with layers fitted on those rows, scored with g14's own scorer. Rows, predictions and a JSON line per
config go to $OUTPUT_DIR.

    python -m gnomepy_research.sessions.cs2_comp.g14.sweep2 --configs '[{"tau": 0.05}, {}]' --tag a [--data /work/in/comp]
        [--cutoffs 2025-03-02,...] [--layer '{...}'] [--no-faceit]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.evaluate import score
from gnomepy_research.sessions.cs2_comp.g14.layer import LayerConfig
from gnomepy_research.sessions.cs2_comp.g14.pipeline import run
from gnomepy_research.sessions.cs2_comp.g14.ratings import RatingConfig
from gnomepy_research.sessions.cs2_comp.g14.skillfit import FitConfig
from gnomepy_research.sessions.cs2_comp.g14.walk import walk

MONTHLY = ("2025-03-03,2025-04-07,2025-05-05,2025-06-02,2025-07-07,2025-08-04,2025-09-01,2025-10-06,2025-11-03,2025-12-01,"
           "2026-01-05,2026-02-02,2026-03-02,2026-04-06,2026-05-04,2026-06-01,2026-07-06,2026-08-03")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--cutoffs", default=MONTHLY)
    ap.add_argument("--layer", default="{}")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    a = ap.parse_args()
    data = Path(a.data)
    out_dir = Path(os.environ.get("OUTPUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_parquet(data / "targets_tune.parquet")
    rcfg = RatingConfig(faceit=a.faceit)
    lcfg = LayerConfig(**json.loads(a.layer))
    for i, over in enumerate(json.loads(a.configs)):
        t0 = time.time()
        fcfg = FitConfig(**over)
        rows = walk(data, [pd.Timestamp(c) for c in a.cutoffs.split(",")], rcfg, fcfg)
        rows.to_parquet(out_dir / f"walk_{a.tag}_{i}.parquet", index=False)
        res = run(data, targets, rcfg, lcfg, verbose=True, fcfg=fcfg, train_rows=rows)
        res.predictions.to_parquet(out_dir / f"pred_{a.tag}_{i}.parquet", index=False)
        res.target_rows.to_parquet(out_dir / f"targets_{a.tag}_{i}.parquet", index=False)
        rec = {"config": over, "layer": json.loads(a.layer), "seconds": round(time.time() - t0)}
        for col in ("q", "q_model"):
            s = score(res.predictions, data, None, col, quiet=True)
            rec[col] = {k: s[k] for k in ("q_minus_market", "pool_minus_market", "q_ll", "pool_ll", "market_ll")}
        with open(out_dir / f"sweep_{a.tag}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
