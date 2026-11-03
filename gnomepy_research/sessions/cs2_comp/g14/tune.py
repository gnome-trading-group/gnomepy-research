"""
Local tuner: load the bundle once, then run the pass and the layers for each config and print how each did, on the
year before dev (raw series log loss, no layer) and on dev with g14's own scorer.

    python -m gnomepy_research.sessions.cs2_comp.g14.tune --configs '[{"drift_player": 1e-4}, {}]' [--layer '{...}']
        [--no-faceit] [--data <bundle>] [--out <jsonl>]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.data import load_inputs
from gnomepy_research.sessions.cs2_comp.g14.evaluate import score
from gnomepy_research.sessions.cs2_comp.g14.layer import LayerConfig
from gnomepy_research.sessions.cs2_comp.g14.pipeline import prepare_targets, run
from gnomepy_research.sessions.cs2_comp.g14.ratings import RatingConfig
from gnomepy_research.sessions.cs2_comp.g14.sweep import pre_dev_ll

DATA = "/Users/mprey/gnome-research-work/cs2/groundup/comp/data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--layers", default="[{}]")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out")
    ap.add_argument("--e12")
    a = ap.parse_args()
    data = Path(a.data)
    targets = prepare_targets(pd.read_parquet(data / "targets_tune.parquet"))
    e12 = pd.read_parquet(a.e12) if a.e12 else None
    t0 = time.time()
    inputs = load_inputs(data, targets.week_cutoff.max(), a.faceit)
    print(f"loaded {len(inputs.maps)} maps in {time.time() - t0:.0f}s", flush=True)
    out = open(a.out, "a") if a.out else None
    for over in json.loads(a.configs):
        for lover in json.loads(a.layers):
            t0 = time.time()
            rcfg = RatingConfig(**{**over, "faceit": a.faceit})
            res = run(data, targets, rcfg, LayerConfig(**lover), inputs=inputs)
            rec = {"config": over, "layer": lover, "pre_dev": pre_dev_ll(res.series_rows), "seconds": round(time.time() - t0, 1)}
            for col in ("q", "q_model"):
                s = score(res.predictions, data, e12, col, quiet=True)
                rec[col] = {k: s[k] for k in s if k in ("q_minus_market", "pool_minus_market", "stack_minus_baseline_pool", "q_ll", "pool_ll")}
            pm = rec["q"]["pool_minus_market"]
            pmm = rec["q_model"]["pool_minus_market"]
            stack = rec["q"].get("stack_minus_baseline_pool", {}).get("mean")
            print(f"{json.dumps(over)} {json.dumps(lover)} pre_dev ll {rec['pre_dev']['ll_series']:.4f} veto {rec['pre_dev']['ll_veto']:.4f} elo {rec['pre_dev']['ll_elo']:.4f} | "
                  f"q: pool-mkt {pm['mean']:+.4f} [{pm['lo']:+.4f},{pm['hi']:+.4f}] q_ll {rec['q']['q_ll']:.4f} | "
                  f"q_model: pool-mkt {pmm['mean']:+.4f} q_ll {rec['q_model']['q_ll']:.4f}"
                  + (f" | stack-base {stack:+.4f}" if stack is not None else "") + f" {rec['seconds']}s", flush=True)
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
