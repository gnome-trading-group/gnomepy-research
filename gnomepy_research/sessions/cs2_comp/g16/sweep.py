"""
Walk-forward evaluation of one skill configuration on the dev targets, from the bundle's own outcomes.

    python -m gnomepy_research.sessions.cs2_comp.g16.sweep --data <bundle> --targets <parquet> --config '<json>' [--out <dir>]

The config JSON holds a name and the values `fit.make_config` reads (filter parameters, faceit_weight, rating_gamma,
rating_sd, off_low/off_mid/off_high). Writes
<out>/<name>.json with the pool score per layer spec and <out>/<name>_{train,targets}.parquet with the skill rows,
so layer specs can be compared again without refitting. On Batch, --out defaults to $OUTPUT_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import target_paths
from gnomepy_research.sessions.cs2_comp.g16.data import Inputs, naive, target_rows
from gnomepy_research.sessions.cs2_comp.g16.fit import make_config
from gnomepy_research.sessions.cs2_comp.g16.market import path_features
from gnomepy_research.sessions.cs2_comp.g16.predict import training_rows
from gnomepy_research.sessions.cs2_comp.g16.skill import weekly_rows
from gnomepy_research.sessions.cs2_comp.g16.walk import SPECS, evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "."))
    ap.add_argument("--specs", default="default,new")
    a = ap.parse_args()
    config = json.loads(a.config)
    cfg = make_config(config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = naive(targets.week_cutoff)
    t = target_rows(targets)
    t0 = time.time()
    inputs = Inputs(a.data, t.week_cutoff.max())
    train = training_rows(inputs, cfg)
    tr = t.merge(weekly_rows(inputs, t, cfg), on=["match_id", "week_cutoff"])
    tr = tr.merge(path_features(target_paths(a.data, t.match_id)), on="match_id", how="left")
    tr = tr.merge(inputs.series[["match_id", "y"]], on="match_id", how="left")
    tr = tr[tr.y.notna() & tr.m0.notna()]
    results = {"config": config, "seconds": time.time() - t0, "n_targets": int(len(tr)), "specs": {}}
    for name in a.specs.split(","):
        results["specs"][name] = evaluate(train, tr, SPECS[name])
        print(config["name"], name, results["specs"][name], flush=True)
    (out / f"{config['name']}.json").write_text(json.dumps(results, indent=1))
    train.drop(columns=["lineup_a", "lineup_b"]).to_parquet(out / f"{config['name']}_train.parquet", index=False)
    tr.drop(columns=["lineup_a", "lineup_b"]).to_parquet(out / f"{config['name']}_targets.parquet", index=False)


if __name__ == "__main__":
    main()
