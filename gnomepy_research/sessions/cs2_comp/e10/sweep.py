"""Dev sweep: one event build, many filter configurations, each scored by evaluate.summarize.

    python -m gnomepy_research.sessions.cs2_comp.e10.sweep --data <bundle> --out <dir> --configs '[{"name": "x", "overrides": {...}}]'
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e10.evaluate import DEV_LAST, summarize
from gnomepy_research.sessions.cs2_comp.e10.predict import build_config
from gnomepy_research.sessions.cs2_comp.e10.run import TRAIN_START, Pass, build_shared, monday_range


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "out")))
    ap.add_argument("--configs", required=True)
    a = ap.parse_args()
    configs = json.loads(a.configs) if a.configs.startswith("[") else json.loads(Path(a.configs).read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    last = DEV_LAST + pd.Timedelta(days=7)
    needs_faceit = any(c.get("faceit", True) for c in configs)
    shared = build_shared(a.data, last, build_config(needs_faceit, {"h_gamma": 0.1}))
    print(f"shared build {time.time() - t0:.0f}s events {shared.info.get('hltv_maps')} + {shared.info.get('faceit_maps')}", flush=True)
    print(json.dumps(shared.info.get("rating_fit"), default=str), flush=True)
    for c in configs:
        t1 = time.time()
        cfg = build_config(c.get("faceit", True), c.get("overrides", {}))
        p = Pass(a.data, last, cfg, shared)
        feats = p.run(monday_range(TRAIN_START, DEV_LAST), None)
        feats.to_parquet(a.out / f"{c['name']}.parquet", index=False)
        res = summarize(feats)
        res.update({"name": c["name"], "config": c, "pass_s": round(time.time() - t1)})
        (a.out / f"{c['name']}.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps({k: v for k, v in res.items() if k != "config"}, default=str), flush=True)


if __name__ == "__main__":
    main()
