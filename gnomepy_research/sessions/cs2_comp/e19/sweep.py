"""
Run several filter configurations in one process (one gate slot) and write each one's weekly channel prices.

    python -m gnomepy_research.sessions.cs2_comp.e19.sweep --data <bundle> --targets <parquet> --out-dir <dir>
        --dev-from 2025-12-01 --weeks 2026-03-30 name1='{"faceit_weight": 0.5}' name2='{}' ...
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e19.filter import FilterConfig
from gnomepy_research.sessions.cs2_comp.e19.predict import _naive, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dev-from", required=True)
    ap.add_argument("--weeks", required=True)
    ap.add_argument("--no-faceit", action="store_true")
    ap.add_argument("configs", nargs="+", help="name=<json overrides of FilterConfig>")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = _naive(targets.week_cutoff)
    targets = targets[targets.week_cutoff.isin([pd.Timestamp(w) for w in a.weeks.split(",")])]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in a.configs:
        name, overrides = spec.split("=", 1)
        cfg = replace(FilterConfig(), **json.loads(overrides))
        print(f"== {name} {overrides}", flush=True)
        preds, weekly = run(a.data, targets, cfg, faceit=not a.no_faceit, dev_from=pd.Timestamp(a.dev_from),
                            log=lambda s: print(s, flush=True))
        weekly.to_parquet(a.out_dir / f"dev_{name}.parquet", index=False)
        preds[["match_id", "q", "fit_cutoff"]].to_parquet(a.out_dir / f"pred_{name}.parquet", index=False)


if __name__ == "__main__":
    main()
