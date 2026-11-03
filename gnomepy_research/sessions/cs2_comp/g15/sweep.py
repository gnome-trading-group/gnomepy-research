"""
Run several batch-fit configurations in one process over the same walk-forward weeks, one row file each.

    python -m gnomepy_research.sessions.cs2_comp.g15.sweep --data <bundle> --out-dir <dir> --first ... --last ...
        --configs '{"hl150": {"half_life_days": 150}, "s35": {"sigma": 0.35}}' [--faceit]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g15.data import load_series
from gnomepy_research.sessions.cs2_comp.g15.features import history_jobs
from gnomepy_research.sessions.cs2_comp.g15.fit_rows import fit_rows
from gnomepy_research.sessions.cs2_comp.g15.skillfit import FitConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--faceit", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    first, last = pd.Timestamp(a.first), pd.Timestamp(a.last)
    jobs = history_jobs(load_series(a.data, last + pd.Timedelta(days=7)), first, last)
    for name, kw in json.loads(a.configs).items():
        cfg = FitConfig(**kw)
        rows = fit_rows(a.data, jobs, cfg, a.faceit, prefix=name, verbose=False)
        rows.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name} {cfg} -> {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
