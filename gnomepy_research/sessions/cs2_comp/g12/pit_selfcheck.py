"""Batch self-check of the predict contract: predictions from the full bundle must match reruns on bundles cut at
single week cutoffs (cut with the organizer's pit_check.py, passed in as an input file)."""
from __future__ import annotations

import argparse
import os
import runpy
import tempfile
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.data import naive
from gnomepy_research.sessions.cs2_comp.g12.predict import predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--pit", default="/work/in/pit/pit_check.py")
    ap.add_argument("--weeks", required=True)
    a = ap.parse_args()
    out = Path(os.environ.get("OUTPUT_DIR", "."))
    pit = runpy.run_path(a.pit)
    pit["cut"].__globals__["DATA"] = Path(a.data)
    t = pd.read_parquet(Path(a.data) / "targets_tune.parquet")
    t["week_cutoff"] = naive(t.week_cutoff)
    weeks = pd.to_datetime(a.weeks.split(","))
    full = predict(a.data, t)
    full.to_parquet(out / "full.parquet", index=False)
    for w in weeks:
        with tempfile.TemporaryDirectory() as d:
            pit["cut"](str(w.date()), Path(d))
            tw = pd.read_parquet(Path(d) / "targets_tune.parquet")
            tw["week_cutoff"] = naive(tw.week_cutoff)
            rerun = predict(d, tw)
            rerun.to_parquet(out / f"rerun_{w.date()}.parquet", index=False)
            pit["compare"](out / "full.parquet", out / f"rerun_{w.date()}.parquet")


if __name__ == "__main__":
    main()
