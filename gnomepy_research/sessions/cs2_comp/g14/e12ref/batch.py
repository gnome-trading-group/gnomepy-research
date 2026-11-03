"""Batch task: the frozen e12 baseline's predictions for a few weeks, for the g14 stack diagnostics.

    python -m gnomepy_research.sessions.cs2_comp.g14.e12ref.batch --weeks 2026-04-06,2026-04-13 [--data <bundle>]

Writes pred_<first week>.parquet to $OUTPUT_DIR."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.e12ref.predict import predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", required=True)
    ap.add_argument("--data", default="/work/in/comp")
    a = ap.parse_args()
    targets = pd.read_parquet(Path(a.data) / "targets_tune.parquet")
    wc = pd.to_datetime(targets.week_cutoff)
    targets["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    weeks = pd.to_datetime(a.weeks.split(","))
    targets = targets[targets.week_cutoff.isin(weeks)]
    out = predict(a.data, targets, faceit=True, verbose=True)
    out_dir = Path(os.environ.get("OUTPUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"pred_{weeks.min():%Y%m%d}.parquet", index=False)
    print(f"wrote {len(out)} predictions")


if __name__ == "__main__":
    main()
