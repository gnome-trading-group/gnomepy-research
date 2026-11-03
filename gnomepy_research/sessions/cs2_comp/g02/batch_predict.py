"""Batch task: run predict for some weeks and write the part to $OUTPUT_DIR."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g02 import predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weeks", type=int, nargs="+", required=True)
    args = ap.parse_args()
    targets = pd.read_parquet("/work/in/comp/targets_tune.parquet")
    cutoffs = sorted(targets.week_cutoff.unique())
    fitter = predict.Fitter("/work/in/comp")
    parts = [predict.predict_week(fitter, targets[targets.week_cutoff == cutoffs[i]], pd.Timestamp(cutoffs[i])) for i in args.weeks]
    pd.concat(parts, ignore_index=True).to_parquet(Path(os.environ["OUTPUT_DIR"]) / f"pred_{args.tag}.parquet", index=False)


if __name__ == "__main__":
    main()
