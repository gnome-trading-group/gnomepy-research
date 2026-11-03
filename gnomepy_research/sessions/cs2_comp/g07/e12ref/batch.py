"""Run the frozen e12 (copied here as a reference) on some weeks of the targets inside a Batch task, so its dev prices
are available for local stack diagnostics.

    python -m gnomepy_research.sessions.cs2_comp.g07.e12ref.batch --weeks 2026-03-30,2026-04-06"""
import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g07.e12ref.predict import predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--weeks", required=True)
    a = ap.parse_args()
    targets = pd.read_parquet(Path(a.data) / "targets_tune.parquet")
    targets["week_cutoff"] = pd.to_datetime(targets.week_cutoff)
    targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, True, verbose=True)
    tag = a.weeks.split(",")[0]
    out.to_parquet(Path(os.environ.get("OUTPUT_DIR", ".")) / f"e12_{tag}.parquet", index=False)


if __name__ == "__main__":
    main()
