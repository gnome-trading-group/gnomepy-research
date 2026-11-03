"""Reproduce the frozen e12 prices on Batch, so stack-style comparisons against e12 can be run locally.

    python -m gnomepy_research.sessions.cs2_comp.g09.batch_e12 <week_cutoff,...> [--targets <parquet>]"""
import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g09.e12ref.predict import predict

DATA = Path("/work/in/comp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weeks")
    ap.add_argument("--targets", default=str(DATA / "targets_tune.parquet"))
    a = ap.parse_args()
    t = pd.read_parquet(a.targets)
    t["week_cutoff"] = pd.to_datetime(t.week_cutoff)
    weeks = pd.to_datetime(a.weeks.split(","))
    t = t[t.week_cutoff.isin(weeks)]
    out = predict(DATA, t, True, verbose=True)
    out.to_parquet(Path(os.environ["OUTPUT_DIR"]) / f"e12_{weeks[0].date()}.parquet", index=False)


if __name__ == "__main__":
    main()
