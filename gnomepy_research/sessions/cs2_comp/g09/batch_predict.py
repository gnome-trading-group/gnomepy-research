"""Run g09's predict for some weeks on Batch and keep the predictions in $OUTPUT_DIR.

    python -m gnomepy_research.sessions.cs2_comp.g09.batch_predict <week_cutoff,...>"""
import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g09.core import naive
from gnomepy_research.sessions.cs2_comp.g09.predict import predict

DATA = Path("/work/in/comp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weeks")
    a = ap.parse_args()
    t = pd.read_parquet(DATA / "targets_tune.parquet")
    t["week_cutoff"] = naive(t.week_cutoff)
    weeks = pd.to_datetime(a.weeks.split(","))
    out = predict(DATA, t[t.week_cutoff.isin(weeks)], verbose=True)
    out.to_parquet(Path(os.environ["OUTPUT_DIR"]) / f"pred_{weeks[0].date()}.parquet", index=False)


if __name__ == "__main__":
    main()
