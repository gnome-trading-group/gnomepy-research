"""Batch runner: the frozen e12 baseline (copied unchanged into e12ref) on chosen weeks, written to $OUTPUT_DIR, so
g12 can measure what it adds beyond e12 without spending harness scores."""
import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.e12ref.predict import predict

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/work/in/comp")
ap.add_argument("--weeks", required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()
t = pd.read_parquet(Path(a.data) / "targets_tune.parquet")
wc = pd.to_datetime(t.week_cutoff)
t["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
t = t[t.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
out = predict(a.data, t, True, verbose=True)
out.to_parquet(Path(os.environ["OUTPUT_DIR"]) / f"{a.name}.parquet", index=False)
