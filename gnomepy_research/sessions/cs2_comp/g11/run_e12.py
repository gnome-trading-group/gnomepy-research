"""Batch wrapper: the frozen e12 baseline (copied here) on the dev targets, for local stacking diagnostics.

    python -m gnomepy_research.sessions.cs2_comp.g11.run_e12 <comma-separated week cutoffs>
"""
import os
import sys
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g11.e12.predict import predict

data = Path("/work/in/comp")
out = Path(os.environ["OUTPUT_DIR"])
weeks = pd.to_datetime(sys.argv[1].split(","))
t = pd.read_parquet(data / "targets_tune.parquet")
wc = pd.to_datetime(t.week_cutoff)
t["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
t = t[t.week_cutoff.isin(weeks)]
p = predict(data, t, True, verbose=True, rows_dir=out / f"rows_{weeks[0]:%Y%m%d}")
p.to_parquet(out / f"e12_{weeks[0]:%Y%m%d}.parquet", index=False)
