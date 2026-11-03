"""
e12 prices for every bo3 series with a Polymarket path (the recalibration layer's training rows), each from the fit at
its own week_cutoff, or at a coarser cutoff given with --cutoff-rule month.

    python -m gnomepy_research.sessions.cs2_comp.g06.e12_hist --data <bundle> --weeks 2026-03-02,2026-03-09 --out <dir>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g06.e12.predict import predict
from gnomepy_research.sessions.cs2_comp.g06.history import market_series_targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weeks", required=True, help="comma-separated week_cutoff dates")
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "."))
    a = ap.parse_args()
    targets = market_series_targets(a.data, pd.Timestamp("2100-01-01"))
    weeks = pd.to_datetime(a.weeks.split(","))
    targets = targets[targets.week_cutoff.isin(weeks)]
    out = predict(a.data, targets, faceit=True, verbose=True, rows_dir=Path(a.out) / "rows")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(a.out) / f"e12_{a.weeks.split(',')[0]}.parquet", index=False)
    print(f"wrote {len(out)}")


if __name__ == "__main__":
    main()
