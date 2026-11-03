"""Run predict for some weeks on Batch, writing predictions and components to $OUTPUT_DIR.
python -m ...g10.dev.batchpredict --data D --targets T --weeks W1,W2"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g10.predict import predict


def main() -> None:
    threadpool_limits(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--weeks", required=True)
    a = ap.parse_args()
    out_dir = Path(os.environ["OUTPUT_DIR"])
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = pd.to_datetime(targets.week_cutoff)
    targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, verbose=True, parts_dir=str(out_dir))
    out.to_parquet(out_dir / f"predictions_{a.weeks.split(',')[0]}.parquet", index=False)


if __name__ == "__main__":
    main()
