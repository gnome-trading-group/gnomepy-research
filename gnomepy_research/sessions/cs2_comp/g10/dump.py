"""Write the feature rows (past series and targets) for development: python -m ...g10.dump --data D --targets T --out P"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g10.rows import build


def main() -> None:
    # Batch containers see every host core but get a share of them; one BLAS thread per process avoids thrashing.
    threadpool_limits(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--weeks")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = pd.to_datetime(targets.week_cutoff)
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = a.out or str(Path(os.environ["OUTPUT_DIR"]) / "rows.parquet")
    build(a.data, targets, verbose=True).to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
