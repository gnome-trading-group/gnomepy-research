"""
Walk-forward feature rows for development: every bo3 series of the weeks from --first to --last, each priced at its
own weekly cutoff, with its outcome and price path.

    python -m gnomepy_research.sessions.cs2_comp.g13.dev_rows --data <bundle> --out <parquet> --first 2026-02-23 --last 2026-04-27
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g13.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.g13.rows import WEEK, history_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    a = ap.parse_args()
    cutoffs = list(pd.date_range(a.first, a.last, freq=WEEK))
    inputs = Inputs(a.data, cutoffs[-1] + WEEK, faceit=True)
    rows = history_rows(a.data, inputs, cutoffs, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(a.out, index=False)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
