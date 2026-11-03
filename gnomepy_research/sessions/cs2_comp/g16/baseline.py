"""
Run the copied frozen baseline e12 on the dev targets and keep its prices, so that what g16 knows beyond the market
and e12 together can be measured before spending a harness score.

    python -m gnomepy_research.sessions.cs2_comp.g16.baseline [--data <bundle>] [--targets <parquet>] [--out <dir>]

The defaults are the Batch paths, so the job needs no arguments.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g16.data import naive
from gnomepy_research.sessions.cs2_comp.g16.e12.predict import predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--targets", default="/work/in/comp/targets_tune.parquet")
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "."))
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = naive(targets.week_cutoff)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    predict(a.data, targets, faceit=True, verbose=True).to_parquet(out / "e12.parquet", index=False)


if __name__ == "__main__":
    main()
