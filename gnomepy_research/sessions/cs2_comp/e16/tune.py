"""
Dev-period variants on Batch: builds the Jan-Mar 2026 dev targets from the bundle, prices them with one Config
variant and writes the predictions to $OUTPUT_DIR (scored locally with dev.score).

    python -m gnomepy_research.sessions.cs2_comp.e16.tune --data /work/in/comp --name <variant> [--no-faceit]
        [--set name=value ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e16 import dev
from gnomepy_research.sessions.cs2_comp.e16.model import Config
from gnomepy_research.sessions.cs2_comp.e16.predict import parse_overrides, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    out = Path(os.environ.get("OUTPUT_DIR", "."))
    targets = out / f"dev_targets_{args.name}.parquet"
    dev.build(args.data, targets)
    cfg = dataclasses.replace(Config(), **parse_overrides(args.set))
    pred = run(args.data, pd.read_parquet(targets), args.faceit, cfg)
    pred.to_parquet(out / f"dev_{args.name}.parquet", index=False)


if __name__ == "__main__":
    main()
