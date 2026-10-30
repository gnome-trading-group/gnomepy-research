"""Batch entry point: writes the FACEIT and no-FACEIT tune predictions plus coverage detail into OUTPUT_DIR."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e04.predict import predict, predict_detail
from gnomepy_research.sessions.cs2_comp.e04.ratings import Params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--targets", default="/work/in/comp/targets_tune.parquet")
    args = ap.parse_args()
    out_dir = Path(os.environ["OUTPUT_DIR"])
    targets = pd.read_parquet(args.targets)
    params = Params()

    detail = predict_detail(args.data, targets, faceit=True, params=params)
    detail.to_parquet(out_dir / "predictions_detail.parquet", index=False)
    detail[["match_id", "q", "fit_cutoff"]].to_parquet(out_dir / "predictions.parquet", index=False)
    predict(args.data, targets, faceit=False, params=params).to_parquet(
        out_dir / "predictions_no_faceit.parquet", index=False)
    print(f"tune predictions: {len(detail)} series", flush=True)


if __name__ == "__main__":
    main()
