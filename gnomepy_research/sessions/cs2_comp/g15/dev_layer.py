"""
Fit predict.py's layer weights on walk-forward rows before a cutoff and print them.

    python -m gnomepy_research.sessions.cs2_comp.g15.dev_layer --rows <rows> --extra <fit rows>... --cols one,z_f,z_g
        --before 2026-03-30 [--warmup 2024-09-01]

Column names follow `layer.design`; a fit row file's prefix names its z column (z_<prefix>).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g15.layer import design, fit_logistic, log_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--cols", required=True)
    ap.add_argument("--before", default="2026-03-30")
    ap.add_argument("--warmup", default="2024-09-01")
    a = ap.parse_args()
    rows = pd.read_parquet(a.rows)
    for e in a.extra:
        rows = rows.merge(pd.read_parquet(e).drop(columns=["week_cutoff"]), on="match_id", how="inner")
    rows["week_cutoff"] = pd.to_datetime(rows.week_cutoff)
    rows = rows[(rows.week_cutoff >= pd.Timestamp(a.warmup)) & (rows.week_cutoff < pd.Timestamp(a.before))].reset_index(drop=True)
    cols = a.cols.split(",")
    X = design(rows, cols)
    y = rows.y.to_numpy(float)
    w = fit_logistic(X, y)
    print(json.dumps({c: float(v) for c, v in zip(cols, w)}, indent=1))
    print(f"n={len(rows)} in-sample log loss {log_loss(expit(X @ w), y).mean():.4f}")


if __name__ == "__main__":
    main()
