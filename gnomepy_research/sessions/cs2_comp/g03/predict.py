"""Predict P(team A wins the series) for every target, refitting once per week_cutoff on data before it."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g03 import features, model


def run(data_dir: str, targets_path: str, weeks: list[str] | None = None) -> pd.DataFrame:
    targets = pd.read_parquet(targets_path)
    targets["week_cutoff"] = pd.to_datetime(targets.week_cutoff)
    if weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(weeks))]
    cutoff = targets.week_cutoff.max()
    feats = features.build(data_dir, targets, cutoff)
    pred = model.predict_targets(feats)
    pred["q"] = np.clip(pred.q, 0.01, 0.99)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", nargs="*")
    a = ap.parse_args()
    run(a.data, a.targets, a.weeks).to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
