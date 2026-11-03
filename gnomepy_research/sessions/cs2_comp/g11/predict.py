"""
g11 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g11.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--rows <dir>]

Per target week: the graph fit (graph.py) prices the target series, and every market series of the preceding
HISTORY_WEEKS weeks at its own week cutoff; the market layer (layer.py) is refitted on those that finished before the
cutoff and combines the target's kickoff price with its graph price.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g11.graph import Config
from gnomepy_research.sessions.cs2_comp.g11.layer import features, fit_logistic
from gnomepy_research.sessions.cs2_comp.g11.pipeline import WEEK, build

CONFIG = Config()
SHORT = None
FACEIT = False
HISTORY_WEEKS = 26
COLUMNS = ["zm", "zr", "zr_nomkt", "one"]
L2 = 1.0


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False, rows_dir: str | Path | None = None) -> pd.DataFrame:
    train, tgt = build(str(data_dir), targets, CONFIG, FACEIT, HISTORY_WEEKS, SHORT, verbose=verbose)
    if rows_dir is not None:
        Path(rows_dir).mkdir(parents=True, exist_ok=True)
        train.to_parquet(Path(rows_dir) / "train.parquet", index=False)
        tgt.to_parquet(Path(rows_dir) / "targets.parquet", index=False)
    Ftr = features(train)[COLUMNS].to_numpy()
    Ftg = features(tgt)[COLUMNS].to_numpy()
    y = train.y.to_numpy(float)
    q = np.full(len(tgt), np.nan)
    for c in sorted(tgt.week_cutoff.unique()):
        c = pd.Timestamp(c)
        use = ((train.known < c) & (train.kickoff >= c - HISTORY_WEEKS * WEEK) & train.m0.notna()).to_numpy()
        w = fit_logistic(Ftr[use], y[use], L2)
        sel = (tgt.week_cutoff == c).to_numpy()
        q[sel] = 1 / (1 + np.exp(-Ftg[sel] @ w))
        if verbose:
            print(f"{c:%Y-%m-%d} layer on {int(use.sum())} series: " + " ".join(f"{k}={v:+.3f}" for k, v in zip(COLUMNS, w)), flush=True)
    return pd.DataFrame({"match_id": tgt.match_id.to_numpy(), "q": q, "fit_cutoff": pd.to_datetime(tgt.week_cutoff).to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--rows", help="directory to also save the feature rows in")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    wc = pd.to_datetime(targets.week_cutoff)
    targets["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, verbose=True, rows_dir=a.rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
