"""
g13 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g13.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--rows <parquet>]

Per distinct week_cutoff: e12's two skill fits (HLTV, HLTV + FACEIT) price each target with its announced five, and
a ridge logistic layer (`layer.py`) combines that price with the smoothed kickoff price of the target's own market.
The layer is refit at every cutoff on every priced bo3 series known before it, each of those priced by the fits at
its own week's cutoff, so a run for one week gives the same answer as a run for many.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths
from gnomepy_research.sessions.cs2_comp.g13.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.g13.layer import Layer, LayerSpec
from gnomepy_research.sessions.cs2_comp.g13.rows import WEEK, history_frame, market_cutoffs, skill_rows, target_frame, with_paths

SPEC = LayerSpec()


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictions for the targets, and every feature row the run built (history rows flagged is_target=False)."""
    weeks = sorted(pd.Timestamp(c) for c in targets.week_cutoff.unique())
    inputs = Inputs(data_dir, weeks[-1], faceit=True)
    cutoffs = market_cutoffs(data_dir, weeks[-1])
    hist = history_frame(inputs, cutoffs).assign(is_target=False)
    tgt = target_frame(targets).assign(is_target=True, y=np.nan, known=pd.NaT)
    frame = pd.concat([hist, tgt], ignore_index=True)
    rows = skill_rows(inputs, frame, verbose)
    paths = load(data_dir, "market/series_paths", weeks[-1])
    hist_rows = with_paths(rows[~rows.is_target], paths[paths.match_id.isin(rows.match_id)])
    tgt_rows = with_paths(rows[rows.is_target], target_paths(data_dir, rows.match_id[rows.is_target]))
    out = []
    for c in weeks:
        train = hist_rows[(hist_rows.week_cutoff < c) & (hist_rows.known < c)]
        layer = Layer.fit(train, SPEC)
        week = tgt_rows[tgt_rows.week_cutoff == c]
        q = layer.predict(week)
        out.append(pd.DataFrame({"match_id": week.match_id.to_numpy(), "q": q, "fit_cutoff": c}))
        if verbose:
            print(f"{c:%Y-%m-%d} layer rows {layer.n_train} weights {layer.info} targets {len(week)}", flush=True)
    return pd.concat(out, ignore_index=True), pd.concat([hist_rows, tgt_rows], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--rows", help="parquet to also save every feature row the run built in")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = _naive(targets.week_cutoff)
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out, rows = predict(a.data, targets, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    if a.rows:
        rows.to_parquet(a.rows, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
