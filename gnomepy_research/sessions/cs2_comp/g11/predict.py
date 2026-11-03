"""
g11 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g11.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--rows <dir>]

Per target week, one graph fit (graph.py) at the week cutoff prices each target with its announced fives: the gap
in mean player skill (with community effects and FACEIT levels in the prior), shifted by each team's pool-average map
offset, turned into a bo3 price.

q is the model's own price, not blended with the market here. A walk-forward blend with the kickoff price, refitted
weekly on the finished market series, scored worse under the scorer's pool (which refits the blend itself): its
weights move from week to week, which a single pooled refit cannot undo.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g11.graph import Config
from gnomepy_research.sessions.cs2_comp.g11.pipeline import build

CONFIG = Config()
FACEIT = True


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False, rows_dir: str | Path | None = None) -> pd.DataFrame:
    _, tgt = build(str(data_dir), targets, CONFIG, FACEIT, 0, verbose=verbose)
    if rows_dir is not None:
        Path(rows_dir).mkdir(parents=True, exist_ok=True)
        tgt.to_parquet(Path(rows_dir) / "targets.parquet", index=False)
    return pd.DataFrame({"match_id": tgt.match_id.to_numpy(), "q": tgt.q.to_numpy(float),
                         "fit_cutoff": pd.to_datetime(tgt.week_cutoff).to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", help="output parquet (default: predictions.parquet in $OUTPUT_DIR, for Batch)")
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--rows", help="directory to also save the feature rows in")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    wc = pd.to_datetime(targets.week_cutoff)
    targets["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out_path = Path(a.out) if a.out else Path(os.environ["OUTPUT_DIR"]) / "predictions.parquet"
    rows = a.rows if a.rows or a.out else out_path.parent / "rows"
    out = predict(a.data, targets, verbose=True, rows_dir=rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out)} predictions to {out_path}")


if __name__ == "__main__":
    main()
