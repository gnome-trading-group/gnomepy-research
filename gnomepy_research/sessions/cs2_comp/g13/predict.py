"""
g13 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g13.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--rows <parquet>]

Per distinct week_cutoff: e12's two skill fits (HLTV, HLTV + FACEIT) price each target with its announced five, and
`q` is e12's own price. Everything else this entry built (a smoothed kickoff price, thin-market gates, a second
skill filter, form fits, map-pool and head-to-head structure) measured as noise next to the market and e12 in the
scorer's own pool, so none of it is carried into `q`; `experiments/` holds the evidence.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g13.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.g13.rows import skill_rows, target_frame


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictions for the targets, and the feature row each one was priced from."""
    frame = target_frame(targets)
    inputs = Inputs(data_dir, frame.week_cutoff.max(), faceit=True)
    rows = skill_rows(inputs, frame, verbose)
    out = pd.DataFrame({"match_id": rows.match_id.to_numpy(), "q": expit(rows.z_e12.to_numpy()),
                        "fit_cutoff": pd.to_datetime(rows.week_cutoff).to_numpy()})
    return out, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--rows", help="parquet to also save the targets' feature rows in")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = _naive(targets.week_cutoff)
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out, rows = predict(a.data, targets, verbose=True)
    # On Batch the output directory is only known through the environment, so paths may name it as $OUTPUT_DIR.
    out_path = Path(os.path.expandvars(a.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    if a.rows:
        rows.to_parquet(Path(os.path.expandvars(a.rows)), index=False)
    print(f"wrote {len(out)} predictions to {out_path}")


if __name__ == "__main__":
    main()
