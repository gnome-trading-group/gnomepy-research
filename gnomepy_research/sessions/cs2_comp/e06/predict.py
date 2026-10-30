"""Predict contract: one pass of the Kalman filters over every event before each target week's cutoff.

    python -m gnomepy_research.sessions.cs2_comp.e06.predict --data <bundle> --targets <parquet> --out <parquet> [--no-faceit]
        [--weeks 2026-03-30,2026-04-06]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e06.run import load_inputs, run_pass
from gnomepy_research.sessions.cs2_comp.e06.skill import Params

logger = logging.getLogger(__name__)


def queries_from_targets(targets: pd.DataFrame) -> pd.DataFrame:
    kick = pd.to_datetime(targets.match_time)
    kick = kick.dt.tz_convert(None) if kick.dt.tz is not None else kick
    return pd.DataFrame({
        "match_id": targets.match_id.to_numpy(),
        "week_cutoff": pd.to_datetime(targets.week_cutoff).to_numpy(),
        "kickoff": kick.astype("datetime64[ns]").astype("int64").to_numpy() / 1e9,
        "team_a_id": targets.team_a_id.to_numpy(),
        "team_b_id": targets.team_b_id.to_numpy(),
    })


def predict(data_dir: Path, targets: pd.DataFrame, use_faceit: bool, params: Params | None = None) -> pd.DataFrame:
    events, rankings, moves = load_inputs(data_dir, use_faceit)
    res = run_pass(events, rankings, moves, params or Params(), queries_from_targets(targets), use_faceit=use_faceit)
    return res[["match_id", "q", "fit_cutoff"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-faceit", action="store_true")
    ap.add_argument("--weeks", default=None, help="comma-separated week_cutoff dates to keep")
    args = ap.parse_args()
    targets = pd.read_parquet(args.targets)
    if args.weeks:
        keep = pd.to_datetime(args.weeks.split(","))
        targets = targets[pd.to_datetime(targets.week_cutoff).isin(keep)]
    out = predict(args.data, targets, not args.no_faceit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    logger.info("wrote %d predictions to %s", len(out), args.out)


if __name__ == "__main__":
    main()
