"""
Series predictions from data dated before each target's week cutoff.

    python -m gnomepy_research.sessions.cs2_comp.e08.predict --data <bundle> --targets <parquet> --out <parquet> [--no-faceit] [--weeks N,...]

One causal pass over the maps prices every week's targets from the state at that week's Monday cutoff, so a bundle
cut at a cutoff and holding only that week's targets gives the same prices. With --no-faceit the FACEIT maps are
never built and the run is the HLTV-only filter.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e08.data import RankPoints, build_maps
from gnomepy_research.sessions.cs2_comp.e08.filter import Params
from gnomepy_research.sessions.cs2_comp.e08.series import Moves, SeriesParams
from gnomepy_research.sessions.cs2_comp.e08.walk import walk


def predict(data_dir: Path, targets: pd.DataFrame, use_faceit: bool) -> pd.DataFrame:
    targets = targets.assign(week_cutoff=pd.to_datetime(targets.week_cutoff))
    before = targets.week_cutoff.max()
    params = dataclasses.replace(Params(), use_faceit=use_faceit)
    maps = build_maps(data_dir, before, use_faceit=use_faceit)
    ranks = RankPoints(load(data_dir, "hltv/cs2_team_rankings", before))
    moves = Moves(load(data_dir, "hltv/cs2_roster_moves", before))
    out, _ = walk(maps, targets, moves, ranks, params, SeriesParams())
    return out[["match_id", "q", "fit_cutoff"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-faceit", action="store_true")
    ap.add_argument("--weeks", default=None, help="comma-separated week cutoffs to price, for splitting across tasks")
    args = ap.parse_args()
    targets = pd.read_parquet(args.targets)
    if args.weeks:
        keep = {pd.Timestamp(w) for w in args.weeks.split(",")}
        targets = targets[pd.to_datetime(targets.week_cutoff).isin(keep)]
    out = predict(args.data, targets, use_faceit=not args.no_faceit)
    dest = args.out if args.out.is_absolute() or "OUTPUT_DIR" not in os.environ else Path(os.environ["OUTPUT_DIR"]) / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"wrote {len(out)} predictions to {dest}", flush=True)


if __name__ == "__main__":
    main()
