"""
Series prices for the competition targets from the credibility rating (model.py), one fit per week_cutoff.

    python -m gnomepy_research.sessions.cs2_comp.e15.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--no-faceit] [--weeks 2026-04-06,2026-04-13]

Each week is fitted on the maps known before its Monday cutoff only, so a run on a bundle cut at that cutoff gives the
same prices. --no-faceit drops every FACEIT map, which also removes the ESEA division classes and FACEIT lineups.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e15.data import Raw, before, load_raw
from gnomepy_research.sessions.cs2_comp.e15.model import Config, fit, series_price


def career_first(raw: Raw) -> pd.Series:
    """Each player's first map day in the data; for any player inside a fit window this is a day before its cutoff."""
    days = raw.maps.day.to_numpy()
    rows = [(p, d) for col in ("five_a", "five_b") for five, d in zip(raw.maps[col], days) for p in five]
    s = pd.DataFrame(rows, columns=["p", "d"]).groupby("p").d.min()
    return s


def predict_week(raw: Raw, first: pd.Series, targets: pd.DataFrame, cutoff: pd.Timestamp, cfg: Config) -> pd.DataFrame:
    maps, rankings = before(raw, cutoff, cfg.window_days)
    f = fit(maps, rankings, cutoff, cfg, first[first < np.datetime64(pd.Timestamp(cutoff))], raw.ratings)
    q = [series_price(f, int(a), int(b)) for a, b in zip(targets.team_a_id, targets.team_b_id)]
    return pd.DataFrame({"match_id": targets.match_id.to_numpy(), "q": q, "fit_cutoff": pd.Timestamp(cutoff)})


def run(data: Path, targets_path: Path, use_faceit: bool, weeks: list[str] | None, cfg: Config) -> pd.DataFrame:
    targets = pd.read_parquet(targets_path)
    targets["week_cutoff"] = pd.to_datetime(targets.week_cutoff)
    cutoffs = sorted(targets.week_cutoff.unique())
    if weeks:
        cutoffs = [c for c in cutoffs if f"{pd.Timestamp(c):%Y-%m-%d}" in set(weeks)]
    raw = load_raw(data, use_faceit, ratings=cfg.rating_weight > 0, faceit_stats=cfg.faceit_stats)
    first = career_first(raw)
    out = []
    for c in cutoffs:
        t0 = time.time()
        out.append(predict_week(raw, first, targets[targets.week_cutoff == c], pd.Timestamp(c), cfg))
        print(f"{pd.Timestamp(c):%Y-%m-%d} {len(out[-1])} series {time.time() - t0:.1f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def output_path(path: Path) -> Path:
    """A relative output path lands in $OUTPUT_DIR on Batch, where the task's outputs are collected."""
    return path if path.is_absolute() else Path(os.environ.get("OUTPUT_DIR", ".")) / path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--weeks", help="comma-separated week cutoffs (YYYY-MM-DD) to price; default all")
    ap.add_argument("--config", help="JSON overrides of Config fields")
    a = ap.parse_args()
    cfg = dataclasses.replace(Config(), **json.loads(a.config)) if a.config else Config()
    preds = run(a.data, a.targets, a.faceit, a.weeks.split(",") if a.weeks else None, cfg)
    out = output_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
