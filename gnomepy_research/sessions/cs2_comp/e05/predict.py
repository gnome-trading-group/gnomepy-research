"""Predict contract: one fit per distinct week_cutoff, using only bundle rows dated before it.

The model is a sequential filter, so "fitting" a week means advancing the filter over every HLTV and FACEIT map from
before that Monday; the weeks of a run share one pass and any single week reproduces on its own."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e05.pricing import PriceConfig, SeriesPricer
from gnomepy_research.sessions.cs2_comp.e05.params import final_config, final_price
from gnomepy_research.sessions.cs2_comp.e05.stream import Config
from gnomepy_research.sessions.cs2_comp.e05.world import load_world


def price_week(pricer: SeriesPricer, lineups: dict, week: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(week))
    for k, r in enumerate(week.itertuples()):
        la, lb = lineups.get(int(r.team_a_id), []), lineups.get(int(r.team_b_id), [])
        out[k] = pricer.price(int(r.team_a_id), la, int(r.team_b_id), lb)
    return out


def predict(data_dir: str | Path, targets: pd.DataFrame, cfg: Config, price: PriceConfig) -> pd.DataFrame:
    targets = targets.assign(week_cutoff=pd.to_datetime(targets.week_cutoff))
    if targets.week_cutoff.dt.tz is not None:
        targets = targets.assign(week_cutoff=targets.week_cutoff.dt.tz_convert(None))
    world = load_world(data_dir, targets.week_cutoff.max(), cfg.faceit)
    flt = world.new_filter(cfg)
    parts = []
    for cutoff, week in targets.groupby("week_cutoff"):
        flt.advance(cutoff)
        lineups = world.lineups(cutoff, set(week.team_a_id.astype(int)) | set(week.team_b_id.astype(int)), price)
        pricer = SeriesPricer(flt, world.rankings, cutoff, price)
        parts.append(pd.DataFrame({"match_id": week.match_id.to_numpy(), "q": price_week(pricer, lineups, week),
                                   "fit_cutoff": cutoff}))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", action="store_true")
    ap.add_argument("--weeks", default="", help="comma-separated week_cutoff dates to predict; default all")
    args = ap.parse_args()
    targets = pd.read_parquet(args.targets)
    if args.weeks:
        keep = {pd.Timestamp(w) for w in args.weeks.split(",")}
        targets = targets[pd.to_datetime(targets.week_cutoff).isin(keep)]
    out = predict(args.data, targets, final_config(not args.no_faceit), final_price(not args.no_faceit))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
