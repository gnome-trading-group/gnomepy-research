"""
Pre-match bo3 series prices for the competition targets, one fit per distinct week_cutoff on data before it.

    python -m gnomepy_research.sessions.cs2_comp.e16.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--no-faceit] [--weeks 2026-04-06,2026-04-13] [--set name=value ...]

A target's lineups are not known before its cutoff, so each team is priced with its five from its latest map in the
window: HLTV maps only without FACEIT, HLTV and FACEIT maps with it (a FACEIT qualifier is often the most recent
sighting of an online team). --set overrides a Config field, for tuning on the dev period only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e16.data import Bundle, read_bundle, team_lineups
from gnomepy_research.sessions.cs2_comp.e16.model import Config, fit, rank_feature


def window(df: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
    return df[(df.cut < cutoff) & (df.time >= cutoff - pd.Timedelta(days=days))]


def predict_week(b: Bundle, targets: pd.DataFrame, cutoff: pd.Timestamp, cfg: Config) -> tuple[pd.DataFrame, dict]:
    maps = window(b.hltv_maps, cutoff, cfg.window_days)
    ratings = window(b.hltv_ratings, cutoff, cfg.window_days).assign(source="hltv")
    levels = None
    if b.faceit_maps is not None:
        maps = pd.concat([maps, window(b.faceit_maps, cutoff, cfg.faceit_window_days)], ignore_index=True)
        ratings = pd.concat([ratings, window(b.faceit_ratings, cutoff, cfg.faceit_window_days).assign(source="faceit")],
                            ignore_index=True)
        levels = b.faceit_levels[b.faceit_levels.cut < cutoff]
    rank_of = rank_feature(b.rankings, cutoff)
    f = fit(maps, ratings, levels, rank_of, cutoff, cfg)
    fives = team_lineups(maps)
    out = []
    for r in targets.itertuples():
        ka, kb = f"h{int(r.team_a_id)}", f"h{int(r.team_b_id)}"
        mean, var = f.gap(ka, fives.get(ka, (None,))[0], rank_of.get(ka, 0.0),
                          kb, fives.get(kb, (None,))[0], rank_of.get(kb, 0.0))
        out.append({"match_id": r.match_id, "q": f.price(mean, var), "gap": mean, "gap_var": var,
                    "fit_cutoff": cutoff})
    return pd.DataFrame(out), f.info


def run(data: Path, targets: pd.DataFrame, faceit: bool, cfg: Config, verbose: bool = True) -> pd.DataFrame:
    cutoffs = sorted(pd.to_datetime(targets.week_cutoff).unique())
    b = read_bundle(data, max(cutoffs), faceit)
    parts = []
    for c in cutoffs:
        t0 = time.time()
        c = pd.Timestamp(c)
        week = targets[pd.to_datetime(targets.week_cutoff) == c]
        p, info = predict_week(b, week, c, cfg)
        parts.append(p)
        if verbose:
            print(f"{c:%Y-%m-%d} {len(p)} series {time.time() - t0:.1f}s {json.dumps(info)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def parse_overrides(items: list[str]) -> dict:
    fields = {f.name: f.type for f in dataclasses.fields(Config)}
    over = {}
    for kv in items:
        k, v = kv.split("=", 1)
        over[k] = int(v) if fields[k] in (int, "int") else float(v)
    return over


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--set", action="append", default=[], help="Config override name=value (dev tuning)")
    args = ap.parse_args()
    cfg = dataclasses.replace(Config(), **parse_overrides(args.set))
    targets = pd.read_parquet(args.targets)
    if args.weeks:
        keep = pd.to_datetime(args.weeks.split(","))
        targets = targets[pd.to_datetime(targets.week_cutoff).isin(keep)]
    pred = run(args.data, targets, args.faceit, cfg)
    pred["q"] = np.clip(pred.q, 1e-4, 1 - 1e-4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(args.out, index=False)
    print(f"wrote {len(pred)} predictions to {args.out}")


if __name__ == "__main__":
    main()
