"""
Walk-forward training rows for the frozen layer: at each cutoff, fit the batch skill model on the data known before
it and price the series that start before the next cutoff with their announced fives, next to the online features
the rating pass had when each series started.

    python -m gnomepy_research.sessions.cs2_comp.g14.walk --cutoffs 2025-04-07,2025-05-05,... --out <parquet>
        [--data <bundle>] [--fit '{"tau": 0.1}'] [--no-faceit]

The last cutoff only bounds the rows of the one before it. On Batch, --out is relative to $OUTPUT_DIR.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.data import load_inputs
from gnomepy_research.sessions.cs2_comp.g14.pipeline import add_fit_features, online_pass, prepare_targets
from gnomepy_research.sessions.cs2_comp.g14.ratings import RatingConfig
from gnomepy_research.sessions.cs2_comp.g14.skillfit import FitConfig, fit

TARGET_COLUMNS = ["match_id", "match_time", "match_date", "team_a_id", "team_a_name", "team_b_id", "team_b_name", "event_name",
                  "event_tier", "is_lan", "bo_type", "week_cutoff", "team_a_lineup_ids", "team_b_lineup_ids",
                  "team_a_standin_ids", "team_b_standin_ids"]


def walk(data_dir: str | Path, cutoffs: list[pd.Timestamp], rcfg: RatingConfig, fcfg: FitConfig, verbose: bool = True) -> pd.DataFrame:
    cutoffs = sorted(pd.Timestamp(c) for c in cutoffs)
    t0 = time.time()
    inputs = load_inputs(data_dir, cutoffs[-1], rcfg.faceit)
    if verbose:
        print(f"loaded {len(inputs.maps)} maps in {time.time() - t0:.0f}s", flush=True)
    empty = prepare_targets(pd.DataFrame({c: pd.Series(dtype=object) for c in TARGET_COLUMNS}))
    sr, _, _ = online_pass(data_dir, inputs, empty, cutoffs[:-1], rcfg, verbose=False)
    sr["time"] = pd.to_datetime(sr.time)
    parts = []
    for c, nxt in zip(cutoffs[:-1], cutoffs[1:]):
        t0 = time.time()
        sf = fit(inputs, c, fcfg)
        rows = sr[(sr.time >= c) & (sr.time < nxt)]
        fives = [(tuple(a), tuple(b)) for a, b in zip(rows.five_a, rows.five_b)]
        part = add_fit_features(rows, sf, fives)
        part["fit_cutoff"] = c
        parts.append(part)
        if verbose:
            print(f"{c:%Y-%m-%d} fit {sf.info} rows {len(part)} {time.time() - t0:.0f}s", flush=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--fit", default="{}")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    a = ap.parse_args()
    out = Path(os.environ.get("OUTPUT_DIR", ".")) / a.out if not Path(a.out).is_absolute() else Path(a.out)
    rows = walk(a.data, [pd.Timestamp(c) for c in a.cutoffs.split(",")], RatingConfig(faceit=a.faceit), FitConfig(**json.loads(a.fit)))
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
