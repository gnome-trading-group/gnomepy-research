"""
g14 predictions: P(team A wins the bo3) for every target series, from data known before its week_cutoff, its
announced fives and its own price path up to kickoff.

    python -m gnomepy_research.sessions.cs2_comp.g14.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--no-faceit] [--rows <dir>] [--model-only]

The rating pass is a forward walk in known order, so a week's prices depend only on rows known before its cutoff;
the layers are refitted at each cutoff on the series known before it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.layer import LayerConfig
from gnomepy_research.sessions.cs2_comp.g14.pipeline import run
from gnomepy_research.sessions.cs2_comp.g14.ratings import RatingConfig

RATING = RatingConfig()
LAYER = LayerConfig()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--model-only", action="store_true", help="write the model's own price, not the market-adjusted one")
    ap.add_argument("--rows", help="directory to also save the series and target rows in")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    wc = pd.to_datetime(targets.week_cutoff)
    targets["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    rcfg = RatingConfig(**{**RATING.__dict__, "faceit": a.faceit})
    res = run(a.data, targets, rcfg, LAYER, verbose=True)
    if a.rows:
        Path(a.rows).mkdir(parents=True, exist_ok=True)
        res.series_rows.to_parquet(Path(a.rows) / "series_rows.parquet", index=False)
        res.target_rows.to_parquet(Path(a.rows) / "target_rows.parquet", index=False)
        res.predictions.to_parquet(Path(a.rows) / "predictions_full.parquet", index=False)
    p = res.predictions
    out = pd.DataFrame({"match_id": p.match_id.to_numpy(), "q": (p.q_model if a.model_only else p.q).to_numpy(),
                        "fit_cutoff": pd.to_datetime(p.fit_cutoff).to_numpy()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
