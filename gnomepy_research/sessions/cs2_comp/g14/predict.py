"""
g14 predictions: P(team A wins the bo3) for every target series, from data known before its week_cutoff, its
announced fives and its own price path up to kickoff.

    python -m gnomepy_research.sessions.cs2_comp.g14.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--no-faceit] [--rows <dir>] [--train-rows <walk parquet>] [--fit '{...}']

Per distinct week_cutoff: the rating pass walks the maps known before it (online features: form, roster churn, map
pool, idle time), the batch skill model is fitted on the window before it, and each target is priced with its
announced fives. The layer weights below were fitted on walk-forward rows from monthly fits before the first dev
cutoff (walk.py), so they are point-in-time for every target; --train-rows refits them instead, for development.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.layer import LayerConfig
from gnomepy_research.sessions.cs2_comp.g14.pipeline import predict_weeks, run
from gnomepy_research.sessions.cs2_comp.g14.ratings import RatingConfig
from gnomepy_research.sessions.cs2_comp.g14.skillfit import FitConfig

RATING = RatingConfig()
FIT = FitConfig()
LAYER = LayerConfig()
WEIGHTS: dict[str, float] | None = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--model-only", action="store_true", help="write the model's own price, not the market-adjusted one")
    ap.add_argument("--rows", help="directory to also save the series and target rows in")
    ap.add_argument("--train-rows", help="walk-forward rows to fit the layers on instead of the frozen weights")
    ap.add_argument("--fit", default="{}", help="FitConfig overrides as JSON")
    ap.add_argument("--no-market-layer", dest="market_layer", action="store_false")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    wc = pd.to_datetime(targets.week_cutoff)
    targets["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    rcfg = RatingConfig(**{**RATING.__dict__, "faceit": a.faceit})
    fcfg = FitConfig(**{**FIT.__dict__, **json.loads(a.fit)})
    if a.train_rows:
        res = run(a.data, targets, rcfg, LAYER, verbose=True, fcfg=fcfg, train_rows=pd.read_parquet(a.train_rows))
    else:
        res = predict_weeks(a.data, targets, rcfg, fcfg, LAYER, WEIGHTS, verbose=True, market_layer=a.market_layer)
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
