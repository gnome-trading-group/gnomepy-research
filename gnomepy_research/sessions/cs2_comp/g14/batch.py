"""Batch task: g14 predictions (and the feature rows) for some or all weeks, written to $OUTPUT_DIR.

    python -m gnomepy_research.sessions.cs2_comp.g14.batch [--weeks ...] [--no-faceit] [--data /work/in/comp] [--tag name]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.layer import LayerConfig
from gnomepy_research.sessions.cs2_comp.g14.pipeline import run
from gnomepy_research.sessions.cs2_comp.g14.predict import LAYER, RATING
from gnomepy_research.sessions.cs2_comp.g14.ratings import RatingConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    targets = pd.read_parquet(Path(a.data) / "targets_tune.parquet")
    wc = pd.to_datetime(targets.week_cutoff)
    targets["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    rcfg = RatingConfig(**{**RATING.__dict__, "faceit": a.faceit})
    res = run(a.data, targets, rcfg, LayerConfig(**LAYER.__dict__), verbose=True)
    out = Path(os.environ.get("OUTPUT_DIR", "."))
    out.mkdir(parents=True, exist_ok=True)
    res.series_rows.to_parquet(out / f"series_rows_{a.tag}.parquet", index=False)
    res.target_rows.to_parquet(out / f"target_rows_{a.tag}.parquet", index=False)
    res.predictions.to_parquet(out / f"predictions_{a.tag}.parquet", index=False)
    print(f"wrote {len(res.predictions)} predictions")


if __name__ == "__main__":
    main()
