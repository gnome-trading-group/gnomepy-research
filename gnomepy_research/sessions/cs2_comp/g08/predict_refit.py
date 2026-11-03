"""g08 refit variant: the scored model with its fixed constants re-selected on all data before 2026-08-01.

    python -m gnomepy_research.sessions.cs2_comp.g08.predict_refit --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-08-03,2026-08-10]

Same structure as `predict`. REFIT_CONFIG and REFIT_WINDOW_DAYS were chosen by `refit_tune` (log loss of the weekly
layer over HLTV bo3 series from 2025-09-01 to the 2026-08-01 cut, every table read before 2026-08-01) and are fixed
numbers, so a bundle reaching further than that cut only enters through the weekly state readout and layer refit."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g08.model import CONFIG, state_prices
from gnomepy_research.sessions.cs2_comp.g08.predict import read_targets

REFIT_CONFIG = dataclasses.replace(CONFIG, drift_per_day=0.00005, market_scale=3.0, form_half_life_maps=25.0)
REFIT_WINDOW_DAYS = 365


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    s = state_prices(data_dir, targets, verbose=verbose, config=REFIT_CONFIG, window_days=REFIT_WINDOW_DAYS)
    return pd.DataFrame({"match_id": s.match_id.to_numpy(), "q": s.q_state.to_numpy(), "fit_cutoff": pd.to_datetime(s.week_cutoff).to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    a = ap.parse_args()
    out = predict(a.data, read_targets(a.targets, a.weeks), verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
