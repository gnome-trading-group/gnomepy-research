"""
e12 prices (HLTV-only and with FACEIT) and context for every bo3 series with a Polymarket path, each from the fits at
its own week_cutoff: the rows the recalibration layer is developed on.

    python -m gnomepy_research.sessions.cs2_comp.g06.e12_hist --data <bundle> --out <parquet>
"""
from __future__ import annotations

import argparse

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g06.history import market_series_targets
from gnomepy_research.sessions.cs2_comp.g06.predict import e12_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    targets = market_series_targets(a.data, pd.Timestamp("2100-01-01"))
    e12_rows(a.data, targets).to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
