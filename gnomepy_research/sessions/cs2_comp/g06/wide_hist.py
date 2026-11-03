"""
Pre-April selection rows for the widened e12: every HLTV bo3 series of 2025-01-06 .. 2026-03-01 priced at its own
week_cutoff with prior sd --sd. --part k --parts n takes every n-th week, so a sweep splits across Batch tasks.

    python -m gnomepy_research.sessions.cs2_comp.g06.wide_hist --data <bundle> --sd 2.0 --part 0 --parts 5 --out <dir>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g06.history import hltv_series, standins, week_start
from gnomepy_research.sessions.cs2_comp.g06.wide import e12_wide

CUT = pd.Timestamp("2026-03-02")
START = pd.Timestamp("2025-01-06")


def selection_series(data_dir: str | Path) -> pd.DataFrame:
    s = hltv_series(data_dir, CUT)
    s = s[s.time >= START].copy()
    s["week_cutoff"] = week_start(s.time)
    return standins(data_dir, s, CUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sd", type=float, required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--newcomer", action="store_true")
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "."))
    a = ap.parse_args()
    s = selection_series(a.data)
    weeks = sorted(s.week_cutoff.unique())[a.part::a.parts]
    s = s[s.week_cutoff.isin(weeks)]
    with threadpool_limits(1):
        e = e12_wide(a.data, s, a.sd, a.newcomer)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    name = f"wide_sd{a.sd}{'_new' if a.newcomer else ''}_p{a.part}.parquet"
    e.merge(s[["match_id", "y"]], on="match_id").to_parquet(Path(a.out) / name, index=False)


if __name__ == "__main__":
    main()
