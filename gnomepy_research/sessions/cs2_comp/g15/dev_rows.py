"""
Walk-forward feature rows for development: every HLTV bo3 series between two weekly cutoffs, priced at its own week's
cutoff, with its outcome and (where a market existed) its kickoff price.

    python -m gnomepy_research.sessions.cs2_comp.g15.dev_rows --data <bundle> --out <parquet> --first 2024-07-01 --last 2026-07-27
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g15.data import load_market, load_series
from gnomepy_research.sessions.cs2_comp.g15.features import build_rows, history_jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    a = ap.parse_args()
    first, last = pd.Timestamp(a.first), pd.Timestamp(a.last)
    until = last + pd.Timedelta(days=7)
    t0 = time.time()
    series = load_series(a.data, until)
    jobs = history_jobs(series, first, last)
    market = load_market(a.data, until)
    jobs = jobs.merge(market[["match_id", "m0", "n_prints", "p_first", "p_24h", "stale_min", "span_hours", "p_min", "p_max"]],
                      on="match_id", how="left")
    print(f"{len(jobs)} series, {jobs.m0.notna().sum()} with a kickoff price, loaded in {time.time() - t0:.1f}s", flush=True)
    rows = build_rows(a.data, jobs, verbose=True)
    rows = rows.merge(jobs[["match_id", "n_prints", "p_first", "p_24h", "stale_min", "span_hours", "p_min", "p_max"]], on="match_id", how="left")
    rows.to_parquet(a.out, index=False)
    print(f"wrote {len(rows)} rows in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
