"""
g06 variant: e12 with the player-skill prior sd widened from 0.5 to PRIOR_SD in both skill fits (HLTV, and
HLTV+FACEIT); its fixed layer is unchanged. PRIOR_SD was chosen from {0.5, 1, 2, 3} on 2025-01-06 .. 2026-03-01 HLTV bo3
outcomes, each series priced at its own week_cutoff, by quarter-held-out log loss of a logistic recalibration.

    python -m gnomepy_research.sessions.cs2_comp.g06.predict_wide --data <bundle> --targets <parquet> --out <parquet>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g06.history import naive
from gnomepy_research.sessions.cs2_comp.g06.wide import e12_wide

PRIOR_SD = 2.0


def read_targets(path: str, weeks: str | None) -> pd.DataFrame:
    t = pd.read_parquet(path)
    t["week_cutoff"] = naive(t.week_cutoff)
    if weeks:
        t = t[t.week_cutoff.isin(pd.to_datetime(weeks.split(",")))]
    return t.reset_index(drop=True)


def arguments(doc: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    return ap.parse_args()


def write(out: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"wrote {len(out)} predictions to {path}")


def main() -> None:
    a = arguments(__doc__)
    targets = read_targets(a.targets, a.weeks)
    with threadpool_limits(1):
        e = e12_wide(a.data, targets, PRIOR_SD)
    write(pd.DataFrame({"match_id": e.match_id, "q": e.e12, "fit_cutoff": e.week_cutoff}), a.out)


if __name__ == "__main__":
    main()
