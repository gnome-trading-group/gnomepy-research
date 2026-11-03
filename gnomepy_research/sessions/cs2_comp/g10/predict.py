"""
g10 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g10.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13]

The market's kickoff price is the starting point; a residual layer (layer.py), refitted every week on all past
market-joined bo3 series, corrects it with the recent price path and the e12 skill price. See rows.py for how each
past series is priced point-in-time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g10.layer import Layer, Spec
from gnomepy_research.sessions.cs2_comp.g10.rows import SETTLE, build

SPEC = Spec()


def predict(data_dir: str | Path, targets: pd.DataFrame, spec: Spec = SPEC, verbose: bool = False) -> pd.DataFrame:
    rows = build(data_dir, targets, verbose=verbose)
    hist = rows[~rows.is_target]
    out = []
    for wk in sorted(rows.loc[rows.is_target, "week_cutoff"].unique()):
        wk = pd.Timestamp(wk)
        train = hist[pd.to_datetime(hist.time) < wk - SETTLE]
        layer = Layer(spec).fit(train, wk)
        tg = rows[rows.is_target & (rows.week_cutoff == wk)]
        out.append(pd.DataFrame({"match_id": tg.match_id.to_numpy(), "q": layer.predict(tg), "fit_cutoff": wk}))
        if verbose:
            print(f"{wk:%Y-%m-%d} train {len(train)} targets {len(tg)} beta {dict(zip(layer.names, np.round(layer.beta, 3)))}",
                  flush=True)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    # Batch containers see every host core but get a share of them; one BLAS thread per process avoids thrashing.
    threadpool_limits(1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    wc = pd.to_datetime(targets.week_cutoff)
    targets["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
