"""
g15 refreshed variant for the confirmation months: the model of `predict.py` with its frozen constants re-estimated
on every week before 2026-08-01.

    python -m gnomepy_research.sessions.cs2_comp.g15.predict_refit --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-08-03,2026-08-10]

What is frozen in `predict.py` is the blend layer, fitted by `dev_layer.py` on the Sep 2024 - Mar 2026 walk-forward
rows. The layer below is the same fit on the same rows extended through the week of 2026-07-27
(`dev_layer.py ... --before 2026-08-01`, 16,483 series instead of 13,335). The skill-fit settings are unchanged: they
were selected by pool-minus-market on every market-joined series before 2026-08-01, so the same procedure on the same
window returns the same settings. Every weekly fit still learns its priors from the data before its own cutoff.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g15.features import target_jobs
from gnomepy_research.sessions.cs2_comp.g15.fit_rows import fit_rows
from gnomepy_research.sessions.cs2_comp.g15.predict import FITS, z_columns

LAYER_REFIT = {"one": 0.018622, "z_g_veto": 0.395969, "z_f": 0.238051}


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    jobs = target_jobs(targets)
    rows = jobs[["match_id", "week_cutoff"]].copy()
    for name, (cfg, faceit) in FITS.items():
        r = fit_rows(data_dir, jobs, cfg, faceit, prefix=name, verbose=verbose)
        rows = rows.merge(r.drop(columns=["week_cutoff"]), on="match_id", how="left")
    cols = z_columns(rows)
    z = sum(weight * cols[col] for col, weight in LAYER_REFIT.items())
    return pd.DataFrame({"match_id": rows.match_id.to_numpy(), "q": expit(z), "fit_cutoff": pd.to_datetime(rows.week_cutoff).to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    if a.weeks:
        wc = pd.to_datetime(targets.week_cutoff)
        wc = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
        targets = targets[wc.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
