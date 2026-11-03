"""
g15 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g15.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13]

Per distinct week_cutoff, two batch skill fits on the maps of the trailing 900 days (half-life 900 days):

- HLTV maps, with the per-player map-rating channel and per-player map deviations; the announced fives are priced on
  each map of the active pool and the bo3 goes through the veto (`skillfit.veto_series`);
- HLTV plus FACEIT maps, priced as one bo3 from the fives' mean skill.

The fixed logistic layer below blends the two bo3 logits; its weights were fitted by `dev_layer.py` on the Sep 2024 -
Mar 2026 weekly walk-forward, all before the first tune cutoff, so they are point-in-time for every target.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g15.features import target_jobs
from gnomepy_research.sessions.cs2_comp.g15.fit_rows import fit_rows
from gnomepy_research.sessions.cs2_comp.g15.layer import z_series
from gnomepy_research.sessions.cs2_comp.g15.skillfit import FitConfig

LONG = dict(window_days=900, half_life_days=900)
FITS = {
    "g": (FitConfig(sigma=0.7, rating_gamma=0.3, rating_sd=0.45, map_sigma=0.35, **LONG), False),
    "f": (FitConfig(sigma=0.7, faceit_weight=0.5, **LONG), True),
}
LAYER = {"one": 0.020948, "z_g_veto": 0.402839, "z_f": 0.231434}


def z_columns(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every layer column: z_<fit> is the bo3 logit of the fit's gap; z_<fit>_veto the veto-priced logit, falling back
    to z_<fit> for a side without a five."""
    out = {"one": np.ones(len(rows))}
    for name in FITS:
        out[f"z_{name}"] = z_series(rows[f"{name}_gap"])
        col = f"{name}_veto_logit"
        if col in rows:
            v = rows[col].to_numpy(float)
            out[f"z_{name}_veto"] = np.where(np.isnan(v), out[f"z_{name}"], v)
    return out


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False, rows_dir: str | Path | None = None) -> pd.DataFrame:
    jobs = target_jobs(targets)
    rows = jobs[["match_id", "week_cutoff"]].copy()
    for name, (cfg, faceit) in FITS.items():
        r = fit_rows(data_dir, jobs, cfg, faceit, prefix=name, verbose=verbose)
        rows = rows.merge(r.drop(columns=["week_cutoff"]), on="match_id", how="left")
    if rows_dir is not None:
        Path(rows_dir).mkdir(parents=True, exist_ok=True)
        rows.to_parquet(Path(rows_dir) / "rows.parquet", index=False)
    cols = z_columns(rows)
    z = sum(weight * cols[col] for col, weight in LAYER.items())
    return pd.DataFrame({"match_id": rows.match_id.to_numpy(), "q": expit(z), "fit_cutoff": pd.to_datetime(rows.week_cutoff).to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--rows", help="directory to also save the per-series fit rows in")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    if a.weeks:
        wc = pd.to_datetime(targets.week_cutoff)
        wc = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
        targets = targets[wc.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, verbose=True, rows_dir=a.rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
