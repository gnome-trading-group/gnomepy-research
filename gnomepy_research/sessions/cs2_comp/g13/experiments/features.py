"""
Candidate feature rows for every bo3 series of the walk-forward weeks, keyed (match_id, week_cutoff).

    python -m gnomepy_research.sessions.cs2_comp.g13.experiments.features --data <bundle> --out <parquet>
        --first 2026-02-23 --last 2026-07-27

EKF skill estimates, map-pool and head-to-head features, and short-memory HLTV skill fits (form, as distinct from
e12's year-long class). Join them to the rows of `g13.dev_rows` for `oracle.py` and `layer_walk.py`.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g13.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.g13.e12.features import feature_rows
from gnomepy_research.sessions.cs2_comp.g13.e12.model import Config, series_price
from gnomepy_research.sessions.cs2_comp.g13.experiments.ekf import ekf_rows
from gnomepy_research.sessions.cs2_comp.g13.experiments.maps import map_rows
from gnomepy_research.sessions.cs2_comp.g13.rows import WEEK, history_frame

SHORT_FITS = {
    "s20": Config(sigma=0.5, half_life_days=20.0, window_days=120, rating_gamma=0.6, rating_sd=0.31),
    "s45": Config(sigma=0.5, half_life_days=45.0, window_days=180, rating_gamma=0.6, rating_sd=0.31),
    "s90": Config(sigma=0.5, half_life_days=90.0, window_days=270, rating_gamma=0.6, rating_sd=0.31),
    "nr300": Config(sigma=0.5, half_life_days=300.0, window_days=365, rating_gamma=0.0),
}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    a = ap.parse_args()
    cutoffs = list(pd.date_range(a.first, a.last, freq=WEEK))
    until = cutoffs[-1] + WEEK
    inputs = Inputs(a.data, until, faceit=False)
    frame = history_frame(inputs, cutoffs)
    out = frame[["match_id", "week_cutoff"]].copy()
    t0 = time.time()
    out = out.join(ekf_rows(a.data, frame, until).drop(columns="match_id"))
    out = out.join(map_rows(a.data, frame, until).drop(columns="match_id"))
    print(f"ekf + maps {time.time() - t0:.0f}s", flush=True)
    for name, cfg in SHORT_FITS.items():
        t0 = time.time()
        h = feature_rows(inputs, frame, cfg, False)
        out["z_" + name] = _logit(series_price(h.gap.to_numpy()))
        print(f"{name} {time.time() - t0:.0f}s", flush=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
