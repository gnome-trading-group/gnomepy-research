"""
e12 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.e12.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--no-faceit] [--weeks 2026-04-06,2026-04-13]

Per distinct week_cutoff: an HLTV-only skill fit, and with FACEIT on a second fit on HLTV + FACEIT maps whose fives
come from the newest appearance in either source. The logistic layer below combines them; its weights were fitted by
`dev.py` on the Apr 2025 - Mar 2026 weekly walk-forward (run as two half-year pieces), all of it before the first
tune cutoff, so they are point-in-time for every target.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.e12.features import feature_rows
from gnomepy_research.sessions.cs2_comp.e12.model import Config, series_price

CONFIG_HLTV = Config(sigma=0.5, half_life_days=300.0, rating_gamma=0.6, rating_sd=0.31)
CONFIG_FACEIT = Config(sigma=0.5, half_life_days=300.0, rating_gamma=0.6, rating_sd=0.31)
LAYER_HLTV = {"zh": 0.727237, "zh_thin": -0.093612, "standin": -0.082797, "idle": -0.037279}
LAYER_FACEIT = {"zh": 0.319305, "zh_thin": -0.051195, "standin": -0.089154, "idle": -0.021486, "zf": 0.418045, "zf_thin": -0.055322}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(h: pd.DataFrame, f: pd.DataFrame | None) -> pd.DataFrame:
    """Antisymmetric in A and B (no intercept): swapping the teams flips every column's sign."""
    zh = _logit(series_price(h.gap.to_numpy()))
    cols = {"zh": zh, "zh_thin": zh * (h.thin_a + h.thin_b).to_numpy(float),
            "standin": (h.standin_a - h.standin_b).to_numpy(float),
            "idle": np.log1p(h.idle_a.to_numpy(float)) - np.log1p(h.idle_b.to_numpy(float))}
    if f is not None:
        zf = _logit(series_price(f.gap.to_numpy()))
        cols.update({"zf": zf, "zf_thin": zf * (f.thin_a + f.thin_b).to_numpy(float)})
    return pd.DataFrame(cols)


def rows(inputs: Inputs, targets: pd.DataFrame, faceit: bool, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    h = feature_rows(inputs, targets, CONFIG_HLTV, False, verbose=verbose)
    f = feature_rows(inputs, targets, CONFIG_FACEIT, True, verbose=verbose) if faceit else None
    return h, f


def predict(data_dir: str | Path, targets: pd.DataFrame, faceit: bool, verbose: bool = False) -> pd.DataFrame:
    t = targets.rename(columns={"team_a_id": "team_a", "team_b_id": "team_b"})[["match_id", "team_a", "team_b", "week_cutoff"]]
    inputs = Inputs(data_dir, t.week_cutoff.max(), faceit)
    h, f = rows(inputs, t, faceit, verbose)
    X = design(h, f)
    layer = LAYER_FACEIT if faceit else LAYER_HLTV
    q = expit(X.to_numpy() @ np.array([layer[c] for c in X.columns]))
    return pd.DataFrame({"match_id": h.match_id.to_numpy(), "q": q, "fit_cutoff": pd.to_datetime(h.week_cutoff).to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    wc = pd.to_datetime(targets.week_cutoff)
    targets["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets, a.faceit, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
