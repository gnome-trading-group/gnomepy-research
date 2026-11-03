"""
g05 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g05.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13]

q is a skill price only. The harness pools q with the kickoff price, so whatever q repeats of the market is wasted
collinearity; every price-path summary tested (late drift, a smoothed kickoff price, liquidity and staleness
interactions, persistent per-team market surprise) added nothing once a skill price sat next to the market, so the
market is left to the pool.

The skill price is the e12 player-skill fit (borrowed: HLTV maps, and HLTV plus FACEIT maps, each side priced with its
announced five) under each configuration in SKILL, turned into a price by a small logistic layer, with the prices
averaged in logit space. SKILL holds only e12's own configuration and layer weights (fitted by e12 on its Apr 2025 -
Mar 2026 walk-forward, before the first tune cutoff, so point-in-time for every target): configuration variants with
layers refitted on Oct 2025 - Mar 2026 (`variants.py --mode pre`) did no better alongside the market.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g05.data import Inputs
from gnomepy_research.sessions.cs2_comp.g05.features import feature_rows
from gnomepy_research.sessions.cs2_comp.g05.model import Config, series_price
from gnomepy_research.sessions.cs2_comp.g05.variants import BASE, prepare_targets

LAYER_COLUMNS = ["zh", "zh_thin", "standin", "idle", "zf", "zf_thin"]
SKILL: dict[str, tuple[Config, list[float]]] = {
    "base": (BASE, [0.149912, -0.043914, -0.099792, -0.026092, 0.646569, 0.00154]),
}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(h: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Layer columns, antisymmetric in A and B: swapping the teams flips each sign."""
    zh = _logit(series_price(h.gap.to_numpy()))
    zf = _logit(series_price(f.gap.to_numpy()))
    return pd.DataFrame({
        "zh": zh, "zh_thin": zh * (h.thin_a + h.thin_b).to_numpy(float),
        "standin": (h.standin_a - h.standin_b).to_numpy(float),
        "idle": np.log1p(h.idle_a.to_numpy(float)) - np.log1p(h.idle_b.to_numpy(float)),
        "zf": zf, "zf_thin": zf * (f.thin_a + f.thin_b).to_numpy(float)})


def predict(data_dir: str | Path, targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    inputs = Inputs(data_dir, targets.week_cutoff.max(), faceit=True)
    z = []
    for name, (cfg, weights) in SKILL.items():
        h = feature_rows(inputs, targets, cfg, False, verbose=verbose).set_index("match_id").loc[targets.match_id]
        f = feature_rows(inputs, targets, cfg, True, verbose=verbose).set_index("match_id").loc[targets.match_id]
        z.append(design(h, f)[LAYER_COLUMNS].to_numpy() @ np.asarray(weights))
    return pd.DataFrame({"match_id": targets.match_id.to_numpy(), "q": expit(np.mean(z, axis=0)),
                         "fit_cutoff": pd.to_datetime(targets.week_cutoff).to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    a = ap.parse_args()
    # BLAS sees every core of a Batch host, not the task's vCPUs; oversubscribed threads made one fit take minutes.
    threadpool_limits(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    targets = prepare_targets(pd.read_parquet(a.targets))
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = predict(a.data, targets.reset_index(drop=True), verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
