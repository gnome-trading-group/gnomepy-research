"""
Fit the frozen layer weights from walk-forward rows (walk.py / sweep2.py) on the series before a cutoff, and report
how they do on the dev targets with g14's own scorer.

    python -m gnomepy_research.sessions.cs2_comp.g14.freeze <walk rows parquet> --before 2026-03-30
        [--features z_fit,z_fit_thin,rank,new,sub] [--targets <target rows parquet>] [--e12 <parquet>] [--l2 2.0]

Prints the weights as JSON, ready to paste into predict.WEIGHTS.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.evaluate import DATA, score
from gnomepy_research.sessions.cs2_comp.g14.layer import fit_ridge_logistic, model_column

DEFAULT_FEATURES = ("z_fit", "z_fit_thin", "rank", "new", "sub")


def freeze(rows: pd.DataFrame, before: pd.Timestamp, features: tuple[str, ...], l2: float = 2.0,
           half_life_days: float = 365.0, start: pd.Timestamp | None = None) -> dict[str, float]:
    time = pd.to_datetime(rows.time)
    known = pd.to_datetime(rows.known)
    keep = (known < before) & (time >= (start if start is not None else before - pd.Timedelta(days=365)))
    tr = rows[keep]
    X = np.column_stack([model_column(tr, f) for f in features] + [np.ones(len(tr))])
    age = (before - time[keep]).dt.total_seconds().to_numpy() / 86400.0
    w = 0.5 ** (age / half_life_days)
    scale = np.maximum(X.std(axis=0), 1e-6)
    scale[-1] = 1.0
    beta = fit_ridge_logistic(X / scale, tr.y.to_numpy(float), w, l2) / scale
    return dict(zip(list(features) + ["intercept"], (float(b) for b in beta)))


def apply(rows: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    z = np.column_stack([model_column(rows, f) for f in weights]) @ np.array(list(weights.values()))
    return 1 / (1 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--before", required=True)
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    ap.add_argument("--l2", type=float, default=2.0)
    ap.add_argument("--targets")
    ap.add_argument("--e12")
    a = ap.parse_args()
    rows = pd.read_parquet(a.rows)
    weights = freeze(rows, pd.Timestamp(a.before), tuple(a.features.split(",")), a.l2)
    print(json.dumps({k: round(v, 6) for k, v in weights.items()}))
    if a.targets:
        t = pd.read_parquet(a.targets)
        p = pd.DataFrame({"match_id": t.match_id.to_numpy(), "q": apply(t, weights)})
        score(p, DATA, pd.read_parquet(a.e12) if a.e12 else None, "q")


if __name__ == "__main__":
    main()
