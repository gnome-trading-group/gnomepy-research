"""
Development walk-forward for g16: the harness's pool score, computed on the dev targets from the bundle's own
outcomes, for one or more layer specs, from rows saved by `predict.py --rows`.

    python -m gnomepy_research.sessions.cs2_comp.g16.walk --rows <dir> [--spec name ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g16.layer import Spec, logit
from gnomepy_research.sessions.cs2_comp.g16.predict import SPEC, combine

SPECS = {
    "default": SPEC,
    "last_only": Spec(columns=("zl", "zs"), prior_mean={"zl": 1.0}),
    "twap_only": Spec(columns=("zm", "zs"), prior_mean={"zm": 1.0}),
    "market_last": Spec(columns=("zl",), prior_mean={"zl": 1.0}),
    "market_twap": Spec(columns=("zm", "dz"), prior_mean={"zm": 1.0}),
    "liq": Spec(columns=("zm", "dz", "zs", "zm_liq", "zs_liq"), prior_mean={"zm": 1.0}),
    "liq_sd": Spec(columns=("zm", "dz", "zs", "zm_liq", "zs_liq", "zs_sd"), prior_mean={"zm": 1.0}),
    "sd": Spec(columns=("zm", "dz", "zs", "zs_sd"), prior_mean={"zm": 1.0}),
    "new": Spec(columns=("zm", "dz", "zs", "new", "zs_new"), prior_mean={"zm": 1.0}),
    "intercept": Spec(columns=("zm", "dz", "zs"), prior_mean={"zm": 1.0}, intercept=True),
    "tight": Spec(columns=("zm", "dz", "zs"), prior_mean={"zm": 1.0}, prior_precision=20.0),
    "loose": Spec(columns=("zm", "dz", "zs"), prior_mean={"zm": 1.0}, prior_precision=0.5),
}


def ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _fit(X, y):
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return w


def pool(m0, q, y, months):
    X = np.column_stack([logit(m0), logit(q), np.ones(len(y))])
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ _fit(X[tr], y[tr])))
    return out


def ci(x, n_boot=2000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(0)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return f"{x.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}]"


def evaluate(train: pd.DataFrame, tr: pd.DataFrame, spec: Spec) -> dict:
    q = combine(train, tr, spec).merge(tr[["match_id", "y", "m0", "is_lan", "tier", "n_active"]], on="match_id")
    y = q.y.to_numpy(float)
    months = pd.to_datetime(q.fit_cutoff).dt.to_period("M").astype(str).to_numpy()
    lq, lm, lp = ll(q.q, y), ll(q.m0, y), ll(pool(q.m0, q.q, y, months), y)
    thin = (q.n_active < 300).to_numpy()
    return {"q-market": ci(lq - lm), "pool-market": ci(lp - lm), "pool-market thin": ci((lp - lm)[thin]),
            "pool-market rest": ci((lp - lm)[~thin]), "market": f"{lm.mean():.4f}", "q": f"{lq.mean():.4f}", "pool": f"{lp.mean():.4f}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--spec", nargs="*", default=list(SPECS))
    a = ap.parse_args()
    rows = Path(a.rows)
    train = pd.read_parquet(rows / "train.parquet")
    tr = pd.read_parquet(rows / "targets.parquet")
    outcomes = train[["match_id", "y"]].drop_duplicates("match_id")
    tr = tr.merge(outcomes, on="match_id", how="left")
    print(f"targets {len(tr)}, with outcome {int(tr.y.notna().sum())}, with skill {int(tr.p_series.notna().sum())}")
    tr = tr[tr.y.notna() & tr.m0.notna()]
    for name in a.spec:
        r = evaluate(train, tr, SPECS[name])
        print(f"{name:12s} " + "  ".join(f"{k} {v}" for k, v in r.items()), flush=True)


if __name__ == "__main__":
    main()
