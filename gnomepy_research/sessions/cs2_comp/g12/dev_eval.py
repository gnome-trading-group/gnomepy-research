"""
Walk-forward evaluation of the stack on the dev targets, scored like the harness: the pool of logit(m0) and logit(q)
refitted with each month held out, against the market alone, and with the e12 reference added against the pool of the
market and e12 (e12's dev predictions come from run_e12 on Batch). Each rater is also scored on its own.

    python -m gnomepy_research.sessions.cs2_comp.g12.dev_eval --data <bundle> [--e12 <dir of e12 parquets>]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.data import load_hltv, load_target_market, naive
from gnomepy_research.sessions.cs2_comp.g12.predict import COLS, panel
from gnomepy_research.sessions.cs2_comp.g12.stack import Stack, fit_logistic, logit, sigmoid


def ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pool(m0, q, y, months, extra=None):
    X = np.column_stack([logit(m0), logit(q)] + ([] if extra is None else [logit(extra)]) + [np.ones(len(y))])
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = sigmoid(X[~tr] @ fit_logistic(X[tr], y[tr], 1e-6))
    return out


def boot(x, n=1000, seed=0):
    r = np.random.default_rng(seed)
    b = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return f"{x.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}]"


def report(name: str, d: pd.DataFrame, q: np.ndarray) -> None:
    y = d.y.to_numpy(float)
    months = pd.to_datetime(d.time).dt.to_period("M").astype(str).to_numpy()
    lm, lq, lp = ll(d.m0, y), ll(q, y), ll(pool(d.m0, q, y, months), y)
    line = f"{name:10s} n={len(d)} market {lm.mean():.4f}  q-m {boot(lq - lm)}  pool-m {boot(lp - lm)}"
    if "e12" in d:
        lb = ll(pool(d.m0, d.e12, y, months), y)
        ls = ll(pool(d.m0, q, y, months, d.e12), y)
        line += f"  e12 pool-m {boot(lb - lm)}  stack-basepool {boot(ls - lb)}"
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--e12")
    a = ap.parse_args()
    t = pd.read_parquet(Path(a.data) / "targets_tune.parquet")
    t["week_cutoff"] = naive(t.week_cutoff)
    p = panel(a.data, t)
    out = os.environ.get("OUTPUT_DIR")
    if out:
        p.to_parquet(Path(out) / "panel.parquet", index=False)
    hist = p[~p.target]
    ev = p[p.target].merge(load_target_market(a.data, t.match_id)[["match_id", "m0"]], on="match_id", how="left")
    ev["y"] = ev.match_id.map(load_hltv(Path(a.data), pd.Timestamp("2026-08-01"))[1].set_index("match_id").y)
    if a.e12:
        e12 = pd.concat([pd.read_parquet(f) for f in sorted(Path(a.e12).glob("*.parquet"))])
        ev = ev.merge(e12[["match_id", "q"]].rename(columns={"q": "e12"}), on="match_id", how="left")
    ev = ev[ev.m0.notna() & ev.y.notna()].reset_index(drop=True)
    for name, cols in [("stack", COLS)] + [(c, (c,)) for c in COLS]:
        q = np.empty(len(ev))
        for w in sorted(ev.week_cutoff.unique()):
            m = (ev.week_cutoff == w).to_numpy()
            q[m] = Stack(cols).fit(hist, pd.Timestamp(w)).predict(ev[m])
        report(name, ev, q)


if __name__ == "__main__":
    main()
