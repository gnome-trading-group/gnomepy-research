"""Dev check of predict's components against the market and the e12 pool. python -m ...g10.dev.compscan --dir D"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g10.dev.stackscan import fitw
from gnomepy_research.sessions.cs2_comp.g10.evaluate import Scorer, ci, ll, logit, pool
from gnomepy_research.sessions.cs2_comp.g10.predict import WEIGHTS, combine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    comp = pd.concat([pd.read_parquet(p) for p in sorted(Path(a.dir).glob("components_*.parquet"))], ignore_index=True)
    ev = pd.read_parquet(Path(a.dir) / "ev.parquet").rename(columns={"q_e12": "q_e12_ref"})
    d = ev.merge(comp, on="match_id")
    print(f"n={len(d)} max |q_e12 - ref| {np.abs(d.q_e12 - d.q_e12_ref).max():.2e}")
    sc = Scorer(d)
    print(sc(d.q_e12, "e12"))
    print(sc(combine(d), "combined"))
    for k in WEIGHTS:
        print(sc(combine(d, {k: WEIGHTS[k]}), f"e12+{k}"))
    for scale in (2.0, 3.0):
        print(sc(combine(d, {k: v * scale for k, v in WEIGHTS.items()}), f"combined x{scale}"))
    y, mo = d.y.to_numpy(float), d.month.to_numpy()
    z0, ze, zb = logit(d.m0), logit(d.q_e12), logit(d.q_hltv)
    base = ll(pool([d.m0, d.q_e12], y, mo), y)
    incs = {k: (logit(d[k]) - zb) if k.startswith("q_") else d[k].to_numpy(float) for k in WEIGHTS}
    X = np.column_stack([z0, ze] + list(incs.values()) + [np.ones(len(y))])
    out = np.empty(len(y))
    for m in np.unique(mo):
        tr = mo != m
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fitw(X[tr], y[tr])))
    print(f"free increments on dev {ci(ll(out, y) - base)} w={fitw(X, y).round(3)}")
    lan = d.is_lan.to_numpy() == 1
    q = combine(d)
    st = ll(pool([d.m0, q, d.q_e12], y, mo), y) - base
    print(f"combined stack lan {st[lan].mean():+.4f} online {st[~lan].mean():+.4f}")
    for m in np.unique(mo):
        print(f"month {m} stack {st[mo == m].mean():+.4f} n={int((mo == m).sum())}")


if __name__ == "__main__":
    main()
