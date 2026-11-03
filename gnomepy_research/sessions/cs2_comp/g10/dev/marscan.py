"""Check market x skill interactions on the pre-dev market months (past series of the rows dump, before the first
target week), folds by week. python -m ...g10.dev.marscan --work W"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g10.dev.stackscan import fitw
from gnomepy_research.sessions.cs2_comp.g10.evaluate import ci, ll, logit
from gnomepy_research.sessions.cs2_comp.g10.layer import context_columns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    R = pd.read_parquet(f"{a.work}/rows.parquet")
    d = R[~R.is_target & R.m0.notna() & R.y.notna() & (pd.to_datetime(R.time) < "2026-03-30")].reset_index(drop=True)
    y = d.y.to_numpy(float)
    wk = d.week_cutoff.astype(str).to_numpy()
    z0, ze = logit(d.m0), logit(d.q_e12)
    ctx = context_columns(d)
    de = ze - z0

    def cv(extra):
        X = np.column_stack([z0, ze] + [np.asarray(e, float) for e in extra] + [np.ones(len(y))])
        out = np.empty(len(y))
        for m in np.unique(wk):
            tr = wk != m
            out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fitw(X[tr], y[tr])))
        return ll(out, y), fitw(X, y)

    base, wb = cv([])
    print(f"n={len(y)} market {ll(d.m0, y).mean():.4f} base pool {base.mean():.4f} w={wb.round(3)}")
    for label, extra in [("twap120", [d.zbar_120m - z0]), ("de*young", [de * ctx["young"].to_numpy()]),
                         ("de*liq", [de * ctx["liq"].to_numpy()]), ("de*thin", [de * ctx["thin"].to_numpy()]),
                         ("z0*liq", [z0 * ctx["liq"].to_numpy()])]:
        l, w = cv(extra)
        print(f"{label:20s} {ci(l - base)} w={w[2:-1].round(3)}", flush=True)


if __name__ == "__main__":
    main()
