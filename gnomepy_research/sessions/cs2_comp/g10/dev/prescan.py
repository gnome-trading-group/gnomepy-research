"""Score the pre-dev skill increments: y ~ logit(q_base) + (logit(q_x) - logit(q_base)), months held out.
python -m ...g10.dev.prescan --pre DIR"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g10.dev.stackscan import fitw
from gnomepy_research.sessions.cs2_comp.g10.evaluate import ci, ll, logit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(p) for p in sorted(Path(a.pre).glob("prehist_*.parquet"))], ignore_index=True)
    y = d.y.to_numpy(float)
    mo = pd.to_datetime(d.time).dt.strftime("%Y-%m").to_numpy()
    zb = logit(d.q_base)

    def cv(extra, mask=None):
        X = np.column_stack([zb] + [np.asarray(e, float) for e in extra] + [np.ones(len(y))])
        out = np.empty(len(y))
        for m in np.unique(mo):
            tr = mo != m
            out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fitw(X[tr], y[tr])))
        return ll(out, y), fitw(X, y)

    base, wb = cv([])
    print(f"n={len(y)} raw base {ll(d.q_base, y).mean():.4f} cv base {base.mean():.4f} w={wb.round(3)}")
    for c in [c for c in d.columns if c.startswith("q_") and c != "q_base"]:
        inc = logit(d[c]) - zb
        l, w = cv([inc])
        lan = d.is_lan.to_numpy() == 1
        print(f"{c:10s} raw {ci(ll(d[c], y) - ll(d.q_base, y))} cv-inc {ci(l - base)} w={w[1]:.3f} "
              f"lan {np.mean((l - base)[lan]):+.4f} online {np.mean((l - base)[~lan]):+.4f} sd_inc {inc.std():.3f}", flush=True)
    for half in (mo < "2025-08", mo >= "2025-08"):
        print("half", np.unique(mo[half])[[0, -1]], end=" ")
        for c in ["q_m240", "q_lan", "q_lanmap"]:
            l, _ = cv([logit(d[c]) - zb])
            print(f"{c} {np.mean((l - base)[half]):+.4f}", end=" ")
        print()


if __name__ == "__main__":
    main()
