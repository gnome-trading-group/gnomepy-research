"""Score matchmaking form: pre-dev on outcomes over the HLTV-only skill price, dev on top of the e12 pool.
python -m ...g10.dev.mmscan --dir D"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g10.dev.stackscan import fitw
from gnomepy_research.sessions.cs2_comp.g10.dev.evaluate import ci, ll, logit, pool


def cvfit(cols, y, groups):
    X = np.column_stack([np.asarray(c, float) for c in cols] + [np.ones(len(y))])
    out = np.empty(len(y))
    for m in np.unique(groups):
        tr = groups != m
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fitw(X[tr], y[tr])))
    return ll(out, y), fitw(X, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D = Path(a.dir)
    mm = pd.read_parquet(D / "mm.parquet")
    names = sorted({c[:-2] for c in mm.columns if c.endswith("_a")})

    for n in names:
        mm[f"{n}_d"] = mm[f"{n}_a"] - mm[f"{n}_b"]
    pre = pd.concat([pd.read_parquet(p) for p in sorted(D.glob("prehist_*.parquet"))], ignore_index=True).merge(mm, on="match_id")
    y = pre.y.to_numpy(float)
    mo = pd.to_datetime(pre.time).dt.strftime("%Y-%m").to_numpy()
    zb = logit(pre.q_base)
    base, wb = cvfit([zb], y, mo)
    print(f"pre-dev n={len(y)} base {base.mean():.4f}")
    for n in names:
        l, w = cvfit([zb, pre[f"{n}_d"]], y, mo)
        early = mo < "2025-08"
        print(f"pre {n:6s} {ci(l - base)} w={w.round(3)} early {np.mean((l - base)[early]):+.4f} late {np.mean((l - base)[~early]):+.4f}", flush=True)
        l2, w2 = cvfit([pre[f"{n}_d"]], y, mo)
        print(f"   {n} alone vs intercept-only: {np.mean(l2) - np.mean(cvfit([np.zeros(len(y))], y, mo)[0]):+.4f} w={w2.round(3)}")
    for combo in (["mm", "mmact"], ["mm60", "mm60act"]):
        l, w = cvfit([zb] + [pre[f"{n}_d"] for n in combo], y, mo)
        print(f"pre joint {combo} {ci(l - base)} w={w.round(3)}")
    for n in ("mm",):
        l, w = cvfit([zb, pre[f"{n}_d"], logit(pre.q_m240s) - zb, logit(pre.q_lan) - zb], y, mo)
        print(f"pre joint {n}+map+lan {ci(l - base)} w={w.round(3)}")
    comp = pd.concat([pd.read_parquet(p) for p in sorted(D.glob("components_*.parquet"))], ignore_index=True)
    ev = pd.read_parquet(D / "ev.parquet").drop(columns="q_e12")
    d = ev.merge(comp, on="match_id").merge(mm, on="match_id", how="left")
    y = d.y.to_numpy(float)
    mo = d.month.to_numpy()
    z0, ze = logit(d.m0), logit(d.q_e12)
    bp = ll(pool([d.m0, d.q_e12], y, mo), y)
    lm = ll(d.m0, y)
    for n in names:
        x = d[f"{n}_d"].fillna(0).to_numpy()
        l, w = cvfit([z0, ze, x], y, mo)
        print(f"dev {n:6s} stack {ci(l - bp)} w={w.round(3)}", flush=True)
        for th in (0.5, 1.0, 2.0):
            q = 1 / (1 + np.exp(-(ze + th * x)))
            print(f"   q=e12+{th}*{n}: pool-mkt {ci(ll(pool([d.m0, q], y, mo), y) - lm)} stack {ci(ll(pool([d.m0, q, d.q_e12], y, mo), y) - bp)}")


if __name__ == "__main__":
    main()
