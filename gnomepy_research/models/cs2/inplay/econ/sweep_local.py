"""Development helper: score model variants on a cached month (see dev.py) against the cached benchmark predictions."""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd

from .. import harness
from .model import DEFAULT, Model

BENCH = "/Users/mprey/gnome-research-work/cs2/comp3/bench_cache/bench_89bd1ebdec_2026-08.parquet"


def score(p, d, bm):
    s = d["states"].assign(p=p, y=d["y"]).merge(bm[["game_id", "event_seq", "p"]], on=["game_id", "event_seq"], suffixes=("", "_b"))
    diff = harness.ll(s.p, s.y) - harness.ll(s.p_b, s.y)
    g = pd.DataFrame(dict(d=diff, g=s.game_id)).groupby("g").d.agg(["sum", "size"])
    rng = np.random.default_rng(0)
    a, n = g["sum"].to_numpy(), g["size"].to_numpy()
    boot = [a[i].sum() / n[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(500))]
    start = (s.kind == "start").to_numpy()
    return dict(diff=round(float(diff.mean()), 5), lo=round(float(np.quantile(boot, .025)), 5), hi=round(float(np.quantile(boot, .975)), 5),
                start=round(float(diff[start].mean()), 5), inround=round(float(diff[~start].mean()), 5))


def main(pkl, variants):
    d = pickle.load(open(pkl, "rb"))
    bm = pd.read_parquet(BENCH)
    base = Model(dict(DEFAULT, eco=False, round_extra=False), 0).fit_rows(d["train"], d["legal"], d["x_tr"], d["rt_tr"])
    print("bench-repro", score(base.predict_rows(d["x_te"], d["rt_te"]), d, bm), flush=True)
    rm = {}
    for name, over in variants:
        cfg = {**DEFAULT, **over}
        key = cfg["round_extra"]
        m = Model(cfg, 0).fit_rows(d["train"], d["legal"], d["x_tr"], d["rt_tr"], map_model=base.map_model, round_model=rm.get(key), aug=(d["x_aug"], d["rt_aug"]))
        rm[key] = m.round_model
        print(name, score(m.predict_rows(d["x_te"], d["rt_te"]), d, bm), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], json.loads(sys.argv[2]))
