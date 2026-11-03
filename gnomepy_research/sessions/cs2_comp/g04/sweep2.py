from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g04 import evalp, model
from gnomepy_research.sessions.cs2_comp.g04.model import MapPoolFit

D = evalp.D
t = evalp.truth_and_market()
tg = pd.read_parquet(f"{D}/targets_tune.parquet")
tg = tg[tg.match_id.isin(t.match_id)]
OUT = "/Users/mprey/gnome-research-work/cs2/groundup/comp/submissions/g04/_sw.parquet"
for lm in [1.0, 3.0, 10.0]:
    for pw in [1.0, 2.0]:
        rows = []
        for cutoff, g in tg.groupby("week_cutoff"):
            f = MapPoolFit(D, cutoff, lam_team=1.0, lam_map=lm)
            for r in g.itertuples():
                rows.append((r.match_id, f.series_prob(r.team_a_id, r.team_b_id, pw), f.series_prob(r.team_a_id, r.team_b_id, pw, False)))
        p = pd.DataFrame(rows, columns=["match_id", "qm", "qt"]).merge(t[["match_id", "m0"]])
        lg = lambda x: np.log(x / (1 - x))
        delta = lg(p.qm) - lg(p.qt)
        print(lm, pw, "delta sd", delta.std().round(4), flush=True)
        for k in [0, 0.5, 1, 2]:
            p["q"] = 1 / (1 + np.exp(-(lg(p.m0.clip(0.01, 0.99)) + k * delta)))
            p[["match_id", "q"]].to_parquet(OUT)
            r = evalp.score(OUT, t)
            print("  k", k, round(r["q-mkt"], 4), round(r["pool-mkt"], 4), flush=True)
