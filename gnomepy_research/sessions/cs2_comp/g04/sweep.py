from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g04 import model, evalp
from gnomepy_research.sessions.cs2_comp.g04.model import MapPoolFit

D = evalp.D
t = evalp.truth_and_market()
tg = pd.read_parquet(f"{D}/targets_tune.parquet")
tg = tg[tg.match_id.isin(t.match_id)]
fits = {}
for hl, lt, lm in itertools.product([120, 240, 480], [0.3, 1.0], [3.0, 10.0, 40.0]):
    model.HALF_LIFE_DAYS = hl
    qs = []
    for cutoff, g in tg.groupby("week_cutoff"):
        f = MapPoolFit(D, cutoff, lam_team=lt, lam_map=lm)
        for r in g.itertuples():
            qs.append((r.match_id, f.series_prob(r.team_a_id, r.team_b_id)))
    p = pd.DataFrame(qs, columns=["match_id", "q"])
    p.to_parquet(f"/Users/mprey/gnome-research-work/cs2/groundup/comp/submissions/g04/_sw.parquet")
    r = evalp.score(f"/Users/mprey/gnome-research-work/cs2/groundup/comp/submissions/g04/_sw.parquet", t)
    print(hl, lt, lm, round(r["q-mkt"], 4), round(r["pool-mkt"], 4), flush=True)
