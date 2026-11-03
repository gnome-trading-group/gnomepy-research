from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g04 import evalp
from gnomepy_research.sessions.cs2_comp.g04 import players
from gnomepy_research.sessions.cs2_comp.g04.players import PlayerFit

D = evalp.D
t = evalp.truth_and_market()
tg = pd.read_parquet(f"{D}/targets_tune.parquet")
tg = tg[tg.match_id.isin(t.match_id)]
OUT = "/Users/mprey/gnome-research-work/cs2/groundup/comp/submissions/g04/_sw.parquet"


def bo3(p):
    return p * p * (3 - 2 * p)


for stl in [['rating','logn','f_adr'], ['rating','logn','f_adr','f_n']]:
    rp, hl, lt = 20, 1200, 1.0
    players.STATS = stl
    st = stl
    lp, sf = 16, 1.0
    rows = []
    for cutoff, g in tg.groupby("week_cutoff"):
        f = PlayerFit(D, cutoff, lp, lt, hl, sf, rp)
        for r in g.itertuples():
            z = f.map_logit(r.team_a_id, r.team_a_lineup_ids, r.team_b_id, r.team_b_lineup_ids)
            rows.append((r.match_id, bo3(1 / (1 + np.exp(-z)))))
    pd.DataFrame(rows, columns=["match_id", "q"]).to_parquet(OUT)
    r = evalp.score(OUT, t)
    print(st, round(r["q-mkt"], 4), round(r["pool-mkt"], 4), r["ci"], flush=True)
