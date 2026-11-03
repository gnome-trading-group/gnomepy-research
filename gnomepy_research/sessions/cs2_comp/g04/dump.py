from __future__ import annotations

import json
from itertools import combinations
import sys

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g04.pm import Fit

D = "/Users/mprey/gnome-research-work/cs2/groundup/comp/data"
cfg = json.loads(sys.argv[1])
tg = pd.read_parquet(f"{D}/targets_tune.parquet")
rows = []
for cutoff, g in tg.groupby("week_cutoff"):
    f = Fit(D, cutoff, **cfg)
    for r in g.itertuples():
        z = f.map_logit(r.team_a_id, r.team_a_lineup_ids, r.team_b_id, r.team_b_lineup_ids)
        dm = 0.0
        if f.pmap is not None:
            zm = z + f.scale * (f.map_effects(r.team_a_lineup_ids) - f.map_effects(r.team_b_lineup_ids))
            p = 1 / (1 + np.exp(-zm))
            tri = [pi * pj + pi * (1 - pj) * pk + (1 - pi) * pj * pk for i, j, k in combinations(range(len(p)), 3) for pi, pj, pk in [(p[i], p[j], p[k])]]
            p0 = 1 / (1 + np.exp(-z))
            dm = np.log(np.mean(tri) / (1 - np.mean(tri))) - np.log(p0 * p0 * (3 - 2 * p0) / (1 - p0 * p0 * (3 - 2 * p0)))
        rows.append((r.match_id, z, f.experience(r.team_a_lineup_ids), f.experience(r.team_b_lineup_ids), dm))
pd.DataFrame(rows, columns=["match_id", "z", "exp_a", "exp_b", "dm"]).to_parquet(sys.argv[2])
