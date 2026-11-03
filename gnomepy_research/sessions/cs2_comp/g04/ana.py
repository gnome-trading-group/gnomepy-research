from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g04 import evalp

C = "/Users/mprey/gnome-research-work/cs2/groundup/comp/submissions/g04/"
t = evalp.truth_and_market()
f = pd.read_parquet(C + "_dump_f.parquet").set_index("match_id")
h = pd.read_parquet(C + "_dump_h.parquet").set_index("match_id")
t = t[t.match_id.isin(f.index)].copy()
bo3 = lambda z: (lambda p: p * p * (3 - 2 * p))(1 / (1 + np.exp(-z)))
lg = evalp.logit
t["qf"] = lg(bo3(f.loc[t.match_id, "z"].values))
t["qh"] = lg(bo3(h.loc[t.match_id, "z"].values))
t["ea"] = f.loc[t.match_id, "exp_a"].values
t["eb"] = f.loc[t.match_id, "exp_b"].values
t["emin"] = np.minimum(t.ea, t.eb)
t["thin"] = (t.emin < 3.0).astype(float)
y, m = t.y.values, t.month.values
base = evalp.ll(t.m0.values, y)
l0 = lg(t.m0.values)
one = np.ones(len(y))


def cv(*cols):
    X = np.column_stack(list(cols) + [one])
    out = np.empty(len(y))
    for mo in np.unique(m):
        tr = m != mo
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ evalp.fit(X[tr], y[tr])))
    d = evalp.ll(out, y) - base
    return round(d.mean(), 4)


print("emin quantiles", t.emin.quantile([.1, .25, .5, .75]).round(2).tolist())
print("faceit only", cv(l0, t.qf.values))
print("hltv only", cv(l0, t.qh.values))
print("both", cv(l0, t.qf.values, t.qh.values))
print("faceit x thin", cv(l0, t.qf.values, t.qf.values * t.thin.values))
print("faceit x emin", cv(l0, t.qf.values, t.qf.values * t.emin.values))
print("faceit-hltv diff", cv(l0, t.qh.values, (t.qf - t.qh).values))

avg = (t.qf + t.qh) / 2
print("avg", cv(l0, avg.values))
out = pd.DataFrame({"match_id": t.match_id, "q": 1 / (1 + np.exp(-avg.values)), "fit_cutoff": pd.to_datetime(t.week_cutoff)})
out.to_parquet(C + "_avg.parquet")
