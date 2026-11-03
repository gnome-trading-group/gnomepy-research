from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g04 import evalp

D = evalp.D
t = evalp.truth_and_market()
p = pd.read_parquet(f"{D}/market/target_paths.parquet")
p = p[p.match_id.isin(t.match_id)].sort_values(["match_id", "t"])
rows = {}
for mid, g in p.groupby("match_id"):
    k = g.kickoff.iloc[0]
    last = g[g.t <= k - pd.Timedelta(minutes=0)]
    def at(h):
        s = last[last.t <= k - pd.Timedelta(hours=h)]
        return s.p_a.iloc[-1] if len(s) else np.nan
    w = last[last.t >= k - pd.Timedelta(hours=6)]
    rows[mid] = dict(n=len(last), d6=at(0.5) - at(6), d24=at(0.5) - at(24), d48=at(0.5) - at(47), vol=w.p_a.diff().abs().sum() if len(w) else np.nan,
                     age=(k - last.t.iloc[-1]).total_seconds() / 60 if len(last) else np.nan)
f = pd.DataFrame(rows).T
t = t.merge(f, left_on="match_id", right_index=True)
lg = evalp.logit
y, m = t.y.values, t.month.values
base = evalp.ll(t.m0.values, y)
def cv(X):
    out = np.empty(len(y))
    for mo in np.unique(m):
        tr = m != mo
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ evalp.fit(X[tr], y[tr])))
    return (evalp.ll(out, y) - base).mean()
l0 = lg(t.m0)
print("n prints", t.n.describe().round(1).to_dict())
for name, col in [("d6", "d6"), ("d24", "d24"), ("d48", "d48"), ("vol", "vol"), ("age", "age")]:
    z = t[col].fillna(0).values
    print(name, round(cv(np.column_stack([l0, z, np.ones(len(y))])), 4))
t["lo"] = (lg(t.m0) )
print("all drift", round(cv(np.column_stack([l0, t.d6.fillna(0), t.d24.fillna(0), np.ones(len(y))])), 4))
print("extremity", round(cv(np.column_stack([l0, np.abs(l0) * l0, np.ones(len(y))])), 4))
