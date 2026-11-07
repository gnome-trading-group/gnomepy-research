import sys, numpy as np, pandas as pd
sys.path.insert(0, 'gnomepy_research/models/cs2/inplay/free/work')
from exp import load, walk, cmp, C, MONTHS
from feat1 import add_hist
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, next_round
rt, st = load()
rt = add_hist(rt, rt.match_begin < pd.Timestamp('2026-01-01', tz='UTC'))
def nr_ext(cols):
    def f(rr, a_won):
        X, done = next_round(rr, a_won)
        rs = rr.res_sum + a_won - rr.p_exp_a
        for c in cols:
            if c == 'res_sum': X[c] = rs.to_numpy()
            elif c == 'res_mean': X[c] = (rs / (rr.n_prev + 5)).to_numpy()
            elif c == 'tier_code': X[c] = rr.tier_code.to_numpy()
        return X, done
    return f
base = walk(rt, st, MONTHS, verbose=False)
base.to_parquet(f'{C}/rep_all.parquet')
cfgs = {
 'res_map': dict(mapcols=MAPCOLS+['res_mean'], nr=nr_ext(['res_mean'])),
 'res_sum_map': dict(mapcols=MAPCOLS+['res_sum'], nr=nr_ext(['res_sum'])),
 'res_tier_map': dict(mapcols=MAPCOLS+['res_mean','tier_code'], nr=nr_ext(['res_mean','tier_code'])),
}
for k, kw in cfgs.items():
    r = walk(rt, st, MONTHS, verbose=False, **kw)
    r.to_parquet(f'{C}/e2_{k}.parquet')
    cmp(r, base, k)
