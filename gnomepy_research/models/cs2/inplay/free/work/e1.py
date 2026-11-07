import sys, numpy as np, pandas as pd
sys.path.insert(0, 'gnomepy_research/models/cs2/inplay/free/work')
from exp import load, walk, cmp, C
from feat1 import add_hist, orient
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, next_round
rt, st = load()
rt = add_hist(rt, rt.match_begin < pd.Timestamp('2026-01-01', tz='UTC'))
st = orient(st, rt, ['res_sum', 'res_mean', 'surv_sum', 'tier_code', 'n_prev'])
j = st.join(rt.set_index(['game_id','round'])[['last_a_alive','last_b_alive','ct_a']], on=['game_id','round'])
st['ct_last_alive'] = np.where(j.ct_a==1, j.last_a_alive, j.last_b_alive); st['t_last_alive'] = np.where(j.ct_a==1, j.last_b_alive, j.last_a_alive)
def nr_ext(cols):
    def f(rr, a_won):
        X, done = next_round(rr, a_won)
        for c in cols:
            if c == 'res_sum': X[c] = (rr.res_sum + a_won - rr.p_exp_a).to_numpy()
            elif c == 'res_mean': X[c] = ((rr.res_sum + a_won - rr.p_exp_a) / (rr.n_prev + 5)).to_numpy()
            elif c == 'tier_code': X[c] = rr.tier_code.to_numpy()
        return X, done
    return f
base = pd.read_parquet(f'{C}/rep_aug.parquet')
M = ['2026-08']
cfgs = {
 'tier_both': dict(roundcols=ROUND+['ct_tier_code'], mapcols=MAPCOLS+['tier_code'], nr=nr_ext(['tier_code'])),
 'res_map': dict(mapcols=MAPCOLS+['res_mean'], nr=nr_ext(['res_mean'])),
 'res_round': dict(roundcols=ROUND+['ct_res_mean','ct_n_prev']),
 'surv_round': dict(roundcols=ROUND+['ct_surv_sum','ct_last_alive','t_last_alive']),
 'prior_round': dict(roundcols=ROUND+['ct_prior','has_prior']),
}
for k, kw in cfgs.items():
    r = walk(rt, st, M, verbose=False, **kw)
    cmp(r, base, k)
