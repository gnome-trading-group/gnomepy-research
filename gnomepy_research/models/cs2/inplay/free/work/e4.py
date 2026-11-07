import sys, numpy as np, pandas as pd
sys.path.insert(0, 'gnomepy_research/models/cs2/inplay/free/work')
from exp import load, walk, cmp, C, MONTHS
from feat1 import add_hist
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, next_round, trees
from gnomepy_research.models.cs2.inplay.harness import ll
from sklearn.linear_model import LinearRegression
rt, st = load()
rt = add_hist(rt, rt.match_begin < pd.Timestamp('2026-01-01', tz='UTC'))
er = pd.read_parquet(f'{C}/ev_round.parquet').set_index(['game_id', 'round'])
rt = rt.join(er, on=['game_id', 'round'])
g = rt.game_id
has_ev = rt.game_id.isin(set(er.reset_index().game_id)).to_numpy()
kd = (rt.b_dead - rt.a_dead)
X = np.c_[np.log1p(rt.a_equip / 1000), np.log1p(rt.b_equip / 1000), rt.ct_a]
m = (rt.match_begin < pd.Timestamp('2026-04-01', tz='UTC')).to_numpy() & has_ev & kd.notna().to_numpy()
kexp = LinearRegression().fit(X[m], kd[m]).predict(X)
print('kexp R2', LinearRegression().fit(X[m], kd[m]).score(X[m], kd[m]))
def lagged(v, name):
    cs = v.groupby(g).cumsum()
    rt[name + '_mean'] = np.where(has_ev, cs.groupby(g).shift(2).fillna(0), np.nan) / (rt['round'] - 2).clip(lower=0).add(4)
    rt[name + '_next'] = np.where(has_ev, cs.groupby(g).shift(1).fillna(0), np.nan) / (rt['round'] - 1 + 4)
    rt[name + '_cur'] = np.where(has_ev, cs.groupby(g).shift(1).fillna(0), np.nan)
lagged(kd, 'kill'); lagged(kd - kexp, 'killx')
def nr_ext(cols):
    def f(rr, a_won):
        X, done = next_round(rr, a_won)
        rs = rr.res_sum + a_won - rr.p_exp_a
        for c in cols:
            if c == 'res_mean': X[c] = (rs / (rr.n_prev + 5)).to_numpy()
            elif c == 'tier_code': X[c] = rr.tier_code.to_numpy()
            elif c.endswith('_mean'): X[c] = rr[c.replace('_mean', '_next')].to_numpy()
        return X, done
    return f
base = pd.read_parquet(f'{C}/rep_all.parquet')
cfgs = {
 'res_killx': ['res_mean', 'killx_mean'],
 'res_kill_tier': ['res_mean', 'kill_mean', 'tier_code'],
 'res_kill_d3': (['res_mean', 'kill_mean'], dict(max_depth=3, n_estimators=1000)),
 'res_kill_mcw200': (['res_mean', 'kill_mean'], dict(min_child_weight=200)),
}
for k, v in cfgs.items():
    cols, mk = (v if isinstance(v, tuple) else (v, {}))
    r = walk(rt, st, MONTHS, verbose=False, mapcols=MAPCOLS + cols, nr=nr_ext(cols), map_kw=mk)
    r.to_parquet(f'{C}/e4_{k}.parquet')
    cmp(r, base, k)
