import sys, numpy as np, pandas as pd
sys.path.insert(0, 'gnomepy_research/models/cs2/inplay/free/work')
from exp import load, walk, cmp, C, MONTHS
from feat1 import add_hist
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, next_round
rt, st = load()
rt = add_hist(rt, rt.match_begin < pd.Timestamp('2026-01-01', tz='UTC'))
er = pd.read_parquet(f'{C}/ev_round.parquet').set_index(['game_id', 'round'])
rt = rt.join(er, on=['game_id', 'round'])
g = rt.game_id
for c, v in [('kill', rt.b_dead - rt.a_dead), ('open', rt.open_a)]:
    cs = v.groupby(g).cumsum()
    lag = cs.groupby(g).shift(2)
    has_ev = rt.game_id.isin(set(er.reset_index().game_id))
    lag = np.where(has_ev, lag.fillna(0), np.nan)
    rt[c + '_mean'] = lag / (rt['round'].clip(lower=2) - 2 + 4)
    # value for the next round = through this round - 1, scaled for n_prev+1
    rt[c + '_next'] = np.where(has_ev, cs.groupby(g).shift(1).fillna(0), np.nan) / (rt['round'] - 1 + 4)
def nr_ext(cols):
    def f(rr, a_won):
        X, done = next_round(rr, a_won)
        rs = rr.res_sum + a_won - rr.p_exp_a
        for c in cols:
            if c == 'res_mean': X[c] = (rs / (rr.n_prev + 5)).to_numpy()
            elif c in ('kill_mean', 'open_mean'): X[c] = rr[c.replace('_mean', '_next')].to_numpy()
        return X, done
    return f
base = pd.read_parquet(f'{C}/rep_all.parquet')
cfgs = {
 'res_kill': ['res_mean', 'kill_mean'],
 'res_open': ['res_mean', 'open_mean'],
 'res_kill_open': ['res_mean', 'kill_mean', 'open_mean'],
}
for k, cols in cfgs.items():
    r = walk(rt, st, MONTHS, verbose=False, mapcols=MAPCOLS + cols, nr=nr_ext(cols))
    r.to_parquet(f'{C}/e3_{k}.parquet')
    cmp(r, base, k)
