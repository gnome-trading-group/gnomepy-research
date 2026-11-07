import sys, numpy as np, pandas as pd
sys.path.insert(0, 'gnomepy_research/models/cs2/inplay/free/work')
from exp import load, walk, cmp, C, MONTHS
from feat1 import add_hist
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, next_round, trees
rt, st = load()
rt = add_hist(rt, rt.match_begin < pd.Timestamp('2026-01-01', tz='UTC'))
er = pd.read_parquet(f'{C}/ev_round.parquet').set_index(['game_id', 'round'])
rt = rt.join(er, on=['game_id', 'round'])
g = rt.game_id
has_ev = rt.game_id.isin(set(er.reset_index().game_id)).to_numpy()
kd = (rt.b_dead - rt.a_dead).fillna(0)
cs = kd.groupby(g).cumsum()
rt['kill_mean'] = np.where(has_ev, cs.groupby(g).shift(2).fillna(0), np.nan) / (rt['round'] - 2).clip(lower=0).add(4)
rt['kill_next'] = np.where(has_ev, cs.groupby(g).shift(1).fillna(0), np.nan) / (rt['round'] - 1 + 4)
rt['kill1_mean'] = np.where(has_ev, cs.groupby(g).shift(1).fillna(0), np.nan) / (rt['round'] - 1 + 4)
rt['kill_cs1'] = np.where(has_ev, cs.groupby(g).shift(1).fillna(0), np.nan)
def nr_ext(cols):
    def f(rr, a_won):
        X, done = next_round(rr, a_won)
        rs = rr.res_sum + a_won - rr.p_exp_a
        for c in cols:
            if c == 'res_mean': X[c] = (rs / (rr.n_prev + 5)).to_numpy()
            elif c == 'kill_mean': X[c] = rr.kill_next.to_numpy()
        return X, done
    return f
base = pd.read_parquet(f'{C}/rep_all.parquet')
cols = ['res_mean', 'kill_mean']
for mcw in [100, 400, 1000]:
    r = walk(rt, st, MONTHS, verbose=False, mapcols=MAPCOLS + cols, nr=nr_ext(cols), map_kw=dict(min_child_weight=mcw))
    r.to_parquet(f'{C}/e5_mcw{mcw}.parquet')
    cmp(r, base, f'res_kill_mcw{mcw}')
# partial-kill variant: state-level next-round feature
from exp import ll
mc = MAPCOLS + ['res_mean', 'kill1_mean']
out = []
for m in MONTHS:
    s = pd.Timestamp(m + '-01', tz='UTC'); e = s + pd.offsets.MonthBegin(1)
    rtr = rt[rt.end < s]
    mm = trees(0, min_child_weight=200).fit(rtr[mc], rtr.map_a_wins)
    tr = st[st.end < s]
    rm = trees(0, max_depth=6, learning_rate=0.05).fit(tr[ROUND], tr.y_ct)
    te = st[(st.match_begin >= s) & (st.match_begin < e)].reset_index(drop=True)
    p_ct = rm.predict_proba(te[ROUND])[:, 1]
    rr = te[['game_id', 'round']].join(rt.set_index(['game_id', 'round']), on=['game_id', 'round'])
    a_ct = (te.side_a == 'CT').to_numpy()
    kd_part = np.where(a_ct, te.ct_alive - te.t_alive, te.t_alive - te.ct_alive)
    mp = {}
    for a_won in (1, 0):
        X, done = next_round(rr, a_won)
        X['res_mean'] = ((rr.res_sum + a_won - rr.p_exp_a) / (rr.n_prev + 5)).to_numpy()
        X['kill1_mean'] = (rr.kill_cs1.to_numpy() + kd_part) / (rr['round'].to_numpy() + 4)
        mp[a_won] = np.where(np.isnan(done), mm.predict_proba(X[mc])[:, 1], done)
    p_a = np.where(a_ct, p_ct, 1 - p_ct)
    p = p_a * mp[1] + (1 - p_a) * mp[0]
    out.append(te[['game_id', 'round', 'event_seq', 'step', 'kind', 'tier']].assign(month=m, p=p, y=te.map_a_wins.to_numpy(float)))
r = pd.concat(out, ignore_index=True)
r.to_parquet(f'{C}/e5_partial.parquet')
cmp(r, base, 'res_kill_partial_mcw200')
