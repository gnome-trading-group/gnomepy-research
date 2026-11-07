"""Shared research prep: benchmark caches plus in-map form features on rt and st."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from exp import load, C
from feat1 import add_hist
from gnomepy_research.models.cs2.inplay.baseline import next_round


def prep():
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
    # per-side residuals from A's view
    res = rt.a_wins - rt.p_exp_a
    for side, flag in (('ct', 1), ('t', 0)):
        v = res.where(rt.ct_a == flag, 0.0)
        n = (rt.ct_a == flag).astype(float)
        rt[f'res_{side}_sum'] = v.groupby(g).cumsum() - v
        rt[f'n_{side}'] = n.groupby(g).cumsum() - n
        rt[f'res_{side}_mean'] = rt[f'res_{side}_sum'] / (rt[f'n_{side}'] + 4)
    # per-side kill form, lagged a round like kill_mean
    for side, flag in (('ct', 1), ('t', 0)):
        v = kd.where(rt.ct_a == flag, 0.0); n = (rt.ct_a == flag).astype(float)
        cv, cn = v.groupby(g).cumsum(), n.groupby(g).cumsum()
        rt[f'kill_{side}_mean'] = np.where(has_ev, cv.groupby(g).shift(2).fillna(0), np.nan) / (cn.groupby(g).shift(2).fillna(0) + 4)
        rt[f'kill_{side}_next'] = np.where(has_ev, cv.groupby(g).shift(1).fillna(0), np.nan) / (cn.groupby(g).shift(1).fillna(0) + 4)
    j = st.join(rt.set_index(['game_id', 'round'])[['ct_a', 'res_ct_mean', 'res_t_mean']], on=['game_id', 'round'])
    st['ctteam_ct_form'] = np.where(j.ct_a == 1, j.res_ct_mean, -j.res_t_mean)
    st['tteam_t_form'] = np.where(j.ct_a == 1, -j.res_ct_mean, j.res_t_mean)
    return rt, st


def nr_ext(cols):
    def f(rr, a_won):
        X, done = next_round(rr, a_won)
        rs = rr.res_sum + a_won - rr.p_exp_a
        for c in cols:
            if c == 'res_mean': X[c] = (rs / (rr.n_prev + 5)).to_numpy()
            elif c.startswith('kill_') and c.endswith('_mean'): X[c] = rr[c.replace('_mean', '_next')].to_numpy()
            elif c in ('res_ct_mean', 'res_t_mean'):
                side = c.split('_')[1]; flag = 1 if side == 'ct' else 0
                cur = (rr.ct_a == flag).astype(float)
                X[c] = ((rr[f'res_{side}_sum'] + cur * (a_won - rr.p_exp_a)) / (rr[f'n_{side}'] + cur + 4)).to_numpy()
            else: X[c] = rr[c].to_numpy()
        return X, done
    return f
