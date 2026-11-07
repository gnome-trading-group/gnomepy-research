import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

TIER = {'s': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}


def exp_feats(rt):
    ct_eq = np.where(rt.ct_a == 1, rt.a_equip, rt.b_equip); t_eq = np.where(rt.ct_a == 1, rt.b_equip, rt.a_equip)
    X = np.c_[np.log1p(ct_eq / 1000), np.log1p(t_eq / 1000), np.log1p(ct_eq / 1000) * np.log1p(t_eq / 1000),
              np.eye(11)[rt.map_code.to_numpy() + 1]]
    return X


def add_hist(rt, fit_mask):
    rt = rt.sort_values(['game_id', 'round']).copy()
    X = exp_feats(rt)
    y_ct = np.where(rt.ct_a == 1, rt.a_wins, 1 - rt.a_wins)
    lr = LogisticRegression(C=10, max_iter=2000).fit(X[fit_mask.loc[rt.index].to_numpy()], y_ct[fit_mask.loc[rt.index].to_numpy()])
    p_ct = lr.predict_proba(X)[:, 1]
    rt['p_exp_a'] = np.where(rt.ct_a == 1, p_ct, 1 - p_ct)
    nxt = rt.groupby('game_id').score_a.shift(-1)
    # a_won from score progression is research-only shorthand; in the entry it comes from the next round's score
    a_won = rt.a_wins
    res = a_won - rt.p_exp_a
    g = rt.game_id
    rt['res_sum'] = res.groupby(g).cumsum() - res
    rt['n_prev'] = rt['round'] - 1
    rt['res_mean'] = rt.res_sum / (rt.n_prev + 4)
    sm = (rt.a_alive_end - rt.b_alive_end)
    rt['surv_sum'] = sm.groupby(g).cumsum() - sm
    rt['last_a_alive'] = rt.groupby(g).a_alive_end.shift(1).fillna(-1)
    rt['last_b_alive'] = rt.groupby(g).b_alive_end.shift(1).fillna(-1)
    rt['tier_code'] = rt.tier.map(TIER).astype(int)
    return rt


def orient(st, rt, cols):
    j = st.join(rt.set_index(['game_id', 'round'])[cols + ['ct_a', 'prior_logit', 'has_prior']], on=['game_id', 'round'])
    s = np.where(j.ct_a == 1, 1, -1)
    for c in cols:
        st['ct_' + c] = j[c].to_numpy() * s if c not in ('tier_code', 'n_prev') else j[c].to_numpy()
    st['ct_prior'] = j.prior_logit.to_numpy() * s
    st['has_prior'] = j.has_prior.to_numpy()
    return st
