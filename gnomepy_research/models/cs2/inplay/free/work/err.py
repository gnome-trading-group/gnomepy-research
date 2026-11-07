import numpy as np, pandas as pd
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.harness import ll
b = Bundle.load('/Users/mprey/gnome-research-work/cs2/comp3/data')
p = pd.read_parquet('/Users/mprey/gnome-research-work/cs2/comp3/bench_cache/bench_89bd1ebdec_2026-08.parquet')
p = p.merge(b.games[['game_id','tier','bo','map_no']], on='game_id').merge(b.priors[['game_id','p_market']], on='game_id', how='left')
p['l'] = ll(p.p, p.y)
print(p.l.mean(), len(p), p.game_id.nunique())
for c in ['tier','kind','bo','map_no']:
    print(p.groupby(c).agg(n=('l','size'), ll=('l','mean'), p=('p','mean'), y=('y','mean')))
p['hasm'] = p.p_market.notna()
print(p.groupby('hasm').l.agg(['size','mean']))
p['bin'] = pd.cut(p.p, np.linspace(0,1,11))
print(p.groupby('bin').agg(n=('y','size'), p=('p','mean'), y=('y','mean')))
p['rb'] = pd.cut(p['round'], [0,1,3,6,12,13,18,24,60])
print(p.groupby('rb').agg(n=('l','size'), ll=('l','mean')))
# calibration by tier: logit slope
from sklearn.linear_model import LogisticRegression
for t,g in p.groupby('tier'):
    z = np.log(np.clip(g.p,1e-4,1-1e-4)/(1-np.clip(g.p,1e-4,1-1e-4)))
    lr = LogisticRegression(C=1e6).fit(z.to_numpy()[:,None], g.y)
    print(t, lr.coef_, lr.intercept_)
