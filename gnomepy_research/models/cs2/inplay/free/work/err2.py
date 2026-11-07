import sys, numpy as np, pandas as pd
sys.path.insert(0, 'gnomepy_research/models/cs2/inplay/free/work')
from exp import load, C
from gnomepy_research.models.cs2.inplay.harness import ll
pd.set_option('display.width', 200)
rt, st = load()
r = pd.read_parquet(f'{C}/rep_all.parquet')
x = r.merge(st[['game_id','event_seq','y_ct','ct_alive','t_alive','planted','t','since_plant','ct_eq_start','t_eq_start','ct_kits']], on=['game_id','event_seq'])
x['rl'] = ll(x.p_ct, x.y_ct)
x['ad'] = x.ct_alive.astype(int).astype(str)+'v'+x.t_alive.astype(int).astype(str)
t = x.groupby('ad').agg(n=('rl','size'), ll=('rl','mean'), p=('p_ct','mean'), y=('y_ct','mean'))
print(t[t.n>2000])
x['tb'] = pd.cut(x.t, [-1,0,20,40,60,80,100,200])
print(x.groupby(['planted','tb'],observed=True).agg(n=('rl','size'), ll=('rl','mean'), p=('p_ct','mean'), y=('y_ct','mean')))
print(x.groupby('tier').agg(n=('rl','size'), ll=('rl','mean'), p=('p_ct','mean'), y=('y_ct','mean')))
# odd economies
x['odd'] = (x.ct_eq_start<1000)|(x.t_eq_start<1000)|(x.ct_eq_start>45000)|(x.t_eq_start>45000)
print(x.groupby('odd').agg(n=('rl','size'), ll=('rl','mean'), p=('p_ct','mean'), y=('y_ct','mean')))
# map-level: decompose map loss at start by round
