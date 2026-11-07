import sys, numpy as np, pandas as pd
sys.path.insert(0, 'gnomepy_research/models/cs2/inplay/free/work')
from exp import C
from gnomepy_research.models.cs2.inplay.harness import ll
f = sys.argv[1]
a = pd.read_parquet(f'{C}/{f}'); b = pd.read_parquet(f'{C}/rep_all.parquet')
d = a.merge(b[['game_id','event_seq','p']], on=['game_id','event_seq'], suffixes=('','_b'))
pr = pd.read_parquet('/Users/mprey/gnome-research-work/cs2/comp3/data/priors.parquet')
d['hasm'] = d.game_id.isin(set(pr.game_id))
d['diff'] = ll(d.p,d.y)-ll(d.p_b,d.y)
print(d.pivot_table(index='tier', columns='month', values='diff', aggfunc='mean').round(4))
print(d.pivot_table(index='hasm', columns='month', values='diff', aggfunc='mean').round(4))
d['rb'] = pd.cut(d['round'], [0,3,6,12,18,24,60])
print(d.pivot_table(index='rb', columns='month', values='diff', aggfunc='mean', observed=True).round(4))
print(d.groupby('month').game_id.nunique())
