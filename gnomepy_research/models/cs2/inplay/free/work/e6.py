import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from prep import prep, nr_ext
from exp import walk, cmp, C, MONTHS
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND
rt, st = prep()
base = pd.read_parquet(f'{C}/rep_all.parquet')
cols = ['res_mean', 'kill_mean']
kw = dict(mapcols=MAPCOLS + cols, nr=nr_ext(cols), map_kw=dict(min_child_weight=200), verbose=False)
for seed in (1, 2):
    r = walk(rt, st, MONTHS, seed=seed, **kw); r.to_parquet(f'{C}/e6_seed{seed}.parquet'); cmp(r, base, f'res_kill_mcw200 seed{seed}')
c2 = ['res_mean', 'kill_mean', 'res_ct_mean', 'res_t_mean']
r = walk(rt, st, MONTHS, **{**kw, 'mapcols': MAPCOLS + c2, 'nr': nr_ext(c2)}); r.to_parquet(f'{C}/e6_side.parquet'); cmp(r, base, 'res_kill_side')
s0 = pd.read_parquet(f'{C}/starts_noev.parquet')
s0 = s0[[c for c in st.columns if c in s0.columns]]
st2 = pd.concat([st, s0], ignore_index=True)
r = walk(rt, st2, MONTHS, **kw); r.to_parquet(f'{C}/e6_starts.parquet'); cmp(r, base, 'res_kill + 2025 starts in round model')
