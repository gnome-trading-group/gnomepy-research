import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from prep import prep, nr_ext
from exp import walk, cmp, C, MONTHS
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND
rt, st = prep()
base = pd.read_parquet(f'{C}/rep_all.parquet')
j = st.join(rt.set_index(['game_id', 'round'])[['ct_a', 'kill_ct_next', 'kill_t_next', 'kill_next', 'res_mean']], on=['game_id', 'round'])
a = j.ct_a == 1
st['ctk_ct'] = np.where(a, j.kill_ct_next, -j.kill_t_next)
st['tk_t'] = np.where(a, -j.kill_ct_next, j.kill_t_next)
st['ctk_all'] = np.where(a, j.kill_next, -j.kill_next)
st['ct_res'] = np.where(a, j.res_mean, -j.res_mean)
cols = ['res_mean', 'kill_mean', 'res_ct_mean', 'res_t_mean', 'kill_ct_mean', 'kill_t_mean']
kw = dict(mapcols=MAPCOLS + cols, nr=nr_ext(cols), map_kw=dict(min_child_weight=200), verbose=False)
r = walk(rt, st, MONTHS, roundcols=ROUND + ['ctk_ct', 'tk_t', 'ctk_all', 'ct_res'], **kw); r.to_parquet(f'{C}/e18_rk.parquet'); cmp(r, base, 'side6 map + kill form in round model')
