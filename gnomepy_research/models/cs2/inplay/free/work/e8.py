import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from prep import prep, nr_ext
from exp import walk, cmp, C, MONTHS
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND
rt, st = prep()
base = pd.read_parquet(f'{C}/rep_all.parquet')
def go(name, cols, **extra):
    kw = dict(mapcols=MAPCOLS + cols, nr=nr_ext(cols), map_kw=dict(min_child_weight=200), verbose=False); kw.update(extra)
    r = walk(rt, st, MONTHS, **kw); r.to_parquet(f'{C}/e8_{name}.parquet'); cmp(r, base, name)
S = ['res_ct_mean', 'res_t_mean', 'kill_ct_mean', 'kill_t_mean']
go('sideonly4', S)
go('side6_plant', ['res_mean', 'kill_mean'] + S + ['plant_ct_mean', 'plant_t_mean'])
go('side6_open', ['res_mean', 'kill_mean'] + S + ['open_ct_mean', 'open_t_mean'])
