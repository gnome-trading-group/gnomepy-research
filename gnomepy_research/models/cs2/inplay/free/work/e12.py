import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from wc import setup, walk
rt, base, slim = setup()
S6 = ["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean"]
cfg = dict(map_extra=S6, map_kw=dict(min_child_weight=200), shrink=4.0, expect='rich', alive_end_kills=False, nop_fix=True)
walk(rt, base, slim, cfg, label='rich nop_fix', save='e12_nop')
walk(rt, base, slim, {**cfg, 'map_kw': dict(min_child_weight=400)}, label='rich nop_fix mcw400', save='e12_nop400')
