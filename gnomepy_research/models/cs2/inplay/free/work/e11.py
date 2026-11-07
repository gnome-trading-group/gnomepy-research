import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from wc import setup, walk
rt, base, slim = setup()
S6 = ["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean"]
P6 = ["res_mean", "kill1_mean", "res_ct_mean", "res_t_mean", "kill1_ct_mean", "kill1_t_mean"]
cfg = dict(map_extra=S6, map_kw=dict(min_child_weight=200), shrink=4.0, expect='rich', alive_end_kills=True)
walk(rt, base, slim, cfg, label='rich alive_end_kills', save='e11_ae')
walk(rt, base, slim, {**cfg, 'map_extra': P6}, label='rich alive_end_kills partial', save='e11_ae_partial')
