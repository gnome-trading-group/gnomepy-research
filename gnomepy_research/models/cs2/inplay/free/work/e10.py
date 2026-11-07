import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from wc import setup, walk
rt, base, slim = setup()
S6 = ["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean"]
P6 = ["res_mean", "kill1_mean", "res_ct_mean", "res_t_mean", "kill1_ct_mean", "kill1_t_mean"]
cfg = dict(map_extra=S6, map_kw=dict(min_child_weight=200), shrink=4.0, expect='rich')
walk(rt, base, slim, {**cfg, 'map_extra': P6}, label='rich partial', save='e10_partial')
walk(rt, base, slim, {**cfg, 'map_extra': S6 + ['tier_code']}, label='rich tier', save='e10_tier')
walk(rt, base, slim, cfg, label='rich w2026x3', save='e10_w2026', wfn=lambda t: np.where(t.match_begin >= pd.Timestamp('2026-01-01', tz='UTC'), 3.0, 1.0))
