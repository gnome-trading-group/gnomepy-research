import sys
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from wc import setup, walk
rt, base, slim = setup()
S6 = ["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean"]
cfg = dict(map_extra=S6 + ['res_dec', 'kill_dec'], map_kw=dict(min_child_weight=200), shrink=4.0, expect='rich', decay=0.2)
walk(rt, base, slim, cfg, label='side6 + decay0.2', save='e16_d2')
walk(rt, base, slim, {**cfg, 'decay': 0.4}, seed=1, label='side6 + decay0.4 seed1', save='e16_d4s1')
walk(rt, base, slim, {**cfg, 'decay': 0.4, 'map_extra': S6 + ['res_dec']}, label='side6 + res_dec0.4 only', save='e16_d4r')
