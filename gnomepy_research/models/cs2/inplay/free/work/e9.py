import sys
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from wc import setup, walk
rt, base, slim = setup()
S6 = ["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean"]
base_cfg = dict(map_extra=S6, map_kw=dict(min_child_weight=200), shrink=4.0, expect='basic')
walk(rt, base, slim, base_cfg, label='core side6 (check)', save='e9_side6')
walk(rt, base, slim, {**base_cfg, 'shrink': 2.0}, label='shrink2', save='e9_sh2')
walk(rt, base, slim, {**base_cfg, 'shrink': 8.0}, label='shrink8', save='e9_sh8')
walk(rt, base, slim, {**base_cfg, 'expect': 'rich'}, label='expect rich', save='e9_rich')
