import sys
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from wc import setup, walk
rt, base, slim = setup()
S6 = ["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean"]
cfg = dict(map_extra=S6 + ['res_dec', 'kill_dec'], map_kw=dict(min_child_weight=200), shrink=4.0, expect='rich', decay=0.2)
walk(rt, base, slim, {**cfg, 'map_extra': cfg['map_extra'] + ['tier_code']}, label='side6 + dec0.2 + tier', save='e20_tier')
walk(rt, base, slim, {**cfg, 'nop_fix': True}, label='side6 + dec0.2 + nop_fix', save='e20_nop')
