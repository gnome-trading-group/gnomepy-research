import sys
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
from wc import setup, walk
rt, base, slim = setup()
S6 = ["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean"]
cfg = dict(map_extra=S6 + ['res_dec', 'kill_dec', 'post_logit'], map_kw=dict(min_child_weight=200), shrink=4.0, expect='rich', decay=0.2)
walk(rt, base, slim, {**cfg, 'post_a': 0.3}, label='final + post_logit a=0.3', save='e21_p3')
walk(rt, base, slim, {**cfg, 'post_a': 0.6}, label='final + post_logit a=0.6', save='e21_p6')
