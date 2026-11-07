import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free')
from wc import setup, walk
import core
from exp import C
rt, base, slim = setup()
core.CFG.update(expect='rich', nop_fix=False, alive_end_kills=False)
expect = core.fit_expect(rt[rt.end < pd.Timestamp('2026-01-01', tz='UTC')])
rt = rt.sort_values(['game_id', 'round'])
p_ct = core.expect_ct(expect, rt)
pa = np.where(rt.ct_a == 1, p_ct, 1 - p_ct)
er = pd.read_parquet(f'{C}/ev_round.parquet').set_index(['game_id', 'round'])
kd = rt.join(er, on=['game_id', 'round']).eval('b_dead - a_dead')
kd = kd.fillna(rt.a_alive_end - rt.b_alive_end)
gs = pd.DataFrame(dict(game_id=rt.game_id, res=rt.a_wins - pa, kd=kd, n=1)).groupby('game_id').sum()
gs = gs.join(rt.groupby('game_id').map_a_wins.first())
G = slim.games.set_index('game_id')[['match_id', 'map_no', 'begin', 'end']].join(gs, how='inner').reset_index()
rows = []
for mid, x in G.sort_values(['match_id', 'map_no']).groupby('match_id'):
    if len(x) == 1:
        continue
    for i in range(1, len(x)):
        cur = x.iloc[i]; prev = x.iloc[:i]; prev = prev[prev.end < cur.begin]
        if len(prev):
            rows.append((cur.game_id, (2 * prev.map_a_wins - 1).sum(), prev.res.sum() / (prev.n.sum() + 10), prev.kd.sum() / (prev.n.sum() + 10), len(prev)))
S = pd.DataFrame(rows, columns=['game_id', 'ser_maps', 'ser_res', 'ser_kill', 'ser_n']).set_index('game_id')
for c in S.columns:
    rt[c] = rt.game_id.map(S[c]).fillna(0.0)
S6 = ["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean"]
cfg = dict(map_extra=S6, map_kw=dict(min_child_weight=200), shrink=4.0, expect='rich')
walk(rt, base, slim, {**cfg, 'map_extra': S6 + ['ser_maps', 'ser_res', 'ser_kill', 'ser_n']}, label='side6 + series form', save='e13_series')
walk(rt, base, slim, {**cfg, 'map_extra': S6 + ['ser_maps', 'ser_n']}, label='side6 + series maps only', save='e13_sermaps')
