"""Research cache: every scored-style state of every legal 2026 map with the benchmark's round features, plus the
round table of all legal maps. Labels included (research only, never used by the entry)."""
import time
import numpy as np, pandas as pd
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states, round_table
from gnomepy_research.models.cs2.inplay.baseline import with_prior, round_rows

OUT = '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work/cache'
t0 = time.time()
b = Bundle.load('/Users/mprey/gnome-research-work/cs2/comp3/data')
legal = legal_maps(b)
bl = b.subset(legal)
rt = with_prior(round_table(bl))
rt = rt.merge(b.games[['game_id', 'tier', 'match_begin', 'end', 'begin', 'bo', 'map_no', 'team_a', 'team_b']], on='game_id')
rt.to_parquet(f'{OUT}/rt.parquet')
print('rt', len(rt), time.time() - t0, flush=True)
ev_games = set(bl.events.game_id)
st = map_states(bl, ev_games)
print('states', len(st), time.time() - t0, flush=True)
parts = []
gids = np.array(sorted(st.game_id.unique()))
for chunk in np.array_split(gids, 8):
    s = st[st.game_id.isin(set(chunk))].reset_index(drop=True)
    x, rti = round_rows(bl.subset(chunk), s)
    x = x.join(rti[['a_wins', 'map_a_wins', 'score_a', 'score_b']], on=['game_id', 'round'])
    parts.append(x)
    print('chunk', len(x), time.time() - t0, flush=True)
x = pd.concat(parts, ignore_index=True)
x['y_ct'] = np.where(x.side_a == 'CT', x.a_wins, 1 - x.a_wins)
x.to_parquet(f'{OUT}/states.parquet')
print('done', x.shape, time.time() - t0)
