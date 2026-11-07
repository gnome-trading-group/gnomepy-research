import time, numpy as np, pandas as pd
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps
from gnomepy_research.models.cs2.inplay.baseline import round_rows
C = '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work/cache'
t0 = time.time()
b = Bundle.load('/Users/mprey/gnome-research-work/cs2/comp3/data')
legal = legal_maps(b)
noev = set(b.games.game_id) - set(b.events.game_id)
ids = legal & noev
bl = b.subset(ids)
s = bl.rounds[bl.rounds.has_players][['game_id', 'round']].copy()
s['event_seq'] = -1; s['elapsed'] = 0; s['step'] = 0; s['kind'] = 'start'; s['ingame_ms'] = 0
s = s.reset_index(drop=True)
parts = []
for chunk in np.array_split(np.array(sorted(ids)), 8):
    ss = s[s.game_id.isin(set(chunk))].reset_index(drop=True)
    x, rti = round_rows(bl.subset(chunk), ss)
    x = x.join(rti[['a_wins', 'map_a_wins']], on=['game_id', 'round'])
    parts.append(x)
x = pd.concat(parts, ignore_index=True)
x['y_ct'] = np.where(x.side_a == 'CT', x.a_wins, 1 - x.a_wins)
x = x.merge(b.games[['game_id', 'end', 'match_begin', 'tier']], on='game_id')
x.to_parquet(f'{C}/starts_noev.parquet')
print(x.shape, x.ct_alive.value_counts().head(), time.time() - t0)
