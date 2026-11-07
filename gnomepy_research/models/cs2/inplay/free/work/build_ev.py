import numpy as np, pandas as pd
from gnomepy_research.models.cs2.inplay.data import Bundle
C = '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work/cache'
b = Bundle.load('/Users/mprey/gnome-research-work/cs2/comp3/data')
ev = b.events[b.events.type == 'kill'][['game_id', 'round', 'seq', 'victim_team', 'killer_team', 'elapsed']]
ev = ev.merge(b.games[['game_id', 'team_a', 'team_b']], on='game_id')
ev['a_dead'] = (ev.victim_team == ev.team_a).astype(int)
ev['b_dead'] = (ev.victim_team == ev.team_b).astype(int)
ev = ev.sort_values(['game_id', 'seq'])
first = ev.groupby(['game_id', 'round']).head(1)
agg = ev.groupby(['game_id', 'round']).agg(a_dead=('a_dead', 'sum'), b_dead=('b_dead', 'sum'), nk=('seq', 'size'))
agg['open_a'] = first.set_index(['game_id', 'round']).b_dead - first.set_index(['game_id', 'round']).a_dead
pl = b.events[b.events.type == 'bomb_planted'].groupby(['game_id', 'round']).size().rename('planted')
agg = agg.join(pl).fillna({'planted': 0})
agg.reset_index().to_parquet(f'{C}/ev_round.parquet')
r = b.rounds.set_index(['game_id', 'round'])
j = agg.join(r[['a_alive_end', 'b_alive_end']])
print(len(agg), ((5 - j.a_dead) == j.a_alive_end).mean(), ((5 - j.b_dead) == j.b_alive_end).mean())
