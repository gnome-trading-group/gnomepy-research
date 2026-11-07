"""Walk-forward research on top of core.py itself (map-model side only; the round model's p_ct is reused from the
benchmark replica since the entry keeps it unchanged)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free')
sys.path.insert(0, '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work')
import core
from exp import C, MONTHS, cmp
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, trees
from gnomepy_research.models.cs2.inplay.data import Bundle


class Slim:
    def __init__(self):
        d = '/Users/mprey/gnome-research-work/cs2/comp3/data'
        ev = pd.read_parquet(f'{d}/events.parquet', columns=['game_id', 'round', 'seq', 'type', 'victim_team', 'killer_team', 'elapsed'])
        self.events = ev[ev.type.isin(['kill', 'bomb_planted', 'round_start'])].reset_index(drop=True)
        self.games = pd.read_parquet(f'{d}/games.parquet')


def setup():
    rt = pd.read_parquet(f'{C}/rt.parquet')
    base = pd.read_parquet(f'{C}/rep_all.parquet')
    sides = pd.read_parquet(f'{C}/states.parquet', columns=['game_id', 'event_seq', 'side_a', 'ct_alive', 't_alive'])
    base = base.merge(sides, on=['game_id', 'event_seq'])
    return rt, base, Slim()


def walk(rt, base, slim, cfg, months=MONTHS, seed=0, label=None, save=None, wfn=None):
    core.CFG.update(cfg)
    extra = core.CFG['map_extra']
    out = []
    for m in months:
        s = pd.Timestamp(m + '-01', tz='UTC'); e = s + pd.offsets.MonthBegin(1)
        expect = core.fit_expect(rt[rt.end < s])
        h = core.history(slim, rt, expect)
        tr = h[h.end < s]
        mm = trees(seed, **core.CFG['map_kw']).fit(tr[MAPCOLS + extra], tr.map_a_wins, sample_weight=wfn(tr) if wfn else None)
        te = base[base.month == m].reset_index(drop=True)
        rr = te[['game_id', 'round']].join(h.set_index(['game_id', 'round']), on=['game_id', 'round'])
        a_ct = (te.side_a == 'CT').to_numpy()
        kd_now = np.where(a_ct, te.ct_alive - te.t_alive, te.t_alive - te.ct_alive)
        p = core.combine(mm, rr, te.p_ct.to_numpy(), a_ct, kd_now)
        out.append(te[['game_id', 'round', 'event_seq', 'step', 'kind', 'tier', 'month', 'y']].assign(p=p))
    r = pd.concat(out, ignore_index=True)
    if save:
        r.to_parquet(f'{C}/{save}.parquet')
    if label:
        cmp(r, base, label)
    return r
