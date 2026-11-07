"""Fast walk-forward experiments on the research cache: replicate the benchmark, then swap in extra features."""
import numpy as np, pandas as pd
from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, next_round, trees
from gnomepy_research.models.cs2.inplay.harness import ll

C = '/Users/mprey/gnome-research-work/cs2/comp3/wt/free/gnomepy_research/models/cs2/inplay/free/work/cache'
MONTHS = ['2026-04', '2026-05', '2026-06', '2026-07', '2026-08']


def load():
    rt = pd.read_parquet(f'{C}/rt.parquet')
    st = pd.read_parquet(f'{C}/states.parquet')
    g = rt.groupby('game_id')[['match_begin', 'end', 'tier']].first()
    st = st.join(g, on='game_id')
    return rt, st


def combine(p_ct, x, rt_test, map_model, mapcols=MAPCOLS, nr=next_round):
    keys = pd.MultiIndex.from_frame(x[['game_id', 'round']])
    rr = rt_test.set_index(['game_id', 'round']).loc[keys.unique()].reset_index()
    m = {}
    for a_won in (1, 0):
        X, done = nr(rr, a_won)
        p = map_model.predict_proba(X[mapcols])[:, 1]
        m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[['game_id', 'round']]))
    p_a = np.where(x.side_a == 'CT', p_ct, 1 - p_ct)
    return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy()


def walk(rt, st, months=MONTHS, roundcols=ROUND, mapcols=MAPCOLS, seed=0, round_kw=None, map_kw=None,
         round_w=None, map_model=None, nr=next_round, verbose=True):
    out = []
    for m in months:
        s = pd.Timestamp(m + '-01', tz='UTC'); e = s + pd.offsets.MonthBegin(1)
        rtr = rt[rt.end < s]
        mm = map_model(rtr, seed) if map_model else trees(seed, **(map_kw or {})).fit(rtr[mapcols], rtr.map_a_wins)
        tr = st[st.end < s]
        kw = dict(max_depth=6, learning_rate=0.05); kw.update(round_kw or {})
        w = round_w(tr) if round_w else None
        rm = trees(seed, **kw).fit(tr[roundcols], tr.y_ct, sample_weight=w)
        te = st[(st.match_begin >= s) & (st.match_begin < e)]
        p_ct = rm.predict_proba(te[roundcols])[:, 1]
        rte = rt[(rt.match_begin >= s) & (rt.match_begin < e)]
        p = combine(p_ct, te, rte, mm, mapcols, nr)
        r = te[['game_id', 'round', 'event_seq', 'step', 'kind', 'tier']].assign(month=m, p=p, y=te.map_a_wins.to_numpy(float), p_ct=p_ct)
        if verbose:
            print(m, len(r), round(ll(r.p, r.y).mean(), 5), flush=True)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def cmp(a, b, label=''):
    d = a.merge(b[['game_id', 'event_seq', 'p']], on=['game_id', 'event_seq'], suffixes=('', '_b'))
    diff = ll(d.p, d.y) - ll(d.p_b, d.y)
    s = pd.DataFrame(dict(d=diff, g=d.game_id)).groupby('g').d.agg(['sum', 'size'])
    a_, n_ = s['sum'].to_numpy(), s['size'].to_numpy()
    rng = np.random.default_rng(0)
    boot = [a_[i].sum() / n_[i].sum() for i in (rng.integers(0, len(a_), len(a_)) for _ in range(500))]
    per = pd.Series(diff).groupby(d.month.to_numpy()).mean()
    st_ = diff[d.kind.to_numpy() == 'start'].mean()
    print(f'{label}: diff {diff.mean():+.5f} [{np.quantile(boot, .025):+.5f},{np.quantile(boot, .975):+.5f}] start {st_:+.5f} months', ' '.join(f'{v:+.4f}' for v in per), flush=True)
    return diff.mean()
