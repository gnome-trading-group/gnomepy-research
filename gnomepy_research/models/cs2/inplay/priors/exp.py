"""Fast experiment loop for prior variants: the round model is fitted once per month and cached, then each variant
only refits its prior and the map model. Scores every state like the harness, paired against the benchmark prior on
the same round model.

    python -m gnomepy_research.models.cs2.inplay.priors.exp --months 2026-08 --variants bench v1 ..."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.inplay import baseline
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, round_table
from gnomepy_research.models.cs2.inplay.harness import ll, month_split
from gnomepy_research.models.cs2.inplay.priors import core, ratings

CACHE = Path(__file__).resolve().parents[5] / ".runs" / "exp_cache"


def round_preds(m, train, ctx, states):
    f = CACHE / f"pct_{m}.parquet"
    if f.exists():
        return pd.read_parquet(f).p_ct.to_numpy()
    model = baseline.fit(train, None, 0)
    x, _ = baseline.round_rows(ctx, states)
    p = model["round_model"].predict_proba(x[baseline.ROUND])[:, 1]
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(p_ct=p)).to_parquet(f)
    return p


def score(map_model, cols, rt, states, p_ct, extra=None):
    keys = pd.MultiIndex.from_frame(states[["game_id", "round"]])
    rr = rt.loc[keys.unique()].reset_index()
    m = {}
    for a_won in (1, 0):
        X, done = baseline.next_round(rr, a_won)
        if extra is not None:
            X = X.assign(**{c: rr[c].to_numpy() for c in extra})
        p = map_model.predict_proba(X[cols])[:, 1]
        m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[["game_id", "round"]]))
    side = rt.side_a.reindex(keys).to_numpy()
    p_a = np.where(side == "CT", p_ct, 1 - p_ct)
    return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy()


def run_variant(name, train, ctx, states, asof, cfg: core.Config, seed=0):
    mdl = core.Model(cfg)
    mdl.params = ratings.Params(**cfg_params(name))
    info = ratings.game_info(train)
    feats, st = ratings.run(info, mdl.params)
    g = core.map_frame(info, feats, train)
    prior_tr = mdl.prior_table(g, mdl.fit_prior(g, train, seed))
    info2 = ratings.game_info(ctx, set(ctx.games.game_id) - st.done)
    feats2, _ = ratings.run(info2, mdl.params, st.copy(), floor=asof)
    g2 = core.map_frame(info2, feats2, ctx)
    prior_te = mdl.prior_table(g2[g2.game_id.isin(set(states.game_id))])
    legal = legal_maps(train)
    rt = core.with_prior(round_table(train.subset(legal)), prior_tr)
    rt = rt[rt.map_a_wins.notna()]
    cols = mdl.map_cols()
    mm = baseline.trees(seed).fit(rt[cols], rt.map_a_wins)
    rte = core.with_prior(round_table(ctx.subset(states.game_id)), prior_te).set_index(["game_id", "round"])
    return mm, cols, rte


def cfg_params(name):
    return {}


VARIANTS = {
    "v1": core.Config(),
    "edge_map": core.Config(edge_map=True),
    "nomk": core.Config(market_prior=False),
    "all25": core.Config(prior_from="2025-01-01"),
    "jun25": core.Config(prior_from="2025-06-01"),
    "p2": core.Config(nm_set="p2"),
    "x1": core.Config(nm_set="x1", nm_kind="xgb"),
    "x2": core.Config(nm_set="x2", nm_kind="xgb"),
    "avg_v1": core.Config(market_mode="avg"),
    "avg_p2": core.Config(market_mode="avg", nm_set="p2"),
    "avg_x1": core.Config(market_mode="avg", nm_set="x1", nm_kind="xgb"),
    "x1nooof": core.Config(nm_set="x1", nm_kind="xgb", folds=1),
}
SEEDS = {"v1s1": ("v1", 1), "v1s2": ("v1", 2), "nomks1": ("nomk", 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=["2026-08"])
    ap.add_argument("--variants", nargs="*", default=["v1"])
    a = ap.parse_args()
    b = Bundle.load(Path.home() / "gnome-research-work/cs2/comp3/data")
    res = {}
    for m in a.months:
        asof, train, ctx, states, y = month_split(b, m)
        t0 = time.time()
        p_ct = round_preds(m, train, ctx, states)
        bm = baseline.trees
        brt = baseline.with_prior(round_table(train.subset(legal_maps(train))))
        brt = brt[brt.map_a_wins.notna()]
        bmm = bm(0).fit(brt[baseline.MAPCOLS], brt.map_a_wins)
        _, brte = baseline.round_rows(ctx, states)
        pb = score(bmm, baseline.MAPCOLS, brte, states, p_ct)
        lb = ll(pb, y)
        print(f"{m}: bench {lb.mean():.4f} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        for v in a.variants:
            base, seed = SEEDS.get(v, (v, 0))
            mm, cols, rte = run_variant(v, train, ctx, states, asof, VARIANTS[base], seed)
            extra = [c for c in cols if c not in baseline.MAPCOLS]
            p = score(mm, cols, rte, states, p_ct, extra)
            d = ll(p, y) - lb
            start = states["round"].to_numpy() <= 6
            res.setdefault(v, []).append((m, d.sum(), len(d), d[start].mean()))
            print(f"  {m} {v}: diff {d.mean():+.5f}  rounds1-6 {d[start].mean():+.5f}", file=sys.stderr, flush=True)
    for v, r in res.items():
        tot = sum(x[1] for x in r) / sum(x[2] for x in r)
        print(f"{v}: pooled {tot:+.5f}  per-month " + " ".join(f"{x[0]}:{x[1] / x[2]:+.5f}" for x in r))


if __name__ == "__main__":
    main()
