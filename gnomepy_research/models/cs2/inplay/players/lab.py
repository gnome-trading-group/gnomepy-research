"""Fast experiment loop: build every legal map's round-model and map-model rows once (player/team strength as of each
map's begin, which does not depend on the training cutoff), then walk forward by month refitting only the trees.
Approximate (the real harness blanks the scored month's outcomes), used to rank ideas before scoring them properly.

    python -m gnomepy_research.models.cs2.inplay.players.lab build OUT_DIR
    python -m gnomepy_research.models.cs2.inplay.players.lab eval OUT_DIR"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, next_round, trees, with_prior
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states, round_table
from gnomepy_research.models.cs2.inplay.players import entry as E

DATA = "/Users/mprey/gnome-research-work/cs2/comp3/data"
MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    E.CFG["map_prior"] = "elo"
    b = Bundle.load(DATA)
    t0 = time.time()
    hist = E.History(b, E.CFG["half_life"])
    model = dict(hist=hist, pri=E.priors_from(hist))
    r = b.rounds
    model["map_ct"] = pd.Series(np.where(r.side_a == "CT", r.a_wins, 1 - r.a_wins)).groupby(r["map"].to_numpy()).mean().to_dict()
    legal = legal_maps(b)
    sub = b.subset(legal)
    rt = with_prior(round_table(sub))
    rates, team = E.strength(model, sub, rt.game_id.unique())
    rt = pd.concat([rt, E.map_skill(sub, rt, rates, team)], axis=1)
    done = b.games[b.games.map_a_wins.notna()]
    r2, t2 = E.strength(model, b, done.game_id)
    pm = E.premap(model, b, done.game_id, r2, t2).join(b.games.set_index("game_id")[["map_a_wins", "begin", "end"]])
    pm = pm.join(b.priors.set_index("game_id")[["p_market", "p_e12"]])
    pm.to_parquet(out / "pm.parquet")
    rt = rt.merge(b.games[["game_id", "begin", "end", "match_begin"]], on="game_id")
    print("map rows", time.time() - t0, flush=True)
    rt.to_parquet(out / "rt.parquet")
    st = map_states(b, legal)
    model["premap"] = None
    x, _ = E.rows(model, sub, st, rates, team, pm)
    x = x.join(rt.set_index(["game_id", "round"])[["a_wins"]], on=["game_id", "round"])
    x["y_ct"] = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
    x = x.merge(b.games[["game_id", "end", "match_begin"]], on="game_id", how="left")
    x.drop(columns=[c for c in x.columns if x[c].dtype == object and c not in ("side_a", "kind")]).to_parquet(out / "states.parquet")
    print("states", len(x), time.time() - t0, flush=True)


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def price(rmodel, mmodel, rcols, mcols, xs, rt):
    p_ct = rmodel.predict_proba(xs[rcols])[:, 1]
    rr = rt.set_index(["game_id", "round"])
    keys = pd.MultiIndex.from_frame(xs[["game_id", "round"]])
    u = rr.loc[keys.unique()].reset_index()
    m = {}
    for a_won in (1, 0):
        X, done = next_round(u, a_won)
        X = pd.concat([X, u[[c for c in mcols if c not in MAPCOLS]]], axis=1)
        p = mmodel.predict_proba(X[mcols])[:, 1]
        m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(u[["game_id", "round"]]))
    p_a = np.where(xs.side_a == "CT", p_ct, 1 - p_ct)
    return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy(), p_ct


def roster_prior(rt, xs, pm, cutoff, stack):
    """The entry's prior as of cutoff: roster LR where there is no market, market stacked with it where there is."""
    E.CFG["stack"] = stack
    tr = pm[pm.end < cutoff]
    lr = E.fit_premap(tr, tr.map_a_wins)
    z = E.premap_z(pm, lr)
    zz = z.reindex(rt.game_id).to_numpy()
    p = rt.prior_logit.to_numpy().copy()
    use = (rt.has_prior.to_numpy() == 0) & ~np.isnan(zz)
    p[use] = zz[use]
    if stack in E.STACK:
        d = tr.assign(z=z.reindex(tr.index)).dropna(subset=["p_market", "p_e12", "map_a_wins", "z"])
        if len(d) >= 300:
            st = LogisticRegression(C=1.0).fit(E.stack_x(d, d.z), d.map_a_wins)
            ok = (rt.p_market.notna() & rt.p_e12.notna()).to_numpy() & ~np.isnan(zz)
            p[ok] = st.decision_function(E.stack_x(rt[ok], zz[ok]))
    rt = rt.assign(prior_logit=p)
    pl = rt.set_index(["game_id", "round"]).prior_logit.reindex(pd.MultiIndex.from_frame(xs[["game_id", "round"]])).to_numpy()
    return rt, xs.assign(ct_prior=np.where(xs.side_a == "CT", pl, -pl))


def evaluate(d: Path, variants: dict, months=MONTHS, seed=0):
    rt0 = pd.read_parquet(d / "rt.parquet")
    xs0 = pd.read_parquet(d / "states.parquet")
    pm = pd.read_parquet(d / "pm.parquet")
    res = {k: [] for k in variants}
    for m in months:
        start = pd.Timestamp(m + "-01", tz="UTC"); end = start + pd.offsets.MonthBegin(1)
        priors = {"elo": (rt0, xs0)}
        for pr in {v[4] for v in variants.values()} - {"elo"}:
            src, stack = pr.split(":")
            E.CFG["pm"] = src
            priors[pr] = roster_prior(rt0, xs0, pm, start, stack)
        fits = {}
        for name, (rc, mc, rkw, mkw, pr) in variants.items():
            rt, xs = priors[pr]
            rtr = rt[rt.end < start]; xtr = xs[xs.end < start]
            te = (xs.match_begin >= start) & (xs.match_begin < end)
            xte = xs[te]
            rte = rt[rt.game_id.isin(set(xte.game_id))]
            y = rte.groupby("game_id").map_a_wins.first().reindex(xte.game_id).to_numpy(float)
            uses_prior = "ct_prior" in rc
            k_r, k_m = (tuple(rc), tuple(sorted(rkw.items())), pr if uses_prior else ""), (tuple(mc), tuple(sorted(mkw.items())), pr)
            if k_r not in fits:
                fits[k_r] = trees(seed, max_depth=6, learning_rate=0.05, **rkw).fit(xtr[rc], xtr.y_ct)
            if k_m not in fits:
                fits[k_m] = trees(seed, **mkw).fit(rtr[mc], rtr.map_a_wins)
            p, p_ct = price(fits[k_r], fits[k_m], rc, mc, xte, rte)
            res[name].append(pd.DataFrame(dict(month=m, game_id=xte.game_id.to_numpy(), kind=xte.kind.to_numpy(), ll=ll(p, y),
                                               rll=ll(p_ct, xte.y_ct.to_numpy()))))
        print(m, {k: round(v[-1].ll.mean(), 5) for k, v in res.items()}, flush=True)
    return {k: pd.concat(v, ignore_index=True) for k, v in res.items()}


def report(res: dict, base="base"):
    b = res[base]
    rng = np.random.default_rng(0)
    for k, v in res.items():
        d = v.ll - b.ll
        s = pd.DataFrame(dict(d=d, g=v.game_id)).groupby("g").d.agg(["sum", "size"])
        a, n = s["sum"].to_numpy(), s["size"].to_numpy()
        boot = [a[i].sum() / n[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(500))]
        per = pd.Series(d.to_numpy()).groupby(v.month.to_numpy()).mean()
        print(f"{k:28s} ll {v.ll.mean():.5f} diff {d.mean():+.5f} [{np.quantile(boot, .025):+.5f},{np.quantile(boot, .975):+.5f}] "
              f"start {d[v.kind == 'start'].mean():+.5f} rll {v.rll.mean():.5f} ({(v.rll - b.rll).mean():+.5f}) "
              f"months {' '.join(f'{x:+.4f}' for x in per)}", flush=True)


def default_variants():
    R, M, P = ROUND, MAPCOLS, ["ct_prior", "has_prior"]
    return {
        "base": (R, M, {}, {}, "elo"),
        "base:none": (R, M, {}, {}, "base:none"),
        "base:mz": (R, M, {}, {}, "base:mz"),
        "pelo:none": (R, M, {}, {}, "pelo:none"),
        "full:none": (R, M, {}, {}, "full:none"),
        "full:mz": (R, M, {}, {}, "full:mz"),
        "full:mez": (R, M, {}, {}, "full:mez"),
        "full:mz+rskill": (R + E.ROUND_X, M, {}, {}, "full:mz"),
    }


if __name__ == "__main__":
    cmd, d = sys.argv[1], Path(sys.argv[2])
    if cmd == "build":
        build(d)
    else:
        report(evaluate(d, default_variants()))
