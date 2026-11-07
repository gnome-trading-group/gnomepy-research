"""Survivor-aware continuation: M(next score, economy, survivors of each team) from rounds, integrated over the
survivor distribution P(survivors | state now, who wins the round) from multiclass models on in-round states."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from gnomepy_research.models.cs2.inplay import baseline
from gnomepy_research.models.cs2.inplay.rounddyn.exp import CACHE, SETS, add_derived, ll
from gnomepy_research.models.cs2.inplay.rounddyn.exp2 import ECON, build, cont_rows

CONT = baseline.MAPCOLS + ECON + ["a_cash_next", "b_cash_next", "a_resid_prev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=["2026-08"])
    ap.add_argument("--pred", default="dyn_form")
    ap.add_argument("--surv_set", default="dyn_form")
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--clip", type=int, default=1)
    a = ap.parse_args()
    kw = json.loads(a.kw)
    rt = build()
    st = add_derived(pd.read_parquet(CACHE / "states.parquet"))
    st = st.merge(rt[["game_id", "round", "a_alive_end", "b_alive_end"]], on=["game_id", "round"], how="left")
    a_ct = (st.side_a == "CT").to_numpy()
    st["ct_alive_end"] = np.where(a_ct, st.a_alive_end, st.b_alive_end).clip(0, 5)
    st["t_alive_end"] = np.where(a_ct, st.b_alive_end, st.a_alive_end).clip(0, 5)
    sfe = SETS[a.surv_set] + ["ct_won"]
    for m in a.months:
        t0 = time.time()
        start = pd.Timestamp(m + "-01", tz="UTC")
        tr = rt[(rt.end < start)].reset_index(drop=True)
        Xs, ys, Xs0, ys0 = [], [], [], []
        for w in (1, 0):
            X, done = cont_rows(tr, w)
            keep = ((tr.a_wins == w) & np.isnan(done)).to_numpy()
            Xk = X[keep].assign(a_alive_end=tr.a_alive_end[keep].to_numpy(), b_alive_end=tr.b_alive_end[keep].to_numpy())
            Xs.append(Xk); ys.append(tr.map_a_wins[keep])
        X = pd.concat(Xs); y = pd.concat(ys)
        cols_s = CONT + ["a_alive_end", "b_alive_end"]
        cmod = baseline.trees(0, n_jobs=4).fit(X[cols_s], y)
        cmod0 = baseline.trees(0, n_jobs=4).fit(X[CONT], y)
        trs = st[(st.end < start) & st.y_ct.notna() & st.ct_alive_end.notna()].assign(ct_won=lambda d: d.y_ct)
        params = dict(n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=50, n_jobs=4, random_state=0, objective="multi:softprob", num_class=6)
        params.update(kw)
        smod = {s: xgb.XGBClassifier(**params).fit(trs[sfe], trs[f"{s}_alive_end"].astype(int)) for s in ("ct", "t")}
        print(m, "fit %ds" % (time.time() - t0), flush=True)
        te = st[(st.match_begin >= start) & (st.match_begin < start + pd.offsets.MonthBegin(1))]
        te = te.merge(pd.read_parquet(CACHE / f"pred_{a.pred}_{m}.parquet"), on=["game_id", "event_seq"])
        a_ct = (te.side_a == "CT").to_numpy()
        rr = rt.set_index(["game_id", "round"]).loc[pd.MultiIndex.from_frame(te[["game_id", "round"]].drop_duplicates())].reset_index()
        rkey = pd.MultiIndex.from_frame(rr[["game_id", "round"]])
        tkey = pd.MultiIndex.from_frame(te[["game_id", "round"]])
        pos = rkey.get_indexer(tkey)
        out, out0 = {}, {}
        for w in (1, 0):
            Xr, done = cont_rows(rr, w)
            grid = np.zeros((len(rr), 6, 6))
            for k in range(6):
                for j in range(6):
                    grid[:, k, j] = cmod.predict_proba(Xr.assign(a_alive_end=k, b_alive_end=j)[cols_s])[:, 1]
            m0 = cmod0.predict_proba(Xr[CONT])[:, 1]
            ct_won = np.where(a_ct, w, 1 - w)
            Xt = te[SETS[a.surv_set]].assign(ct_won=ct_won)
            pc = smod["ct"].predict_proba(Xt[sfe]); pt = smod["t"].predict_proba(Xt[sfe])
            if a.clip:
                ks = np.arange(6)[None, :]
                pc = np.where(ks <= te.ct_alive.to_numpy()[:, None], pc, 0); pc /= pc.sum(1, keepdims=True)
                pt = np.where(ks <= te.t_alive.to_numpy()[:, None], pt, 0); pt /= pt.sum(1, keepdims=True)
            pa_s = np.where(a_ct[:, None], pc, pt); pb_s = np.where(a_ct[:, None], pt, pc)
            g = grid[pos]
            ev = np.einsum("nk,nkj,nj->n", pa_s, g, pb_s)
            d = done[pos]
            evs = pd.Series(np.where(te.step.to_numpy() == 0, ev, np.nan)).groupby(pos).transform("max").to_numpy()
            ev = np.where(np.isnan(evs), m0[pos], m0[pos] + ev - evs)
            out[w] = np.where(np.isnan(d), ev, d); out0[w] = np.where(np.isnan(d), m0[pos], d)
        pa = np.where(a_ct, te.p_ct, 1 - te.p_ct)
        yv = te.map_a_wins.to_numpy()
        base = pd.read_parquet(CACHE / f"pred_base_{m}.parquet")
        sb = te[["game_id", "event_seq", "side_a", "m1", "m0", "map_a_wins"]].merge(base, on=["game_id", "event_seq"])
        pb = np.where(sb.side_a == "CT", sb.p_ct, 1 - sb.p_ct)
        bench = ll(pb * sb.m1 + (1 - pb) * sb.m0, sb.map_a_wins).mean()
        lnew = ll(pa * out[1] + (1 - pa) * out[0], yv); lold = ll(pa * out0[1] + (1 - pa) * out0[0], yv)
        print(pd.Series(lnew - lold).groupby(te.step.clip(upper=8).to_numpy()).mean().round(5).to_dict())
        new, old = lnew.mean(), lold.mean()
        print(m, a.kw, "bench %.5f econ4 %.5f (%.5f) surv %.5f (%.5f) %ds" % (bench, old, old - bench, new, new - bench,
                                                                              time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
