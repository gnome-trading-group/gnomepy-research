"""State-level continuation: E[map win | state now, round outcome] trained on in-round states (2026 timelines), with
the round-level economy continuation's logit as a base margin, so survivors and equipment alive can shape the next
round's prospects."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from gnomepy_research.models.cs2.inplay import baseline
from gnomepy_research.models.cs2.inplay.rounddyn.exp import CACHE, ll
from gnomepy_research.models.cs2.inplay.rounddyn.exp2 import ECON, build, cont_rows

STATE = ["a_alive", "b_alive", "a_eq_alive", "b_eq_alive", "planted", "t", "a_awp_alive", "b_awp_alive"]


def orient(s):
    a_ct = (s.side_a == "CT").to_numpy()
    o = {}
    for k, (c, t) in dict(alive=("ct_alive", "t_alive"), eq_alive=("ct_eq", "t_eq"), awp_alive=("ct_awp", "t_awp")).items():
        o[f"a_{k}"] = np.where(a_ct, s[c], s[t]); o[f"b_{k}"] = np.where(a_ct, s[t], s[c])
    return s.assign(**o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=["2026-08"])
    ap.add_argument("--pred", default="dyn_prior")
    ap.add_argument("--mode", default="margin")
    a = ap.parse_args()
    rt = build()
    st = pd.read_parquet(CACHE / "states.parquet")
    st = orient(st)
    for m in a.months:
        t0 = time.time()
        start = pd.Timestamp(m + "-01", tz="UTC")
        tr = rt[(rt.end < start)].reset_index(drop=True)
        rows = []
        for w in (1, 0):
            X, done = cont_rows(tr, w)
            keep = ((tr.a_wins == w) & np.isnan(done)).to_numpy()
            rows.append((X[keep], tr.map_a_wins[keep]))
        cols = baseline.MAPCOLS + ECON
        X = pd.concat([r[0] for r in rows]); y = pd.concat([r[1] for r in rows])
        rmdl = baseline.trees(0, n_jobs=4).fit(X[cols], y)

        def state_rows(s, w):
            ri = rt.set_index(["game_id", "round"]).loc[pd.MultiIndex.from_frame(s[["game_id", "round"]])].reset_index()
            X, done = cont_rows(ri, w)
            X = X.reset_index(drop=True)
            margin = np.log(np.clip(rmdl.predict_proba(X[cols])[:, 1], 1e-4, 1 - 1e-4))
            margin = margin - np.log(np.clip(1 - np.exp(margin), 1e-4, 1))
            X = X.assign(**{c: s[c].to_numpy() for c in STATE})
            return X, done, margin

        trs = st[(st.end < start) & st.y_ct.notna()]
        trs = trs[trs.event_seq % 1 == 0]
        Xs, ys, ms = [], [], []
        for w in (1, 0):
            sel = (trs.a_wins == w).to_numpy()
            X, done, mg = state_rows(trs[sel], w)
            k = np.isnan(done)
            Xs.append(X[k]); ys.append(trs[sel].map_a_wins.to_numpy()[k]); ms.append(mg[k])
        X = pd.concat(Xs); y = np.concatenate(ys); mg = np.concatenate(ms)
        scols = cols + STATE
        smdl = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                 min_child_weight=200, reg_lambda=5, n_jobs=4, random_state=0)
        smdl.fit(X[scols], y, base_margin=mg)
        te = st[(st.match_begin >= start) & (st.match_begin < start + pd.offsets.MonthBegin(1))]
        te = te.merge(pd.read_parquet(CACHE / f"pred_{a.pred}_{m}.parquet"), on=["game_id", "event_seq"])
        pr, pr0 = {}, {}
        for w in (1, 0):
            X, done, mg = state_rows(te, w)
            p = smdl.predict_proba(X[scols], base_margin=mg)[:, 1]
            p0 = 1 / (1 + np.exp(-mg))
            pr[w] = np.where(np.isnan(done), p, done); pr0[w] = np.where(np.isnan(done), p0, done)
        pa = np.where(te.side_a == "CT", te.p_ct, 1 - te.p_ct)
        y = te.map_a_wins.to_numpy()
        new = pa * pr[1] + (1 - pa) * pr[0]
        old = pa * pr0[1] + (1 - pa) * pr0[0]
        print(m, "round-cont %.5f state-cont %.5f (%ds)" % (ll(old, y).mean(), ll(new, y).mean(), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
