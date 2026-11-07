"""Round model trained with weights |M(win) - M(lose)|: capacity where the round matters for the map price. Scored
through the econ4 continuation."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from gnomepy_research.models.cs2.inplay import baseline
from gnomepy_research.models.cs2.inplay.rounddyn.exp import CACHE, SETS, add_derived, fit_round, ll
from gnomepy_research.models.cs2.inplay.rounddyn.exp2 import build, cont_rows

CONT = baseline.MAPCOLS + ["a_money", "b_money", "a_equip", "b_equip", "a_money_pre", "b_money_pre", "a_awp", "b_awp",
                           "a_rifles", "b_rifles", "a_cash_next", "b_cash_next", "a_resid_prev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=["2026-08"])
    ap.add_argument("--eps", nargs="*", type=float, default=[0.05])
    ap.add_argument("--set", default="dyn_form")
    a = ap.parse_args()
    rt = build()
    st = add_derived(pd.read_parquet(CACHE / "states.parquet"))
    for m in a.months:
        start = pd.Timestamp(m + "-01", tz="UTC")
        tr = rt[(rt.end < start)].reset_index(drop=True)
        Xs, ys = [], []
        for w in (1, 0):
            X, done = cont_rows(tr, w)
            keep = ((tr.a_wins == w) & np.isnan(done)).to_numpy()
            Xs.append(X[keep]); ys.append(tr.map_a_wins[keep])
        cm = baseline.trees(0, n_jobs=4).fit(pd.concat(Xs)[CONT], pd.concat(ys))
        idx = pd.MultiIndex.from_frame(rt[["game_id", "round"]])
        mm = {}
        for w in (1, 0):
            X, done = cont_rows(rt, w)
            mm[w] = pd.Series(np.where(np.isnan(done), cm.predict_proba(X[CONT])[:, 1], done), index=idx)
        k = pd.MultiIndex.from_frame(st[["game_id", "round"]])
        s = st.assign(c1=mm[1].reindex(k).to_numpy(), c0=mm[0].reindex(k).to_numpy())
        trs = s[(s.end < start) & s.y_ct.notna()]
        te = s[(s.match_begin >= start) & (s.match_begin < start + pd.offsets.MonthBegin(1))]
        for eps in [None] + a.eps:
            t0 = time.time()
            wts = None if eps is None else (np.abs(trs.c1 - trs.c0) + eps).to_numpy()
            mdl = fit_round(trs, SETS[a.set]) if wts is None else \
                xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                                    colsample_bytree=0.8, min_child_weight=50 * np.mean(wts), reg_lambda=5,
                                                    n_jobs=4, random_state=0).fit(trs[SETS[a.set]], trs.y_ct, sample_weight=wts)
            p = mdl.predict_proba(te[SETS[a.set]])[:, 1]
            p = np.where(te.over == 1, np.where(te.over_ct_won.notna(), te.over_ct_won, p), p)
            pa = np.where(te.side_a == "CT", p, 1 - p)
            pm = pa * te.c1 + (1 - pa) * te.c0
            print(m, eps, "round %.5f map %.5f (%ds)" % (ll(p, te.y_ct).mean(), ll(pm, te.map_a_wins).mean(), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
