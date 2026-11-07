"""Continuation experiments: M(next score) learned from round n's own start-of-round economy as well as the next
score state, trained on every round with its actual outcome, against the benchmark's score-only map model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.models.cs2.inplay import baseline
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps
from gnomepy_research.models.cs2.inplay.rounddyn.exp import CACHE, ll

MASK = True
DATA = Path.home() / "gnome-research-work/cs2/comp3/data"
FORM_C = ["a_upsets_prev", "a_gun_wr_prev", "gun_prev", "a_kd_prev", "a_resid_prev", "rounds_prev", "s_maps", "s_w", "s_rd", "a_alive_end", "b_alive_end", "elo_logit", "tier_code", "bo", "a_r500", "a_r1500", "a_r2500", "a_minm", "b_r500", "b_r1500", "b_r2500", "b_minm", "a_map_rwr", "b_map_rwr", "a_all_rwr", "b_all_rwr", "a_map_n", "b_map_n", "a_ct_map", "a_t_map", "b_ct_map", "b_t_map"]
ECON = ["a_money", "b_money", "a_equip", "b_equip", "a_money_pre", "b_money_pre", "a_awp", "b_awp", "a_rifles", "b_rifles"]


def cont_rows(rt: pd.DataFrame, a_won):
    X, done = baseline.next_round(rt, a_won)
    extra = [c for c in ECON + FORM_C if c in rt]
    X = X.assign(**{c: rt[c].to_numpy() for c in extra}, won=a_won)
    if MASK:
        hp = rt.has_players.to_numpy()
        for c in ECON:
            X[c] = np.where(hp, X[c], np.nan)
    def income(ctr, won):
        return np.where(won, 3250.0, 1400.0 + 500.0 * np.clip(ctr - 1, 0, 4)) * 5
    X["a_cash_next"] = X.a_money + income(X.a_loss_ctr.to_numpy(), a_won == 1)
    X["b_cash_next"] = X.b_money + income(X.b_loss_ctr.to_numpy(), a_won == 0)
    return X, done


def build():
    f = CACHE / "rounds.parquet"
    if f.exists():
        return pd.read_parquet(f)
    b = Bundle.load(DATA)
    rt = baseline.with_prior(baseline.round_table(b.subset(legal_maps(b))))
    rt = rt.merge(b.games[["game_id", "end", "match_begin"]], on="game_id")
    rt.to_parquet(f)
    return rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=["2026-08"])
    ap.add_argument("--pred", default="dyn_prior")
    ap.add_argument("--feats", default="econ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--adj", default="")
    ap.add_argument("--nseeds", type=int, default=1)
    a = ap.parse_args()
    rt0 = build()
    st = pd.read_parquet(CACHE / "states.parquet", columns=["game_id", "round", "event_seq", "side_a", "map_a_wins", "m1", "m0"])
    for m in a.months:
        t0 = time.time()
        start = pd.Timestamp(m + "-01", tz="UTC")
        rt = rt0.copy()
        if a.adj:
            adj = a.adj.split(",")
            rt["dmr"] = rt.a_map_rwr - rt.b_map_rwr
            rt["dall"] = rt.a_all_rwr - rt.b_all_rwr
            g1 = rt[(rt["round"] == 1) & (rt.end < start)]
            Xl = lambda z: np.c_[z.prior_logit, z.has_prior, z[adj].to_numpy(), z[adj].to_numpy() * z.has_prior.to_numpy()[:, None]]
            lr = LogisticRegression(C=10, max_iter=1000).fit(Xl(g1), g1.map_a_wins)
            coef = lr.coef_[0]
            rt["prior_logit"] = rt.prior_logit + Xl(rt)[:, 2:] @ coef[2:] / coef[0]
        tr = rt[(rt.end < start)].reset_index(drop=True)
        rows = []
        for w in (1, 0):
            X, done = cont_rows(tr, w)
            keep = (tr.a_wins == w) & np.isnan(done)
            rows.append((X[keep.to_numpy()], tr.map_a_wins[keep.to_numpy()]))
        X = pd.concat([r[0] for r in rows]); y = pd.concat([r[1] for r in rows])
        cols = baseline.MAPCOLS + {"none": [], "econ": ECON, "econ_form": ECON + FORM_C[:3], "econ_kd": ECON + FORM_C[:4], "econ_resid": ECON + FORM_C[:3] + ["a_resid_prev"], "econ_r": ECON + ["a_resid_prev"], "econ2": ECON + FORM_C[:3] + ["a_cash_next", "b_cash_next"], "econ3": ECON + ["a_cash_next", "b_cash_next"], "econ4": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev"], "econ5": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev", "s_maps", "s_w", "s_rd"], "econ6": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev", "s_w"], "econ7": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev", "elo_logit"], "econ8": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev", "tier_code"], "econ9": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev", "a_r500", "a_r1500", "a_r2500", "a_minm", "b_r500", "b_r1500", "b_r2500", "b_minm"], "econ10": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev", "a_map_rwr", "b_map_rwr", "a_all_rwr", "b_all_rwr"], "econ11": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev", "a_map_rwr", "b_map_rwr", "a_all_rwr", "b_all_rwr", "a_map_n", "b_map_n", "a_ct_map", "a_t_map", "b_ct_map", "b_t_map"], "oracle": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev", "a_alive_end", "b_alive_end"]}[a.feats]
        mdls = [baseline.trees(a.seed + i, n_jobs=4, **json.loads(a.kw)).fit(X[cols], y) for i in range(a.nseeds)]
        te = rt[(rt.match_begin >= start) & (rt.match_begin < start + pd.offsets.MonthBegin(1))].reset_index(drop=True)
        mm = {}
        for w in (1, 0):
            Xt, done = cont_rows(te, w)
            p = np.mean([mdl.predict_proba(Xt[cols])[:, 1] for mdl in mdls], axis=0)
            mm[w] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(te[["game_id", "round"]]))
        s = st.merge(pd.read_parquet(CACHE / f"pred_{a.pred}_{m}.parquet"), on=["game_id", "event_seq"])
        k = pd.MultiIndex.from_frame(s[["game_id", "round"]])
        pa = np.where(s.side_a == "CT", s.p_ct, 1 - s.p_ct)
        new = pa * mm[1].reindex(k).to_numpy() + (1 - pa) * mm[0].reindex(k).to_numpy()
        old = pa * s.m1 + (1 - pa) * s.m0
        base = pd.read_parquet(CACHE / f"pred_base_{m}.parquet")
        sb = st.merge(base, on=["game_id", "event_seq"])
        pb = np.where(sb.side_a == "CT", sb.p_ct, 1 - sb.p_ct)
        bench = ll(pb * sb.m1 + (1 - pb) * sb.m0, sb.map_a_wins).mean()
        s.assign(p_new=new, p_old=old).drop(columns=["m1", "m0"]).to_parquet(CACHE / f"map_{a.feats}_{a.pred}_{m}.parquet")
        print(m, a.feats, a.kw, "bench %.5f old-cont %.5f new-cont %.5f diff %.5f (%ds)" % (
            bench, ll(old, s.map_a_wins).mean(), ll(new, s.map_a_wins).mean(), ll(new, s.map_a_wins).mean() - bench,
            time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
