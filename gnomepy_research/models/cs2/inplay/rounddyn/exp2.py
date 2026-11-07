"""Continuation experiments: M(next score) learned from round n's own start-of-round economy as well as the next
score state, trained on every round with its actual outcome, against the benchmark's score-only map model."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.inplay import baseline
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps
from gnomepy_research.models.cs2.inplay.rounddyn.exp import CACHE, ll

MASK = True
DATA = Path.home() / "gnome-research-work/cs2/comp3/data"
FORM_C = ["a_upsets_prev", "a_gun_wr_prev", "gun_prev", "a_kd_prev", "a_resid_prev", "rounds_prev"]
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
    a = ap.parse_args()
    rt = build()
    st = pd.read_parquet(CACHE / "states.parquet", columns=["game_id", "round", "event_seq", "side_a", "map_a_wins", "m1", "m0"])
    for m in a.months:
        t0 = time.time()
        start = pd.Timestamp(m + "-01", tz="UTC")
        tr = rt[(rt.end < start)].reset_index(drop=True)
        rows = []
        for w in (1, 0):
            X, done = cont_rows(tr, w)
            keep = (tr.a_wins == w) & np.isnan(done)
            rows.append((X[keep.to_numpy()], tr.map_a_wins[keep.to_numpy()]))
        X = pd.concat([r[0] for r in rows]); y = pd.concat([r[1] for r in rows])
        cols = baseline.MAPCOLS + {"none": [], "econ": ECON, "econ_form": ECON + FORM_C[:3], "econ_kd": ECON + FORM_C[:4], "econ_resid": ECON + FORM_C[:3] + ["a_resid_prev"], "econ_r": ECON + ["a_resid_prev"], "econ2": ECON + FORM_C[:3] + ["a_cash_next", "b_cash_next"], "econ3": ECON + ["a_cash_next", "b_cash_next"], "econ4": ECON + ["a_cash_next", "b_cash_next", "a_resid_prev"]}[a.feats]
        mdl = baseline.trees(a.seed, n_jobs=4).fit(X[cols], y)
        te = rt[(rt.match_begin >= start) & (rt.match_begin < start + pd.offsets.MonthBegin(1))].reset_index(drop=True)
        mm = {}
        for w in (1, 0):
            Xt, done = cont_rows(te, w)
            p = mdl.predict_proba(Xt[cols])[:, 1]
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
        print(m, a.feats, "bench %.5f old-cont %.5f new-cont %.5f diff %.5f (%ds)" % (
            bench, ll(old, s.map_a_wins).mean(), ll(new, s.map_a_wins).mean(), ll(new, s.map_a_wins).mean() - bench,
            time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
