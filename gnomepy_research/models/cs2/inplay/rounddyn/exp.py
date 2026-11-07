"""Quick round-model experiments on the research cache: walk-forward by month, scored as map log loss through the
benchmark map model's continuations, against the benchmark round model refit on the same rows."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from gnomepy_research.models.cs2.inplay import baseline

CACHE = Path(__file__).resolve().parents[5] / ".cache"
EPS = 1e-4


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def add_derived(x: pd.DataFrame) -> pd.DataFrame:
    a_ct = x.side_a == "CT"
    x = x.copy()
    x["ct_prior"] = np.where(a_ct, x.prior_logit, -x.prior_logit)
    x["ct_money_start"] = np.where(a_ct, x.a_money, x.b_money)
    x["t_money_start"] = np.where(a_ct, x.b_money, x.a_money)
    x["alive_diff"] = x.ct_alive - x.t_alive
    x["eq_share"] = x.ct_eq / (x.ct_eq + x.t_eq + 1)
    if "a_kd_prev" in x:
        sg = np.where(a_ct, 1.0, -1.0)
        x["ct_kd_prev"] = sg * x.a_kd_prev
        x["ct_upsets_prev"] = sg * x.a_upsets_prev
        x["ct_gun_wr_prev"] = np.where(a_ct, x.a_gun_wr_prev, 1 - x.a_gun_wr_prev)
    return x


SETS = {
    "base": baseline.ROUND,
}
SETS["dyn"] = baseline.ROUND + ["ct_rifles", "t_rifles", "ct_smgs", "t_smgs", "ct_helm", "t_helm", "ct_armor", "t_armor",
                                "site", "bomb_left", "clock_left", "defuse_kit_ok", "defuse_nokit_ok", "since_kill",
                                "last_dead", "ct_dead_5s", "t_dead_5s", "last_trade", "open_dead", "ct_max_cls", "t_max_cls"]
SETS["dyn_prior"] = SETS["dyn"] + ["ct_prior", "has_prior"]
SETS["dyn_all"] = SETS["dyn_prior"] + ["ct_dead_hs", "t_dead_hs", "ct_dead_util", "t_dead_util", "last_kill_cls",
                                       "ct_pickup", "t_pickup", "ct_money", "t_money", "open_t", "n_kills", "alive_diff", "eq_share"]

SETS["dyn_form"] = SETS["dyn_prior"] + ["ct_kd_prev", "ct_upsets_prev", "ct_gun_wr_prev", "rounds_prev", "gun_prev"]
SETS["dyn_kd"] = SETS["dyn_prior"] + ["ct_kd_prev", "rounds_prev"]


def fit_round(tr, cols, seed=0, **kw):
    params = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=50, reg_lambda=5, n_jobs=4, random_state=seed, tree_method="hist")
    params.update(kw)
    return xgb.XGBClassifier(**params).fit(tr[cols], tr.y_ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["base", "dyn"])
    ap.add_argument("--months", nargs="*", default=["2026-08"])
    ap.add_argument("--override", type=int, default=1)
    ap.add_argument("--kw", default="{}")
    a = ap.parse_args()
    kw = json.loads(a.kw)
    x = add_derived(pd.read_parquet(CACHE / "states.parquet"))
    res = {}
    for m in a.months:
        start = pd.Timestamp(m + "-01", tz="UTC")
        tr = x[(x.end < start) & x.y_ct.notna()]
        te = x[(x.match_begin >= start) & (x.match_begin < start + pd.offsets.MonthBegin(1))]
        for name in a.sets:
            t0 = time.time()
            mdl = fit_round(tr, SETS[name], **({} if name == "base" else kw))
            p = mdl.predict_proba(te[SETS[name]])[:, 1]
            if a.override and name != "base":
                p = np.where(te.over == 1, np.where(te.over_ct_won.notna(), te.over_ct_won, p), p)
            pa = np.where(te.side_a == "CT", p, 1 - p)
            pm = pa * te.m1 + (1 - pa) * te.m0
            r = dict(round_ll=float(ll(p, te.y_ct).mean()), map_ll=float(ll(pm, te.map_a_wins).mean()), n=len(te),
                     sec=round(time.time() - t0))
            res.setdefault(m, {})[name] = r
            pd.DataFrame(dict(game_id=te.game_id.to_numpy(), event_seq=te.event_seq.to_numpy(), p_ct=p)).to_parquet(
                CACHE / f"pred_{name}_{m}.parquet")
            print(m, name, r, file=sys.stderr, flush=True)
    for m, d in res.items():
        b = d.get("base")
        for k, r in d.items():
            if b:
                print(m, k, "map diff %.5f round diff %.5f" % (r["map_ll"] - b["map_ll"], r["round_ll"] - b["round_ll"]))


if __name__ == "__main__":
    main()
