"""Walk-forward evaluation on the feature panel built by dev_panel: weekly refits of the stack, scored like the
harness (pool of market and q with each month held out), plus each rater on its own."""
from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.data import load_hltv
from gnomepy_research.sessions.cs2_comp.g12.predict import panel
from gnomepy_research.sessions.cs2_comp.g12.raters import EXTRA, RaterConfig
from gnomepy_research.sessions.cs2_comp.g12.stack import Stack, fit_logistic, logit, sigmoid, stage1_design

DATA = "/Users/mprey/gnome-research-work/cs2/groundup/comp/data"
WORK = os.environ.get("OUTPUT_DIR", "/Users/mprey/gnome-research-work/cs2/groundup/g12work") + "/"


def ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pool(m0, q, y, months, extra=None):
    X = np.column_stack([logit(m0), logit(q)] + ([] if extra is None else [logit(extra)]) + [np.ones(len(y))])
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = sigmoid(X[~tr] @ fit_logistic(X[tr], y[tr], 1e-6))
    return out


def boot(x, n=1000, seed=0):
    r = np.random.default_rng(seed)
    b = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return f"{x.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}]"


def walk(panel: pd.DataFrame, weeks, make_stack, eval_rows) -> pd.DataFrame:
    out = []
    for w in weeks:
        st = make_stack().fit(panel[~panel.target], w)
        d = eval_rows[eval_rows.week_cutoff == w]
        out.append(pd.DataFrame({"match_id": d.match_id.to_numpy(), "q": st.predict(d), "z": sigmoid(st.z(d)),
                                 "week_cutoff": w}))
    return pd.concat(out, ignore_index=True)


def report(name: str, res: pd.DataFrame, ev: pd.DataFrame) -> dict:
    d = ev.merge(res, on=["match_id", "week_cutoff"])
    if E12 is not None:
        d = d.merge(E12, on="match_id", how="left")
    d = d[d.m0.notna() & d.y.notna()]
    y = d.y.to_numpy(float)
    months = pd.to_datetime(d.time).dt.to_period("M").astype(str).to_numpy()
    lm = ll(d.m0, y)
    lq = ll(d.q, y)
    lp = ll(pool(d.m0, d.q, y, months), y)
    lz = ll(pool(d.m0, d.z, y, months), y)
    if E12 is not None and "e12" in d:
        lb = ll(pool(d.m0, d.e12, y, months), y)
        ls = ll(pool(d.m0, d.q, y, months, d.e12), y)
        lsz = ll(pool(d.m0, d.z, y, months, d.e12), y)
        print(f"{'':28s} e12 pool-m {boot(lb - lm)}  stack-basepool {boot(ls - lb)}  stackz-basepool {boot(lsz - lb)}")
    print(f"{name:28s} n={len(d)} market {lm.mean():.4f} q-m {boot(lq - lm)} pool-m {boot(lp - lm)} poolz-m {boot(lz - lm)}", flush=True)
    return {"name": name, "q_m": float((lq - lm).mean()), "pool_m": float((lp - lm).mean()), "poolz_m": float((lz - lm).mean())}


E12 = None


def main():
    global E12, DATA, WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=WORK + "panel.parquet")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--raters", default="")
    ap.add_argument("--period", default="dev", choices=["dev", "pre"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--e12", default=WORK + "e12run")
    a = ap.parse_args()
    DATA, WORK = a.data, a.work.rstrip("/") + "/"
    if a.panel.startswith("@"):
        a.panel = WORK + a.panel[1:]
    e12 = sorted(Path(a.e12).glob("*.parquet"))
    if e12:
        E12 = pd.concat([pd.read_parquet(f) for f in e12])[["match_id", "q"]].rename(columns={"q": "e12"})
    if a.build:
        t = pd.read_parquet(DATA + "/targets_tune.parquet")
        p = panel(DATA, t, cfg=RaterConfig(extra=EXTRA))
        p.to_parquet(a.panel)
    p = pd.read_parquet(a.panel)
    if a.period == "dev":
        ev = p[p.target].copy()
        full = load_hltv(Path(DATA), pd.Timestamp("2026-08-01"))[1]
        ev["y"] = ev.match_id.map(full.set_index("match_id").y)
    else:
        ev = p[(~p.target) & (p.bo_type == 3) & (p.week_cutoff >= "2026-03-02") & (p.week_cutoff < "2026-03-30")]
    weeks = sorted(ev.week_cutoff.unique())
    mix = ("gauss", "gauss_unc", "gauss_h", "joint_elo", "team_elo", "log_points", "hltv_rating", "faceit_level",
           "unknown", "idle", "new", "sub", "standin", "form", "market_elo")
    allseg = ("s_unknown", "s_new", "s_young", "s_still", "s_lan", "s_unranked", "s_var")
    variants = {
        "ens": partial(Stack, use1=("gauss", "gx_a", "gx_b", "gx_fast", "gx_slow"), model_only=True),
        "ens_coh": partial(Stack, use1=("gauss", "gx_a", "gx_b", "gx_fast", "gx_slow") + ("together", "gauss_lowcoh"), model_only=True),
        "ens_lan": partial(Stack, use1=("gauss", "gx_a", "gx_b", "gx_fast", "gx_slow") + ("lan_res",), model_only=True),
        "ens_coh_lan_sub": partial(Stack, use1=("gauss", "gx_a", "gx_b", "gx_fast", "gx_slow") + ("together", "gauss_lowcoh", "lan_res", "sub"), model_only=True),
    }
    for name, mk in variants.items():
        res = walk(p, weeks, mk, ev)
        report(name, res, ev)
        res.to_parquet(WORK + f"res_{name}.parquet")
    if not a.raters:
        return
    for r in a.raters.split(",") if a.raters != "all" else ["team_elo", "player_elo", "joint_elo", "hltv_rating", "faceit_level", "log_points", "market_elo", "gauss", "gauss_h"]:
        res_r = walk(p, weeks, lambda r=r: SingleRater(r), ev)
        report(r, res_r, ev)
        res_r.to_parquet(WORK + f"res_r_{r}.parquet")


class SingleRater:
    def __init__(self, col):
        self.col = col

    def fit(self, hist, cutoff):
        h = hist[(hist.known < cutoff) & hist.y.notna() & (hist.bo_type == 3) & (hist.week_cutoff >= cutoff - pd.Timedelta(days=365))]
        self.b = fit_logistic(stage1_design(h)[[self.col]].to_numpy(), h.y.to_numpy(float), 1.0)
        return self

    def z(self, d):
        return stage1_design(d)[[self.col]].to_numpy() @ self.b

    def predict(self, d):
        return sigmoid(self.z(d))


if __name__ == "__main__":
    main()
