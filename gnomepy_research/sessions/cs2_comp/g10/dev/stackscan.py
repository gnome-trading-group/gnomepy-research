"""Scan single extra columns on top of the e12 pool (month held out), dev targets only.
python -m ...g10.dev.stackscan --work W --data D"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g10.dev.evaluate import ci, ll, logit, pool
from gnomepy_research.sessions.cs2_comp.g10.dev.layer import context_columns


def fitw(X, y):
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    R = pd.read_parquet(f"{a.work}/rows.parquet")
    E = pd.read_parquet(f"{a.work}/ev.parquet")
    X_ = pd.read_parquet(f"{a.work}/exp.parquet")
    T = pd.read_parquet(f"{a.data}/targets_tune.parquet")[["match_id", "is_lan", "event_tier", "team_a_standin_ids", "team_b_standin_ids"]]
    d = E.merge(R[R.is_target].drop(columns=["y", "q_e12", "m0"]), on="match_id").merge(T, on="match_id").merge(X_, on="match_id")
    y = d.y.to_numpy(float)
    mo = d.month.to_numpy()
    z0, ze = logit(d.m0), logit(d.q_e12)
    base = ll(pool([d.m0, d.q_e12], y, mo), y)

    def cvx(extra, label):
        X = np.column_stack([z0, ze] + [np.asarray(e, float) for e in extra] + [np.ones(len(y))])
        out = np.empty(len(y))
        for m in np.unique(mo):
            tr = mo != m
            out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fitw(X[tr], y[tr])))
        print(f"{label:30s} {ci(ll(out, y) - base)} w={fitw(X, y)[2:-1].round(3)}", flush=True)

    ctx = context_columns(d)
    de = ze - z0
    T2 = pd.read_parquet(f"{a.data}/targets_tune.parquet")
    long = pd.concat([T2[["match_id", "match_time", "week_cutoff", "team_a_id"]].set_axis(["match_id", "t", "wk", "team"], axis=1),
                      T2[["match_id", "match_time", "week_cutoff", "team_b_id"]].set_axis(["match_id", "t", "wk", "team"], axis=1)])
    long = long.sort_values("t")
    long["nprev"] = long.groupby(["wk", "team"]).cumcount()
    nprev = long.groupby("match_id").nprev.sum()
    d["nprev"] = d.match_id.map(nprev).fillna(0).to_numpy()
    days = (pd.to_datetime(T2.set_index("match_id").match_time).dt.tz_convert(None) - pd.to_datetime(T2.set_index("match_id").week_cutoff)).dt.total_seconds() / 86400
    d["days"] = d.match_id.map(days).to_numpy()
    print(d[["nprev", "days"]].describe(), flush=True)
    cvx([de * (d.days - 3.5) / 3.5], "de*days")
    cvx([de * np.minimum(d.nprev, 3)], "de*nprev")
    cvx([de * (d.nprev > 0)], "de*anyprev")
    cvx([de * ctx["young"].to_numpy(), de * (d.days - 3.5) / 3.5], "de*young + de*days")
    cvx([de * ctx["young"].to_numpy(), de * np.minimum(d.nprev, 3)], "de*young + de*nprev")
    cvx([], "none")
    cvx([d.zbar_120m - z0], "twap120")
    cvx([d.zbar_30m - z0], "twap30")
    cvx([logit(d.p_open) - z0], "open-z0")
    cvx([np.nan_to_num(logit(d.p_360m) - z0)], "p6h-z0")
    for c in ctx.columns:
        cvx([de * ctx[c].to_numpy()], f"de*{c}")
    for c in ctx.columns:
        cvx([z0 * ctx[c].to_numpy()], f"z0*{c}")
    lan = d.is_lan.to_numpy(float)
    cvx([de * lan], "de*lan")
    cvx([z0 * lan], "z0*lan")
    t0 = (d.event_tier == 0).to_numpy(float)
    cvx([de * t0], "de*tier0")
    sub = np.array([len(x) - len(z) for x, z in zip(d.team_a_standin_ids, d.team_b_standin_ids)], float)
    cvx([sub], "standin diff")
    cvx([d.h_new_a - d.h_new_b], "new diff")
    cvx([np.log1p(d.h_idle_a) - np.log1p(d.h_idle_b)], "idle diff")
    cvx([logit(d.q_e12h) - ze], "hltv-only minus e12")
    cvx([de ** 3 / (1 + de ** 2)], "de cubic")
    for m in ["q_m120", "q_m60", "q_m240", "q_m120w"]:
        cvx([logit(d[m]) - logit(d.q_base)], f"{m}-base")
    cvx([logit(d.q_hl90) - logit(d.q_base)], "hl90-base")


if __name__ == "__main__":
    main()
