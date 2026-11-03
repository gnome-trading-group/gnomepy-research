"""
Score candidate layers on the walk-forward rows the way the harness scores a submission.

    python -m gnomepy_research.sessions.cs2_comp.g15.dev_eval --rows <parquet> [--first-dev 2026-03-30]

Pure variants fit the layer once on every row before the first dev cutoff. Market-aware variants refit each dev month
on the market rows before it (expanding), which is what predict.py does week by week. Every variant is then pooled
with the market by month-held-out logistic regression, as in the harness.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g15.layer import clip_logit, design, fit_logistic, log_loss, thin_flag

CTX = ["standin", "new", "idle", "rank"]
PURE = {
    "kalman": ["one", "z_pk"],
    "seq3+ctx": ["one", "z_tm", "z_pm", "z_pk", "z_tm_thin", "z_pm_thin", "z_pk_thin", "few"] + CTX,
    "bf": ["one", "z_bf"],
    "bf+ctx": ["one", "z_bf", "z_bf_thin", "z_bf_few", "bf_few"] + CTX,
    "bt": ["one", "z_bt_p", "z_bt_t"],
    "bt+ctx": ["one", "z_bt_p", "z_bt_t", "z_bt_p_thin", "z_bt_p_few", "bt_few"] + CTX,
    "bff": ["one", "z_bff"],
    "bff+ctx": ["one", "z_bff", "z_bff_thin", "z_bff_few", "bff_few"] + CTX,
    "bf+bff+ctx": ["one", "z_bf", "z_bff", "z_bf_thin", "z_bff_thin", "z_bf_few", "z_bff_few", "bf_few", "bff_few"] + CTX,
    "all+ctx": ["one", "z_bf", "z_bff", "z_bt_t", "z_pk", "z_tm", "z_bf_thin", "z_bff_thin", "z_bf_few", "z_bff_few", "bf_few", "bff_few"] + CTX,
}
MARKET = {
    "m+bf+bff+ctx": ["one", "zm", "z_bf", "z_bff", "z_bf_thin", "z_bff_thin", "z_bf_few", "z_bff_few", "bf_few", "bff_few"] + CTX,
}


def pool(m0, q, y, months, extra=None):
    X = np.column_stack([clip_logit(m0), clip_logit(q)] + ([] if extra is None else [clip_logit(extra)]) + [np.ones(len(y))])
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = expit(X[~tr] @ fit_logistic(X[tr], y[tr], l2=1e-6))
    return out


def ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(1)
    return f"{x.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--first-dev", default="2026-03-30")
    ap.add_argument("--warmup", default="2024-09-01")
    ap.add_argument("--extra", nargs="*", default=[], help="batch-fit row files to merge on match_id")
    ap.add_argument("--each", action="store_true", help="score every *_gap column alone instead of the named variants")
    ap.add_argument("--variants", help="JSON {name: [columns]} replacing the built-in pure variants")
    ap.add_argument("--market-variants", help="JSON {name: [columns]} replacing the built-in market-aware variants")
    a = ap.parse_args()
    rows = pd.read_parquet(a.rows)
    for e in a.extra:
        rows = rows.merge(pd.read_parquet(e).drop(columns=["week_cutoff"]), on="match_id", how="inner")
    rows["week_cutoff"] = pd.to_datetime(rows.week_cutoff)
    rows = rows[rows.week_cutoff >= pd.Timestamp(a.warmup)].reset_index(drop=True)
    first_dev = pd.Timestamp(a.first_dev)
    y = rows.y.to_numpy(float)
    months = rows.week_cutoff.dt.to_period("M").astype(str).to_numpy()
    train = (rows.week_cutoff < first_dev).to_numpy()
    early = (rows.week_cutoff < pd.Timestamp("2025-10-06")).to_numpy()
    late = train & ~early
    dev = (~train & rows.m0.notna()).to_numpy()
    thin = thin_flag(rows).astype(bool)
    m0 = rows.m0.to_numpy(float)
    lm = log_loss(m0[dev], y[dev])
    print(f"rows train {train.sum()} (early {early.sum()}, late {late.sum()}) dev-with-market {dev.sum()}  market ll {lm.mean():.4f}")
    print(f"{'variant':16s} {'late q ll':10s} {'dev q ll':9s} {'q - market':30s} {'pool - market':30s} {'thin pool-mkt':30s} rest pool-mkt")
    results = {}
    variants = json.loads(a.variants) if a.variants else dict(PURE)
    market_variants = json.loads(a.market_variants) if a.market_variants else dict(MARKET)
    if a.each:
        variants = {}
        for c in rows.columns:
            if c.endswith("_gap"):
                z = "z_" + c[:-4]
                variants[c[:-4]] = ["one", z]
                if f"{c[:-4]}_few_a" in rows:
                    variants[c[:-4] + "+few"] = ["one", z, f"{z}_few", f"{c[:-4]}_few", "standin"]
    for name, cols in variants.items():
        X = design(rows, cols)
        w_early = fit_logistic(X[early], y[early])
        ll_late = log_loss(expit(X[late] @ w_early), y[late]).mean()
        w = fit_logistic(X[train], y[train])
        q = expit(X @ w)
        results[name] = q
        lq = log_loss(q[dev], y[dev])
        lp = log_loss(pool(m0[dev], q[dev], y[dev], months[dev]), y[dev])
        td = thin[dev]
        print(f"{name:16s} {ll_late:<10.4f} {lq.mean():<9.4f} {ci(lq - lm):30s} {ci(lp - lm):30s} {ci((lp - lm)[td]):30s} {ci((lp - lm)[~td])}")
        print("    weights:", {c: round(float(v), 3) for c, v in zip(cols, w)})
    has_m = rows.m0.notna().to_numpy()
    for name, cols in ({} if a.each else market_variants).items():
        X = design(rows, cols)
        q = np.full(len(rows), np.nan)
        for m in np.unique(months[dev]):
            tr = has_m & (months < m) & (rows.week_cutoff < pd.Timestamp(m + "-01")).to_numpy()
            te = dev & (months == m)
            w = fit_logistic(X[tr], y[tr], l2=1.0)
            q[te] = expit(X[te] @ w)
        lq = log_loss(q[dev], y[dev])
        lp = log_loss(pool(m0[dev], q[dev], y[dev], months[dev]), y[dev])
        td = thin[dev]
        print(f"{name:16s} {'-':10s} {lq.mean():<9.4f} {ci(lq - lm):30s} {ci(lp - lm):30s} {ci((lp - lm)[td]):30s} {ci((lp - lm)[~td])}")
        print("    last weights:", {c: round(float(v), 3) for c, v in zip(cols, w)})
    if a.each or a.variants:
        return
    print("by month: market ll / pool(bf+bff+ctx) ll")
    q = results["bf+bff+ctx"]
    lp = log_loss(pool(m0[dev], q[dev], y[dev], months[dev]), y[dev])
    for m in np.unique(months[dev]):
        sel = months[dev] == m
        print(f"  {m}: n={sel.sum()} market {lm[sel].mean():.4f} pool {lp[sel].mean():.4f} q {log_loss(q[dev][sel], y[dev][sel]).mean():.4f}")


if __name__ == "__main__":
    main()
