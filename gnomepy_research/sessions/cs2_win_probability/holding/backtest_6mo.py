"""
Six-month walk-forward backtest against Polymarket: is the edge stable, or was it September?

Each calendar month is predicted by an L1 model trained only on matches before
that month - the last ~12% of the pre-month series held out for early stopping
and calibration, so a monthly retrain is exactly what is simulated. Series
probabilities come from the uncalibrated series DP, which is how production
composes them.

The verdict is per month. A pooled number can be carried by one good month; an
edge that is real should show up in most months on its own.
"""
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR
from gnomepy_research.sessions.cs2_win_probability.pre_map_features import PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import _fit, _matrix, fit_calibrator
from gnomepy_research.sessions.cs2_win_probability.series_dp import build_node_frame, series_win_prob
from gnomepy_research.sessions.cs2_win_probability.splits import (
    clustered_bootstrap_delta, split_val_for_calibration, temporal_split_by_series,
)
from gnomepy_research.sessions.cs2_win_probability.symmetry import augment, build_swap_plan, symmetric_predict_proba

logging.basicConfig(level=logging.WARNING)
NAMES = list(PRE_MAP_FEATURE_NAMES)
MONTHS = pd.date_range("2026-03-01", "2026-09-01", freq="MS")
TAKER = 0.07

history = pd.read_parquet(DATA_DIR / "cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history.match_date)
priors = pd.read_parquet(DATA_DIR / "priors_full.parquet")
cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
mf = priors.merge(history[cols], on=["match_id", "map_name"])
mf = mf[mf.team_a_score != mf.team_b_score].sort_values(["match_date", "match_id"]).reset_index(drop=True)
by_mid = {k: v for k, v in priors.groupby("match_id")}
plan = build_swap_plan(NAMES)
series_won = history.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() > len(w) - w.sum()))


def train_before(cutoff):
    pre = mf[mf.match_date < cutoff].reset_index(drop=True)
    tr, val, _ = temporal_split_by_series(pre, frac_train=0.88, frac_val=0.12)
    es, cal = split_val_for_calibration(pre, val)
    X, y = _matrix(pre, NAMES), pre.team_a_won.to_numpy().astype(int)
    Xtr, ytr = augment(X[tr], y[tr], plan)
    model = _fit(Xtr, ytr, X[es], y[es], 1000, 5)
    calib = fit_calibrator(symmetric_predict_proba(model, X[cal], plan), y[cal])
    return lambda M: calib.predict(symmetric_predict_proba(model, M, plan)), len(tr)


def series_dp(predict, ids):
    frames, idx = [], []
    for mid in ids:
        nf, nodes = build_node_frame(by_mid[mid], 2, NAMES)
        frames.append(nf); idx.append(nodes)
    pr = predict(_matrix(pd.concat(frames, ignore_index=True), NAMES))
    out, cur = [], 0
    for f, nodes in zip(frames, idx):
        ch = pr[cur:cur + len(f)]; cur += len(f)
        out.append(series_win_prob(dict(zip(nodes, ch)), 2, n_veto_maps=3))
    return np.array(out)


PREDS = DATA_DIR / "backtest_6mo_preds.parquet"
# Refitting seven monthly models is the slow part; set CS2_BACKTEST_REUSE_PREDS=1 to re-score saved predictions.
REUSE = os.environ.get("CS2_BACKTEST_REUSE_PREDS") == "1" and PREDS.exists()
preds = []
for start in ([] if REUSE else MONTHS):
    end = start + pd.offsets.MonthBegin(1)
    predict, n_train = train_before(start)
    month = mf[(mf.match_date >= start) & (mf.match_date < end)]
    b3 = month[month.bo_type == 3]
    maps12 = b3[b3.map_position_in_series.isin([1, 2])]
    q_map = predict(_matrix(maps12, NAMES))
    for r, q in zip(maps12.itertuples(), q_map):
        preds.append({"month": start.strftime("%Y-%m"), "match_id": r.match_id,
                      "market": f"game{int(r.map_position_in_series)}", "q": q, "y": int(r.team_a_won)})
    ids = [m for m in b3.match_id.unique() if m in by_mid]
    for mid, q in zip(ids, np.clip(series_dp(predict, ids), 0.01, 0.99)):
        preds.append({"month": start.strftime("%Y-%m"), "match_id": mid, "market": "series",
                      "q": q, "y": int(series_won[mid])})
    print(f"{start:%Y-%m}: trained on {n_train} maps; predicted {len(maps12)} maps, {len(ids)} series", flush=True)
if REUSE:
    P = pd.read_parquet(PREDS)
    P["market"] = P.market.str.replace(r"\.0$", "", regex=True)
else:
    P = pd.DataFrame(preds)
    P.to_parquet(PREDS, index=False)

paths = pd.read_parquet(DATA_DIR / "pm_all_paths.parquet").sort_values(["match_id", "market", "t"])
matched = pd.read_parquet(DATA_DIR / "pm_all_matched.parquet")
kick = dict(zip(matched.match_id, pd.to_datetime(matched.kickoff)))
vol = dict(zip(zip(matched.match_id, matched.market), matched.volume))
by_key = {k: (g.t.to_numpy(), g.p_a.to_numpy()) for k, g in paths.groupby(["match_id", "market"])}


def at(key, ts):
    if key not in by_key:
        return np.nan
    t, p = by_key[key]
    v = p[t <= ts]
    return float(v[-1]) if len(v) else np.nan


def decided(key, after):
    if key not in by_key:
        return np.nan
    t, p = by_key[key]
    hit = np.flatnonzero((t > after) & ((p >= 0.97) | (p <= 0.03)))
    return float(t[hit[0]]) if len(hit) else np.nan


rows = []
for r in P.itertuples():
    if r.match_id not in kick:
        continue
    k = kick[r.match_id].timestamp()
    key = (r.match_id, r.market)
    base = {"month": r.month, "match_id": r.match_id, "market": r.market, "q": r.q, "y": r.y,
            "volume": vol.get(key, np.nan)}
    if r.market == "series":
        if key in by_key:
            base["open"] = float(by_key[key][1][0])
        for h in (12, 6, 1, 0):
            base[f"k{h}"] = at(key, k - h * 3600)
    elif r.market == "game1":
        for h in (6, 1, 0):
            base[f"k{h}"] = at(key, k - h * 3600)
    else:
        m1 = decided((r.match_id, "game1"), k)
        base["m1end"] = at(key, m1) if m1 == m1 else np.nan
    rows.append(base)
E = pd.DataFrame(rows)
E.to_parquet(DATA_DIR / "backtest_6mo_eval.parquet", index=False)


def simulate(p, q, y, hs, th=0.05):
    fee = TAKER * p * (1 - p)
    ea, eb = q - (p + hs) - fee, (1 - q) - ((1 - p) + hs) - fee
    ba = ea > th
    bb = (eb > th) & ~ba
    pnl = np.where(ba, y - (p + hs) - fee, 0.0) + np.where(bb, (1 - y) - ((1 - p) + hs) - fee, 0.0)
    cost = np.where(ba, p + hs, 0.0) + np.where(bb, 1 - p + hs, 0.0)
    t = ba | bb
    return pnl[t], cost[t]


def boot(x, n=1000, seed=0):
    if len(x) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    return np.percentile([rng.choice(x, len(x)).sum() for _ in range(n)], [2.5, 97.5])


lg = lambda x: np.log(x / (1 - x))
CASES = [("series", "k12", "series, 12h before", 0.01), ("series", "k6", "series, 6h before", 0.01),
         ("series", "k0", "series, kickoff", 0.01), ("game1", "k0", "map 1, kickoff", 0.0125),
         ("game1", "k0", "map 1, kickoff (5.5c spread)", 0.055), ("game2", "m1end", "map 2, at map-1 end", 0.0125)]

summary = []
for market, col, label, hs in CASES:
    print(f"\n=== {label}  (half-spread {hs*100:.2f}c, 5c edge threshold) ===")
    print(f"{'month':>8} {'n':>5} {'mkt LL':>7} {'mdl LL':>7} {'gap':>8} {'blend-mkt':>10} {'trades':>7} {'P&L':>8} {'95% CI':>17} {'per $':>7}")
    sub = E[(E.market == market) & E[col].notna()] if col in E else E.iloc[0:0]
    for month, g in list(sub.groupby("month")) + [("POOLED", sub)]:
        if len(g) < 40:
            continue
        y, q = g.y.to_numpy(), np.clip(g.q.to_numpy(), 1e-4, 1 - 1e-4)
        mk = np.clip(g[col].to_numpy(), 1e-4, 1 - 1e-4)
        blend = 1 / (1 + np.exp(-(lg(mk) + lg(q)) / 2))
        db, blo, bhi = clustered_bootstrap_delta(y, blend, mk, g.match_id.to_numpy(), log_loss, n_boot=300)
        pnl, cost = simulate(mk, q, y, hs)
        lo, hi = boot(pnl)
        roi = pnl.sum() / cost.sum() if cost.sum() else np.nan
        print(f"{month:>8} {len(g):5d} {log_loss(y, mk):7.4f} {log_loss(y, q):7.4f} {log_loss(y, q)-log_loss(y, mk):+8.4f} "
              f"{db:+9.4f}{'*' if (blo>0)==(bhi>0) else ' '} {len(pnl):7d} {pnl.sum():+8.2f} [{lo:+6.1f},{hi:+6.1f}] {roi:+7.1%}")
        summary.append({"case": label, "month": month, "n": len(g), "gap": log_loss(y, q) - log_loss(y, mk),
                        "blend_gain": db, "trades": len(pnl), "pnl": pnl.sum(), "lo": lo, "hi": hi, "roi": roi})

S = pd.DataFrame(summary)
S.to_parquet(DATA_DIR / "backtest_6mo_summary.parquet", index=False)
print("\n=== stability: months with positive P&L / months tested ===")
for label, g in S[S.month != "POOLED"].groupby("case", sort=False):
    print(f"  {label:32s} {int((g.pnl > 0).sum())}/{len(g)} positive, {int((g.lo > 0).sum())} with CI above zero")
