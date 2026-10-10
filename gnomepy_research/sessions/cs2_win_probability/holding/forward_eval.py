"""
Forward test on matches played after the model was frozen.

The model is trained only on maps before CUTOFF; every match scored here was
played on or after it, so nothing in its design or fit has seen them. Matching
and pricing rules are the backtest's: moneyline slugs end in the date, map
winners end -gameN, orientation comes from `outcomes`, settlement must agree
with HLTV, map 1 is priced at kickoff and map 2 when map 1 is decided.

    poetry run python .../holding/forward_eval.py 2026-09-30 2026-10-02
"""
import datetime as dt
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import log_loss

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.pipelines.hltv_cs2.build_priors import build_priors
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR
from gnomepy_research.sessions.cs2_win_probability.pre_map_features import PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import _fit, _matrix, fit_calibrator
from gnomepy_research.sessions.cs2_win_probability.series_dp import build_node_frame, series_win_prob
from gnomepy_research.sessions.cs2_win_probability.splits import (
    clustered_bootstrap_delta, split_val_for_calibration, temporal_split_by_series,
)
from gnomepy_research.sessions.cs2_win_probability.symmetry import augment, build_swap_plan, symmetric_predict_proba

CUTOFF = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-09-30")
END = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2026-10-02")
NAMES = list(PRE_MAP_FEATURE_NAMES)
TAKER = 0.07
GAMMA = "https://gamma-api.polymarket.com"
SLUG = re.compile(r"-(\d{4}-\d{2}-\d{2})(?:-game(\d))?$")


def norm(n):
    n = (n or "").lower().strip()
    n = re.sub(r"\b(esports|esport|team|gaming|club|academy|nxt|fe)\b", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


# ---- 1. Polymarket markets created around the window -------------------------------
s = requests.Session()
s.headers["User-Agent"] = "research/1.0"
rows = []
lo, hi = CUTOFF - pd.Timedelta(days=10), END + pd.Timedelta(days=2)
off = 0
while off <= 3900:
    r = s.get(f"{GAMMA}/events", params={"limit": 100, "offset": off, "tag_slug": "counter-strike-2",
                                         "start_date_min": lo.strftime("%Y-%m-%dT00:00:00Z"),
                                         "start_date_max": hi.strftime("%Y-%m-%dT00:00:00Z")}, timeout=30)
    data = r.json() if r.status_code == 200 else []
    if not data:
        break
    for e in data:
        for m in e.get("markets", []):
            loads = lambda k: json.loads(m.get(k) or "[]")
            rows.append({"market_slug": m.get("slug"), "outcomes": loads("outcomes"), "tokens": loads("clobTokenIds"),
                         "final": loads("outcomePrices"), "volume": float(m.get("volumeNum") or 0),
                         "closed": m.get("closed")})
    off += 100
    time.sleep(0.2)
pm = pd.DataFrame(rows).drop_duplicates("market_slug")
ext = pm.market_slug.str.extract(SLUG)
pm["slug_date"], pm["game"] = ext[0], ext[1].fillna("0")
pm = pm[pm.slug_date.notna() & (pm.outcomes.str.len() == 2) & (pm.tokens.str.len() == 2)].copy()
pm["slug_date"] = pd.to_datetime(pm.slug_date).dt.date
pm["market"] = np.where(pm.game == "0", "series", "game" + pm.game)
pm["o0"] = [norm(o[0]) for o in pm.outcomes]
pm["o1"] = [norm(o[1]) for o in pm.outcomes]
pm["pair"] = [tuple(sorted(p)) for p in zip(pm.o0, pm.o1)]
print(f"Polymarket CS2 markets in window: {len(pm)} ({pm.market.value_counts().to_dict()})", flush=True)

# ---- 2. HLTV data and point-in-time features ---------------------------------------
ds = DatasetStore()
history = ds.load("cs2_match_history")
history["match_date"] = pd.to_datetime(history.match_date)
new = history[(history.match_date >= CUTOFF) & (history.match_date <= END)]
print(f"HLTV maps in {CUTOFF.date()} -> {END.date()}: {len(new)} across {new.match_id.nunique()} series", flush=True)
priors = build_priors(history, ds.load("cs2_team_rankings"), player_stats=ds.load("cs2_player_map_stats"),
                      veto=ds.load("cs2_match_veto"), h2h=ds.load("cs2_h2h_history"))
cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
mf = priors.merge(history[cols], on=["match_id", "map_name"])
mf = mf[mf.team_a_score != mf.team_b_score].sort_values(["match_date", "match_id"]).reset_index(drop=True)

# ---- 3. model frozen at the cutoff -------------------------------------------------
plan = build_swap_plan(NAMES)
pre = mf[mf.match_date < CUTOFF].reset_index(drop=True)
tr, val, _ = temporal_split_by_series(pre, frac_train=0.88, frac_val=0.12)
es, cal = split_val_for_calibration(pre, val)
X, y = _matrix(pre, NAMES), pre.team_a_won.to_numpy().astype(int)
Xtr, ytr = augment(X[tr], y[tr], plan)
model = _fit(Xtr, ytr, X[es], y[es], 1000, 5)
calib = fit_calibrator(symmetric_predict_proba(model, X[cal], plan), y[cal])
predict = lambda M: calib.predict(symmetric_predict_proba(model, M, plan))
print(f"model trained on {len(tr)} maps up to {pre.match_date.max().date()}", flush=True)

# ---- 4. predictions for the new matches --------------------------------------------
fwd = mf[(mf.match_date >= CUTOFF) & (mf.match_date <= END) & (mf.bo_type == 3)]
preds = []
maps12 = fwd[fwd.map_position_in_series.isin([1, 2])]
for r, q in zip(maps12.itertuples(), predict(_matrix(maps12, NAMES))):
    preds.append({"match_id": r.match_id, "market": f"game{int(r.map_position_in_series)}", "q": q,
                  "y": int(r.team_a_won)})
by_mid = {k: v for k, v in priors.groupby("match_id")}
won = history.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() > len(w) - w.sum()))
for mid in fwd.match_id.unique():
    nf, nodes = build_node_frame(by_mid[mid], 2, NAMES)
    q = series_win_prob(dict(zip(nodes, predict(_matrix(nf, NAMES)))), 2, n_veto_maps=3)
    preds.append({"match_id": mid, "market": "series", "q": float(np.clip(q, 0.01, 0.99)), "y": int(won[mid])})
P = pd.DataFrame(preds)

# ---- 5. match markets, fetch prices ------------------------------------------------
idx = {}
for r in pm.itertuples():
    idx.setdefault((r.slug_date, r.pair, r.market), []).append(r)
first = history[history.match_id.isin(fwd.match_id)].drop_duplicates("match_id").set_index("match_id")
kickoff = history.groupby("match_id").match_time.min()
dropped, tokens = {}, {}
for r in P.itertuples():
    h = first.loc[r.match_id]
    a, b = norm(h.team_a_name), norm(h.team_b_name)
    pair = tuple(sorted([a, b]))
    cands = idx.get((h.match_date.date(), pair, r.market), [])
    if not cands:
        for o in (-1, 1):
            cands = idx.get((h.match_date.date() + dt.timedelta(days=o), pair, r.market), [])
            if cands:
                break
    reason = None
    if len(cands) != 1:
        reason = "no market" if not cands else "ambiguous"
    else:
        mk = cands[0]
        a_is_o0 = mk.o0 == a
        fin = [float(x) for x in mk.final] if mk.final else []
        if len(fin) != 2 or max(fin) < 0.9:
            reason = "not yet settled"
        elif int((fin[0] > fin[1]) == a_is_o0) != r.y:
            reason = "settlement disagrees with HLTV"
        else:
            tokens[(r.match_id, r.market)] = mk.tokens[0 if a_is_o0 else 1]
    if reason:
        dropped[f"{r.market}: {reason}"] = dropped.get(f"{r.market}: {reason}", 0) + 1
print(f"matched markets: {len(tokens)}   dropped: {dropped}", flush=True)

paths = {}
for (mid, market), tok in tokens.items():
    k = pd.Timestamp(kickoff[mid])
    resp = s.get("https://clob.polymarket.com/prices-history",
                 params={"market": tok, "startTs": int((k - pd.Timedelta(days=4)).timestamp()),
                         "endTs": int((k + pd.Timedelta(hours=8)).timestamp()), "fidelity": 1}, timeout=30)
    hist = resp.json().get("history", []) if resp.status_code == 200 else []
    if hist:
        paths[(mid, market)] = (np.array([x["t"] for x in hist]), np.array([float(x["p"]) for x in hist]))
    time.sleep(0.12)


def at(key, ts):
    if key not in paths:
        return np.nan
    t, p = paths[key]
    v = p[t <= ts]
    return float(v[-1]) if len(v) else np.nan


def decided(key, after):
    if key not in paths:
        return np.nan
    t, p = paths[key]
    hit = np.flatnonzero((t > after) & ((p >= 0.97) | (p <= 0.03)))
    return float(t[hit[0]]) if len(hit) else np.nan


ev = []
for r in P.itertuples():
    if (r.match_id, r.market) not in paths:
        continue
    k = pd.Timestamp(kickoff[r.match_id]).timestamp()
    rec = {"match_id": r.match_id, "market": r.market, "q": r.q, "y": r.y}
    if r.market == "game2":
        m1 = decided((r.match_id, "game1"), k)
        rec["price"] = at((r.match_id, "game2"), m1) if m1 == m1 else np.nan
    else:
        rec["price"] = at((r.match_id, r.market), k)
        if r.market == "series":
            rec["price_6h"] = at((r.match_id, "series"), k - 6 * 3600)
    ev.append(rec)
E = pd.DataFrame(ev)
E.to_parquet(DATA_DIR / f"forward_{CUTOFF:%Y%m%d}_{END:%Y%m%d}.parquet", index=False)


# ---- 6. score -----------------------------------------------------------------------
def simulate(p, q, yy, hs, th=0.05):
    fee = TAKER * p * (1 - p)
    ea, eb = q - (p + hs) - fee, (1 - q) - ((1 - p) + hs) - fee
    ba = ea > th
    bb = (eb > th) & ~ba
    pnl = np.where(ba, yy - (p + hs) - fee, 0.0) + np.where(bb, (1 - yy) - ((1 - p) + hs) - fee, 0.0)
    cost = np.where(ba, p + hs, 0.0) + np.where(bb, 1 - p + hs, 0.0)
    return pnl[ba | bb], cost[ba | bb]


lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
CASES = [("game1", "price", "map 1, at kickoff", 0.0125), ("game2", "price", "map 2, at map-1 end", 0.0125),
         ("series", "price_6h", "series, 6h before", 0.01), ("series", "price", "series, at kickoff", 0.01)]
print(f"\n{'case':22s} {'n':>4} {'market LL':>9} {'model LL':>8} {'model - market [95% CI]':>27} {'blend - mkt':>11} "
      f"{'model acc':>9} {'mkt acc':>7} {'trades':>6} {'P&L/share':>9} {'per $':>7}")
for market, col, label, hs in CASES:
    e = E[(E.market == market) & E[col].notna()] if col in E else E.iloc[0:0]
    if len(e) < 10:
        print(f"{label:22s} {len(e):4d}  (too few to score)")
        continue
    yy, q, mk = e.y.to_numpy(), e.q.to_numpy(), np.clip(e[col].to_numpy(), 1e-4, 1 - 1e-4)
    d, lo, hi_ = clustered_bootstrap_delta(yy, q, mk, e.match_id.to_numpy(), log_loss, n_boot=1000)
    blend = 1 / (1 + np.exp(-(lg(mk) + lg(q)) / 2))
    pnl, cost = simulate(mk, q, yy, hs)
    print(f"{label:22s} {len(e):4d} {log_loss(yy, mk):9.4f} {log_loss(yy, q):8.4f} "
          f"{d:+.4f} [{lo:+.3f},{hi_:+.3f}]{'*' if (lo > 0) == (hi_ > 0) else ' '} "
          f"{log_loss(yy, blend) - log_loss(yy, mk):+11.4f} {np.mean((q > 0.5) == yy):9.1%} {np.mean((mk > 0.5) == yy):7.1%} "
          f"{len(pnl):6d} {pnl.sum():+9.2f} {pnl.sum() / max(cost.sum(), 1e-9):+7.1%}")
print("\n(* = 95% CI excludes zero; negative = model better than market)")
