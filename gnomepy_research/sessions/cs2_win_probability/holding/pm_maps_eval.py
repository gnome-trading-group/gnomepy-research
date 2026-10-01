"""
#2a: L1 map predictions vs Polymarket map-winner markets, priced before each map starts.

Map 1 is priced at kickoff (and earlier). Map 2 is priced at the moment map 1
was decided, which the map-1 market itself timestamps: the first minute after
kickoff its price leaves [0.03, 0.97]. That moment can precede the formal end of
map 1 by a few rounds, so it errs early - safe, since early cannot leak map-2
play into the market's price.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from gnomepy_research.sessions.cs2_win_probability.ablate import BASELINE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_features import PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import CS2PreMapModel, _matrix
from gnomepy_research.sessions.cs2_win_probability.splits import calibration_slope, clustered_bootstrap_delta
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

D = str(DATA_DIR)
TAKER = 0.07

history = pd.read_parquet(f"{D}/cs2_match_history.parquet")
priors = pd.read_parquet(f"{D}/priors_full.parquet")
mm = pd.read_parquet(f"{D}/pm_maps_matched.parquet")
paths = pd.read_parquet(f"{D}/pm_maps_paths.parquet").sort_values(["match_id", "map_pos", "t"])

rows = priors.merge(history[["match_id", "map_name", "team_a_won"]],
                    on=["match_id", "map_name"], how="inner")
rows = rows.merge(mm[["match_id", "map_pos", "volume", "kickoff"]],
                  left_on=["match_id", "map_position_in_series"], right_on=["match_id", "map_pos"], how="inner")
for label, names, path in (("harvested", list(PRE_MAP_FEATURE_NAMES), f"{D}/cs2_l1_harvested.xgb"),
                           ("baseline", list(BASELINE_NAMES), f"{D}/cs2_l1_baseline.xgb")):
    rows[f"q_{label}"] = CS2PreMapModel(path).predict_batch(_matrix(rows, names))

by_key = {k: (g.t.to_numpy(), g.p_a.to_numpy()) for k, g in paths.groupby(["match_id", "map_pos"])}


def price_at(key, ts):
    if key not in by_key:
        return np.nan
    t, p = by_key[key]
    prior = p[t <= ts]
    return float(prior[-1]) if len(prior) else np.nan


def decided_at(key, after):
    if key not in by_key:
        return np.nan
    t, p = by_key[key]
    hit = np.flatnonzero((t > after) & ((p >= 0.97) | (p <= 0.03)))
    return float(t[hit[0]]) if len(hit) else np.nan


recs = []
for r in rows.itertuples():
    k = pd.Timestamp(r.kickoff).timestamp()
    rec = {"match_id": r.match_id, "map_pos": r.map_pos, "y": int(r.team_a_won), "volume": r.volume,
           "q_h": r.q_harvested, "q_b": r.q_baseline}
    if r.map_pos == 1:
        rec["mk_pre"] = price_at((r.match_id, 1), k)
        rec["mk_6h"] = price_at((r.match_id, 1), k - 6 * 3600)
        rec["mk_1h"] = price_at((r.match_id, 1), k - 3600)
    else:
        m1_end = decided_at((r.match_id, 1), k)
        rec["mk_pre"] = price_at((r.match_id, 2), m1_end) if m1_end == m1_end else np.nan
        rec["mk_pre5"] = price_at((r.match_id, 2), m1_end + 300) if m1_end == m1_end else np.nan
        rec["gap_min"] = (m1_end - k) / 60 if m1_end == m1_end else np.nan
    recs.append(rec)
ev = pd.DataFrame(recs)
print(f"maps with a model prediction and a market path: {len(ev)}  by position {ev.map_pos.value_counts().sort_index().to_dict()}")
m2 = ev[ev.map_pos == 2]
print(f"map 1 decided at median {m2.gap_min.median():.0f} min after HLTV kickoff "
      f"(IQR {m2.gap_min.quantile(.25):.0f}-{m2.gap_min.quantile(.75):.0f})")


def simulate(p, q, y, half_spread, threshold):
    fee = TAKER * p * (1 - p)
    ea = q - (p + half_spread) - fee
    eb = (1 - q) - ((1 - p) + half_spread) - fee
    ba, bb = ea > threshold, (eb > threshold) & ~(ea > threshold)
    pnl = np.where(ba, y - (p + half_spread) - fee, 0.0) + np.where(bb, (1 - y) - ((1 - p) + half_spread) - fee, 0.0)
    cost = np.where(ba, p + half_spread, 0.0) + np.where(bb, 1 - p + half_spread, 0.0)
    t = ba | bb
    return pnl[t], cost[t]


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile([rng.choice(x, len(x)).sum() for _ in range(n)], [2.5, 97.5]) if len(x) > 5 else (np.nan, np.nan)


lg = lambda x: np.log(x / (1 - x))
CASES = [(1, "mk_6h", "map 1, 6h before kickoff"), (1, "mk_1h", "map 1, 1h before kickoff"),
         (1, "mk_pre", "map 1, at kickoff"), (2, "mk_pre", "map 2, as map 1 is decided"),
         (2, "mk_pre5", "map 2, 5 min after that")]
print(f"\n{'case':>28} {'n':>4} {'mkt LL':>7} {'harv LL':>8} {'base LL':>8} {'harv-mkt [CI]':>25} {'blend-mkt':>10} {'harv AUC':>8} {'mkt AUC':>8}")
for pos, col, lab in CASES:
    e = ev[(ev.map_pos == pos) & ev[col].notna()]
    if len(e) < 60:
        continue
    y, mk = e.y.to_numpy(), np.clip(e[col].to_numpy(), 1e-4, 1 - 1e-4)
    qh, qb, g = e.q_h.to_numpy(), e.q_b.to_numpy(), e.match_id.to_numpy()
    d, lo, hi = clustered_bootstrap_delta(y, qh, mk, g, log_loss, n_boot=400)
    blend = 1 / (1 + np.exp(-(lg(mk) + lg(qh)) / 2))
    db, blo, bhi = clustered_bootstrap_delta(y, blend, mk, g, log_loss, n_boot=400)
    print(f"{lab:>28} {len(e):4d} {log_loss(y, mk):7.4f} {log_loss(y, qh):8.4f} {log_loss(y, qb):8.4f} "
          f"{d:+.4f}[{lo:+.3f},{hi:+.3f}]{'*' if (lo>0)==(hi>0) else ' '} {db:+.4f}{'*' if (blo>0)==(bhi>0) else ' '} "
          f"{roc_auc_score(y, qh):8.4f} {roc_auc_score(y, mk):8.4f}")
print("  (* = 95% CI excludes zero; negative = model/blend better than market)")

print(f"\nfee-aware P&L, 1 share per map, 5c edge threshold (taker fee 0.07*p*(1-p))")
print(f"{'case':>28} {'half-spread':>11} {'trades':>7} {'P&L $':>7} {'95% CI':>17} {'per $':>7}")
for pos, col, lab in CASES:
    e = ev[(ev.map_pos == pos) & ev[col].notna()]
    if len(e) < 60:
        continue
    for hs in (0.01, 0.0125, 0.025):
        pnl, cost = simulate(e[col].to_numpy(), e.q_h.to_numpy(), e.y.to_numpy(), hs, 0.05)
        lo, hi = boot(pnl)
        print(f"{lab:>28} {hs:11.4f} {len(pnl):7d} {pnl.sum():+7.2f} [{lo:+6.2f},{hi:+6.2f}] {pnl.sum()/max(cost.sum(),1e-9):+7.1%}")

ev.to_parquet(f"{D}/pm_maps_eval.parquet", index=False)
