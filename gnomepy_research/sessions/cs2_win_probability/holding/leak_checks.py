"""
Leakage checks on the six-month backtest.

1. Map-3 asymmetry. In a 2-0 series map 3 is never played, so the series DP
   priced the decider as an unknown map; in a 2-1 series it used the real map-3
   row. The input therefore depended on whether the series went the distance -
   an outcome. Fix: treat map 3 as an unknown map in every series.
2. Veto and lineup timing. The veto happens close to kickoff and late stand-ins
   are announced late, so at T-12h/T-6h the market cannot know the maps or a
   last-minute lineup. A strict model drops every map, veto and lineup feature
   and prices all three maps as unknown.
3. Legacy priors. compute_priors_for_row predates pit.py and was never put
   through a truncation test.

If the edge survives (1) at kickoff and (2) at 12h/6h, it is not these leaks.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from gnomepy_research.pipelines.hltv_cs2.build_priors import compute_priors_for_row
from gnomepy_research.pipelines.hltv_cs2.context_features import VETO_FEATURES
from gnomepy_research.pipelines.hltv_cs2.player_form import PLAYER_FORM_FEATURES
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR
from gnomepy_research.sessions.cs2_win_probability.pre_map_features import PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import _fit, _matrix, fit_calibrator
from gnomepy_research.sessions.cs2_win_probability.series_dp import series_nodes, series_win_prob
from gnomepy_research.sessions.cs2_win_probability.splits import (
    clustered_bootstrap_delta, split_val_for_calibration, temporal_split_by_series,
)
from gnomepy_research.sessions.cs2_win_probability.symmetry import augment, build_swap_plan, symmetric_predict_proba

logging.basicConfig(level=logging.WARNING)
MONTHS = pd.date_range("2026-03-01", "2026-09-01", freq="MS")
TAKER = 0.07
MAP_SPECIFIC = ["team_a_map_winrate_long", "team_b_map_winrate_long", "team_a_map_winrate_short",
                "team_b_map_winrate_short", "elo_map_diff", "elo_map_resid_diff"]
VETO_DEPENDENT = ["team_a_picked_map", "is_decider"] + VETO_FEATURES + MAP_SPECIFIC
LINEUP_DEPENDENT = list(PLAYER_FORM_FEATURES)
FULL = list(PRE_MAP_FEATURE_NAMES)
STRICT = [n for n in FULL if not n.startswith("map_de_") and n not in set(VETO_DEPENDENT + LINEUP_DEPENDENT)]

history = pd.read_parquet(DATA_DIR / "cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history.match_date)
priors = pd.read_parquet(DATA_DIR / "priors_full.parquet")
cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
mf = priors.merge(history[cols], on=["match_id", "map_name"])
mf = mf[mf.team_a_score != mf.team_b_score].sort_values(["match_date", "match_id"]).reset_index(drop=True)
by_mid = {k: v for k, v in priors.groupby("match_id")}
series_won = history.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() > len(w) - w.sum()))

print(f"strict feature set: {len(STRICT)} of {len(FULL)} (drops map one-hots, veto, map-specific and lineup features)")

# ---- check 3: legacy priors under truncation ---------------------------------------
rank = pd.read_parquet(DATA_DIR / "cs2_team_rankings.parquet").sort_values("date").reset_index(drop=True)
hs = history.sort_values("match_date").reset_index(drop=True)
legacy = ["team_a_map_winrate_long", "team_b_map_winrate_long", "team_a_map_winrate_short", "team_b_map_winrate_short",
          "team_a_overall_winrate", "team_b_overall_winrate", "h2h_win_rate", "recent_form_diff", "rank_diff"]
sample = hs[hs.match_date > "2026-03-01"].sample(150, random_state=0)
mism = 0
for r in sample.itertuples():
    row = hs.loc[r.Index]
    full = compute_priors_for_row(row, hs, rank)
    trunc_hist = hs[(hs.match_date < row.match_date) | (hs.match_id == row.match_id)].reset_index(drop=True)
    trunc = compute_priors_for_row(row, trunc_hist, rank[rank.date < row.match_date + pd.Timedelta(days=1)])
    if full is None or trunc is None:
        continue
    for c in legacy:
        a, b = full.get(c), trunc.get(c)
        if not ((a is None and b is None) or (a != a and b != b) or a == b):
            mism += 1
            break
print(f"\nCHECK 3 legacy priors: {mism} of {len(sample)} sampled rows change when later matches are withheld")


# ---- monthly refits for checks 1 and 2 ---------------------------------------------
def train_before(cutoff, names):
    plan = build_swap_plan(names)
    pre = mf[mf.match_date < cutoff].reset_index(drop=True)
    tr, val, _ = temporal_split_by_series(pre, frac_train=0.88, frac_val=0.12)
    es, cal = split_val_for_calibration(pre, val)
    X, y = _matrix(pre, names), pre.team_a_won.to_numpy().astype(int)
    Xtr, ytr = augment(X[tr], y[tr], plan)
    model = _fit(Xtr, ytr, X[es], y[es], 1000, 5)
    calib = fit_calibrator(symmetric_predict_proba(model, X[cal], plan), y[cal])
    return lambda M: calib.predict(symmetric_predict_proba(model, M, plan))


def node_frame(series_priors, unknown_positions):
    """Series lattice where the listed map positions are priced as unknown maps."""
    by_pos = {int(r.map_position_in_series): r for r in series_priors.itertuples()}
    first = by_pos[min(by_pos)]
    rows, nodes = [], []
    for a, b in series_nodes(2):
        pos = a + b + 1
        src = by_pos.get(pos, first) if pos not in unknown_positions else first
        row = {c: getattr(src, c, np.nan) for c in series_priors.columns}
        row.update({"team_a_series_score": float(a), "team_b_series_score": float(b),
                    "map_position_in_series": float(pos)})
        if pos in unknown_positions or pos not in by_pos:
            row.update({"is_decider": float(pos == 3), "team_a_picked_map": np.nan, "map_name": ""})
            for c in MAP_SPECIFIC + VETO_FEATURES:
                row[c] = np.nan
        rows.append(row); nodes.append((a, b))
    return pd.DataFrame(rows), nodes


def series_dp(predict, names, ids, unknown_positions):
    frames, idx = [], []
    for mid in ids:
        f, n = node_frame(by_mid[mid], unknown_positions)
        frames.append(f); idx.append(n)
    pr = predict(_matrix(pd.concat(frames, ignore_index=True), names))
    out, cur = [], 0
    for f, nodes in zip(frames, idx):
        ch = pr[cur:cur + len(f)]; cur += len(f)
        out.append(series_win_prob(dict(zip(nodes, ch)), 2, n_veto_maps=3))
    return np.clip(np.array(out), 0.01, 0.99)


rows = []
for start in MONTHS:
    end = start + pd.offsets.MonthBegin(1)
    ids = [m for m in mf[(mf.match_date >= start) & (mf.match_date < end) & (mf.bo_type == 3)].match_id.unique()
           if m in by_mid]
    p_full = train_before(start, FULL)
    p_strict = train_before(start, STRICT)
    q_sym = series_dp(p_full, FULL, ids, unknown_positions={3})
    q_strict = series_dp(p_strict, STRICT, ids, unknown_positions={1, 2, 3})
    for mid, a, b in zip(ids, q_sym, q_strict):
        rows.append({"month": f"{start:%Y-%m}", "match_id": mid, "q_map3_sym": a, "q_strict": b, "y": series_won[mid]})
    print(f"{start:%Y-%m}: {len(ids)} series", flush=True)
Q = pd.DataFrame(rows)
Q.to_parquet(DATA_DIR / "leak_checks_preds.parquet", index=False)

old = pd.read_parquet(DATA_DIR / "backtest_6mo_preds.parquet")
old = old[old.market == "series"][["match_id", "q"]].rename(columns={"q": "q_original"})
E = pd.read_parquet(DATA_DIR / "backtest_6mo_eval.parquet")
E = E[E.market == "series"][["match_id", "open", "k12", "k6", "k1", "k0"]]
J = Q.merge(old, on="match_id").merge(E, on="match_id")
y = J.y.to_numpy()
print(f"\nseries with all predictions and a Polymarket path: {len(J)}")
print(f"original vs map-3-symmetric predictions: mean |diff| {np.abs(J.q_original - J.q_map3_sym).mean():.4f}")


def simulate(p, q, yy, hs=0.01, th=0.05):
    fee = TAKER * p * (1 - p)
    ea, eb = q - (p + hs) - fee, (1 - q) - ((1 - p) + hs) - fee
    ba = ea > th
    bb = (eb > th) & ~ba
    pnl = np.where(ba, yy - (p + hs) - fee, 0.0) + np.where(bb, (1 - yy) - ((1 - p) + hs) - fee, 0.0)
    cost = np.where(ba, p + hs, 0.0) + np.where(bb, 1 - p + hs, 0.0)
    return pnl[ba | bb], cost[ba | bb]


lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
print(f"\n{'variant':>22} {'horizon':>8} {'n':>5} {'mdl LL':>7} {'mkt LL':>7} {'blend-mkt':>10} {'trades':>7} {'P&L':>8} {'per $':>7} {'months +':>9}")
for qcol, label in (("q_original", "original"), ("q_map3_sym", "map-3 symmetric"), ("q_strict", "strict (no veto/lineup)")):
    for hcol in ("k12", "k6", "k0"):
        s = J[hcol].notna().to_numpy()
        mk, q, yy, g = np.clip(J[hcol].to_numpy()[s], 1e-4, 1 - 1e-4), J[qcol].to_numpy()[s], y[s], J.match_id.to_numpy()[s]
        blend = 1 / (1 + np.exp(-(lg(mk) + lg(q)) / 2))
        db, lo, hi = clustered_bootstrap_delta(yy, blend, mk, g, log_loss, n_boot=300)
        pnl, cost = simulate(mk, q, yy)
        months = J[s].assign(q=q, mk=mk).groupby("month").apply(
            lambda d: simulate(d.mk.to_numpy(), d.q.to_numpy(), d.y.to_numpy())[0].sum(), include_groups=False)
        print(f"{label:>22} {hcol:>8} {s.sum():5d} {log_loss(yy, q):7.4f} {log_loss(yy, mk):7.4f} "
              f"{db:+9.4f}{'*' if (lo>0)==(hi>0) else ' '} {len(pnl):7d} {pnl.sum():+8.2f} {pnl.sum()/cost.sum():+7.1%} "
              f"{int((months > 0).sum())}/{len(months)}")
