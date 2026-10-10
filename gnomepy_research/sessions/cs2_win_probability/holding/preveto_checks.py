"""
Pre-veto models: what can be traded when the veto is not public before kickoff?

HLTV posts the veto when the match goes live (Spirit vs MOUZ, 2026-10-05), so at
the scheduled kickoff neither we nor the market know which map map 1 is. The
map-1 backtest used the full model, which did know - part of that edge may be
lookahead. Two questions, one walk-forward:

1. Map 1, veto-blind: does a model with no map identity, picker or veto
   features still beat the map-1 market at kickoff?
2. Series, Phase 0 gate: does STRICT + player form (the chosen pre-veto model)
   do at least as well as STRICT, the variant the leak check validated?

Feature sets: STRICT drops map one-hots, veto features, map-specific records
and player form; STRICT_FORM adds player form back (announced lineups are
public before kickoff).
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

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
UNKNOWN_MAP = ["team_a_picked_map", "is_decider"] + VETO_FEATURES + MAP_SPECIFIC
FULL = list(PRE_MAP_FEATURE_NAMES)
STRICT = [n for n in FULL if not n.startswith("map_de_") and n not in set(UNKNOWN_MAP + PLAYER_FORM_FEATURES)]
STRICT_FORM = STRICT + list(PLAYER_FORM_FEATURES)
SETS = {"strict": STRICT, "strict_form": STRICT_FORM}

history = pd.read_parquet(DATA_DIR / "cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history.match_date)
priors = pd.read_parquet(DATA_DIR / "priors_full.parquet")
cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
mf = priors.merge(history[cols], on=["match_id", "map_name"])
mf = mf[mf.team_a_score != mf.team_b_score].sort_values(["match_date", "match_id"]).reset_index(drop=True)
by_mid = {k: v for k, v in priors.groupby("match_id")}
series_won = history.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() > len(w) - w.sum()))
map1_won = history[history.map_position_in_series == 1].set_index("match_id").team_a_won
print(f"STRICT {len(STRICT)} features, STRICT_FORM {len(STRICT_FORM)}", flush=True)


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


def unknown_map_frame(series_priors):
    """Every lattice node priced on an unknown map, carrying series-level features from map 1."""
    first = series_priors.sort_values("map_position_in_series").iloc[0]
    rows, nodes = [], []
    for a, b in series_nodes(2):
        row = first.to_dict()
        row.update({"team_a_series_score": float(a), "team_b_series_score": float(b),
                    "map_position_in_series": float(a + b + 1), "map_name": "",
                    "is_decider": float(a + b + 1 == 3), "team_a_picked_map": np.nan})
        for c in VETO_FEATURES + MAP_SPECIFIC:
            row[c] = np.nan
        rows.append(row)
        nodes.append((a, b))
    return pd.DataFrame(rows), nodes


rows = []
for start in MONTHS:
    end = start + pd.offsets.MonthBegin(1)
    ids = [m for m in mf[(mf.match_date >= start) & (mf.match_date < end) & (mf.bo_type == 3)].match_id.unique()
           if m in by_mid]
    frames, nodes = zip(*(unknown_map_frame(by_mid[m]) for m in ids))
    stacked = pd.concat(frames, ignore_index=True)
    out = {"match_id": ids}
    for name, names in SETS.items():
        pr = train_before(start, names)(_matrix(stacked, names))
        series_q, map1_q, cur = [], [], 0
        for f, nd in zip(frames, nodes):
            ch = dict(zip(nd, pr[cur:cur + len(f)]))
            cur += len(f)
            series_q.append(series_win_prob(ch, 2, n_veto_maps=3))
            map1_q.append(ch[(0, 0)])
        out[f"series_{name}"] = np.clip(series_q, 0.01, 0.99)
        out[f"map1_{name}"] = np.clip(map1_q, 0.01, 0.99)
    df = pd.DataFrame(out).assign(month=f"{start:%Y-%m}")
    rows.append(df)
    print(f"{start:%Y-%m}: {len(ids)} series", flush=True)
Q = pd.concat(rows, ignore_index=True)
Q["y_series"] = Q.match_id.map(series_won)
Q["y_map1"] = Q.match_id.map(map1_won)
Q.to_parquet(DATA_DIR / "preveto_checks_preds.parquet", index=False)

E = pd.read_parquet(DATA_DIR / "backtest_6mo_eval.parquet")
full = pd.read_parquet(DATA_DIR / "backtest_6mo_preds.parquet")
full = full.pivot_table(index="match_id", columns="market", values="q").rename(
    columns={"series": "series_full", "game1": "map1_full"})[["series_full", "map1_full"]]
Q = Q.join(full, on="match_id")


def simulate(p, q, y, hs, th):
    fee = TAKER * p * (1 - p)
    ea, eb = q - (p + hs) - fee, (1 - q) - ((1 - p) + hs) - fee
    ba = ea > th
    bb = (eb > th) & ~ba
    pnl = np.where(ba, y - (p + hs) - fee, 0.0) + np.where(bb, (1 - y) - ((1 - p) + hs) - fee, 0.0)
    cost = np.where(ba, p + hs, 0.0) + np.where(bb, 1 - p + hs, 0.0)
    return pnl[ba | bb], cost[ba | bb]


lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
CASES = [("game1", "k0", "map1", "y_map1", 0.0125, 0.10, "map 1 @ kickoff"),
         ("series", "k12", "series", "y_series", 0.010, 0.05, "series @ 12h"),
         ("series", "k6", "series", "y_series", 0.010, 0.05, "series @ 6h"),
         ("series", "k0", "series", "y_series", 0.010, 0.05, "series @ kickoff")]
print(f"\n{'case':18s} {'model':12s} {'n':>5} {'mdl LL':>7} {'mkt LL':>7} {'gap':>8} {'blend-mkt':>10} "
      f"{'trades':>6} {'P&L':>8} {'per $':>7} {'months +':>8}")
for market, col, prefix, ycol, hs, th, label in CASES:
    e = E[(E.market == market) & E[col].notna()][["match_id", col]]
    j = Q.merge(e, on="match_id").dropna(subset=[ycol])
    y, mk, g = j[ycol].to_numpy().astype(int), np.clip(j[col].to_numpy(), 1e-4, 1 - 1e-4), j.match_id.to_numpy()
    for variant in ("full", "strict", "strict_form"):
        q = j[f"{prefix}_{variant}"].to_numpy()
        ok = ~np.isnan(q)
        blend = 1 / (1 + np.exp(-(lg(mk[ok]) + lg(q[ok])) / 2))
        db, lo, hi = clustered_bootstrap_delta(y[ok], blend, mk[ok], g[ok], log_loss, n_boot=300)
        pnl, cost = simulate(mk[ok], q[ok], y[ok], hs, th)
        months = j[ok].assign(q=q[ok], mk=mk[ok]).groupby("month").apply(
            lambda d: simulate(d.mk.to_numpy(), d.q.to_numpy(), d[ycol].to_numpy().astype(int), hs, th)[0].sum(),
            include_groups=False)
        print(f"{label:18s} {variant:12s} {ok.sum():5d} {log_loss(y[ok], q[ok]):7.4f} {log_loss(y[ok], mk[ok]):7.4f} "
              f"{log_loss(y[ok], q[ok]) - log_loss(y[ok], mk[ok]):+8.4f} {db:+9.4f}{'*' if (lo > 0) == (hi > 0) else ' '} "
              f"{len(pnl):6d} {pnl.sum():+8.2f} {pnl.sum() / max(cost.sum(), 1e-9):+7.1%} "
              f"{int((months > 0).sum())}/{len(months)}")
    print()
