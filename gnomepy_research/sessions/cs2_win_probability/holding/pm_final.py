"""
The gate: does the harvested feature set move the Polymarket gap?

Runs the same comparison twice on the same test series - once with the
pre-harvest baseline feature set, once with all 95 - so the difference is
attributable to the features rather than to anything else that changed.
"""
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from gnomepy_research.sessions.cs2_win_probability.ablate import BASELINE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_features import PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import (
    CS2PreMapModel, _matrix, fit_calibrator, train,
)
from gnomepy_research.sessions.cs2_win_probability.series_dp import build_node_frame, series_win_prob
from gnomepy_research.sessions.cs2_win_probability.series_model import load_series_frame
from gnomepy_research.sessions.cs2_win_probability.splits import (
    calibration_slope, clustered_bootstrap_delta, split_val_for_calibration, temporal_split_by_series,
)
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

logging.basicConfig(level=logging.ERROR)
D = str(DATA_DIR)

history = pd.read_parquet(f"{D}/cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history["match_date"])
priors = pd.read_parquet(f"{D}/priors_full.parquet")

cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
train_df = priors.merge(history[cols], on=["match_id", "map_name"], how="inner")
train_df = train_df[train_df.team_a_score != train_df.team_b_score]
train_df = train_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
print(f"L1 frame: {len(train_df)} rows, {train_df.match_id.nunique()} series, "
      f"{train_df.match_date.min().date()} -> {train_df.match_date.max().date()}", flush=True)

sf = load_series_frame(priors=priors, history=history)
sf = sf[sf.bo_type == 3].reset_index(drop=True)
tr, val, te = temporal_split_by_series(sf, group_col="match_id")
es, cal = split_val_for_calibration(sf, val, group_col="match_id")
y = sf.team_a_won_series.to_numpy().astype(int)

tj = pd.read_parquet(f"{D}/pm_traj_v2.parquet")
HCOLS = [("pm_open", "OPEN"), ("k_12h", "T-12h"), ("k_6h", "T-6h"), ("k_3h", "T-3h"),
         ("k_1h", "T-1h"), ("k_0h", "kickoff")]
j = sf.loc[te].merge(tj[["match_id"] + [c for c, _ in HCOLS]], on="match_id", how="inner")
print(f"test series matched to Polymarket: {len(j)}", flush=True)
if len(j) == 0:
    sys.exit("no overlap between the test split and the Polymarket sample")

yy = j.team_a_won_series.to_numpy().astype(int)
g = j.match_id.to_numpy()
by_mid = {k: v for k, v in priors.groupby("match_id")}


def dp_for(model, names, ids):
    frames, idx = [], []
    for mid in ids:
        g_ = by_mid.get(mid)
        if g_ is None or g_.empty:
            frames.append(None); idx.append(None); continue
        nf, nodes = build_node_frame(g_, 2, names)
        frames.append(nf); idx.append(nodes)
    st = pd.concat([f for f in frames if f is not None], ignore_index=True)
    pr = model.predict_batch(_matrix(st, names))
    out, cur = [], 0
    for f, nodes in zip(frames, idx):
        if f is None:
            out.append(0.5); continue
        ch = pr[cur:cur + len(f)]; cur += len(f)
        out.append(series_win_prob(dict(zip(nodes, ch)), 2, n_veto_maps=3))
    return np.array(out)


results = {}
for label, names in (("baseline(34)", list(BASELINE_NAMES)), ("harvested(95)", list(PRE_MAP_FEATURE_NAMES))):
    path = f"{D}/cs2_l1_{label.split('(')[0]}.xgb"
    m = train(train_df, path, feature_names=names, report=False)
    model = CS2PreMapModel(path)
    c = dp_for(model, names, sf.loc[cal].match_id.tolist())
    calib = fit_calibrator(c, y[sf.index.get_indexer(cal)])
    p = calib.predict(dp_for(model, names, j.match_id.tolist()))
    results[label] = p
    print(f"\n{label}: L1 auc={m['auc']:.4f} ll={m['log_loss']:.4f} cal={m['calibrator']} | "
          f"series auc={roc_auc_score(yy,p):.4f} ll={log_loss(yy,p):.4f} slope={calibration_slope(yy,p):.3f}",
          flush=True)

print(f"\n{'horizon':>8} {'n':>4} {'market':>8} | {'base':>8} {'gap':>8} | {'harvest':>8} {'gap':>8}"
      f"   {'harvested vs market [CI]':>30}")
for col, lab in HCOLS:
    s = j[col].notna().to_numpy()
    if s.sum() < 60:
        continue
    mk = np.clip(j[col].to_numpy()[s], 1e-4, 1 - 1e-4)
    mk_ll = log_loss(yy[s], mk)
    pb, ph = results["baseline(34)"][s], results["harvested(95)"][s]
    d, lo, hi = clustered_bootstrap_delta(yy[s], ph, mk, g[s], log_loss, n_boot=400)
    print(f"{lab:>8} {int(s.sum()):4d} {mk_ll:8.4f} | {log_loss(yy[s],pb):8.4f} "
          f"{log_loss(yy[s],pb)-mk_ll:+8.4f} | {log_loss(yy[s],ph):8.4f} "
          f"{log_loss(yy[s],ph)-mk_ll:+8.4f}   {d:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"{'SIG' if (lo>0)==(hi>0) else 'ns'}", flush=True)

print("\n(negative gap = model beats market)")
print(f"\n{'horizon':>8}  harvested - baseline (paired)")
for col, lab in HCOLS:
    s = j[col].notna().to_numpy()
    if s.sum() < 60:
        continue
    d, lo, hi = clustered_bootstrap_delta(
        yy[s], results["harvested(95)"][s], results["baseline(34)"][s], g[s], log_loss, n_boot=400)
    print(f"{lab:>8}  {d:+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if (lo>0)==(hi>0) else 'ns'}", flush=True)

print("\nhow often does each side commit, and how often is it right?  (vs T-6h market)")
s = j["k_6h"].notna().to_numpy()
mk = j["k_6h"].to_numpy()[s]
cands = {"market T-6h": mk, "baseline": results["baseline(34)"][s], "harvested": results["harvested(95)"][s]}
print(f"{'':>13}" + "".join(f"{th:>16.2f}" for th in (0.65, 0.75, 0.85)))
for name, p in cands.items():
    cells = []
    for th in (0.65, 0.75, 0.85):
        sel = np.maximum(p, 1 - p) > th
        acc = np.mean((p[sel] > 0.5) == yy[s][sel].astype(bool)) if sel.sum() else float("nan")
        cells.append(f"{int(sel.sum()):5d} @ {acc:5.1%}")
    print(f"{name:>13}" + "".join(f"{c:>16}" for c in cells))
