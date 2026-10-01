"""Walk-forward: is series overconfidence persistent (fixable) or month-to-month noise?"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from gnomepy_research.sessions.cs2_win_probability.pre_map_features import PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import (
    CS2PreMapModel, _calibrated, _fit_temperature, _matrix, train,
)
from gnomepy_research.sessions.cs2_win_probability.series_dp import build_node_frame, series_win_prob
from gnomepy_research.sessions.cs2_win_probability.series_model import load_series_frame
from gnomepy_research.sessions.cs2_win_probability.splits import calibration_slope, walk_forward_folds
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

logging.basicConfig(level=logging.ERROR)
D = str(DATA_DIR)
NAMES = list(PRE_MAP_FEATURE_NAMES)

history = pd.read_parquet(f"{D}/cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history["match_date"])
priors = pd.read_parquet(f"{D}/priors_full.parquet")
cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
mf = priors.merge(history[cols], on=["match_id", "map_name"], how="inner")
mf = mf[mf.team_a_score != mf.team_b_score].sort_values(["match_date", "match_id"]).reset_index(drop=True)
sf = load_series_frame(priors=priors, history=history)
sf = sf[sf.bo_type == 3].reset_index(drop=True)
y_series = dict(zip(sf.match_id, sf.team_a_won_series.astype(int)))
by_mid = {k: v for k, v in priors.groupby("match_id")}


def raw_dp(model, ids):
    frames, idx = [], []
    for mid in ids:
        nf, nodes = build_node_frame(by_mid[mid], 2, NAMES)
        frames.append(nf); idx.append(nodes)
    pr = model.predict_batch(_matrix(pd.concat(frames, ignore_index=True), NAMES))
    out, cur = [], 0
    for f, nodes in zip(frames, idx):
        ch = pr[cur:cur + len(f)]; cur += len(f)
        out.append(series_win_prob(dict(zip(nodes, ch)), 2, n_veto_maps=3))
    return np.array(out)


def slope_ci(y, p, n=400, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(calibration_slope(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


rows, oos = [], []
for k, (tr_idx, te_idx) in enumerate(walk_forward_folds(mf, n_folds=6, min_train_frac=0.35)):
    path = f"{D}/cs2_wf_{k}.xgb"
    m = train(mf.loc[tr_idx].reset_index(drop=True), path, feature_names=NAMES, report=False)
    model = CS2PreMapModel(path)
    te = mf.loc[te_idx]
    pm = model.predict_batch(_matrix(te, NAMES))
    ym = te.team_a_won.to_numpy().astype(int)
    ids = [i for i in te.match_id.unique() if i in y_series]
    ps = raw_dp(model, ids)
    ys = np.array([y_series[i] for i in ids])
    lo, hi = slope_ci(ys, ps)
    rows.append({"fold": k, "from": te.match_date.min().date(), "to": te.match_date.max().date(),
                 "map_slope": calibration_slope(ym, pm), "series_n": len(ids),
                 "series_slope": calibration_slope(ys, ps), "lo": lo, "hi": hi, "l1_T": m["temperature"]})
    oos.append(pd.DataFrame({"fold": k, "match_id": ids, "raw": ps, "y": ys}))
    print(f"fold {k}: {rows[-1]['from']} -> {rows[-1]['to']}  map slope {rows[-1]['map_slope']:.3f}  "
          f"series slope {rows[-1]['series_slope']:.3f} [{lo:.3f},{hi:.3f}] n={len(ids)}", flush=True)

oos = pd.concat(oos, ignore_index=True)
oos.to_parquet(f"{D}/series_wf_oos.parquet", index=False)
pooled = oos[oos.fold < oos.fold.max()]
last = oos[oos.fold == oos.fold.max()]
T = _fit_temperature(pooled.raw.to_numpy(), pooled.y.to_numpy())
print(f"\npooled out-of-sample series slope (folds 0..{oos.fold.max()-1}, n={len(pooled)}): "
      f"{calibration_slope(pooled.y.to_numpy(), pooled.raw.to_numpy()):.3f}  -> series T fit on them: {T:.3f}")
y_l, r_l = last.y.to_numpy(), last.raw.to_numpy()
print(f"last fold n={len(last)}: raw ll {log_loss(y_l, np.clip(r_l,.01,.99)):.4f} slope {calibration_slope(y_l, r_l):.3f}"
      f"  | pooled-T ll {log_loss(y_l, _calibrated(T, r_l)):.4f} slope {calibration_slope(y_l, _calibrated(T, r_l)):.3f}")
