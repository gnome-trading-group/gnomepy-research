"""Why is the series-level probability overconfident (slope 0.86) when the map level is not?"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from gnomepy_research.sessions.cs2_win_probability.pre_map_features import PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import CS2PreMapModel, _matrix, fit_calibrator
from gnomepy_research.sessions.cs2_win_probability.series_dp import build_node_frame, series_win_prob
from gnomepy_research.sessions.cs2_win_probability.series_model import load_series_frame
from gnomepy_research.sessions.cs2_win_probability.splits import (
    calibration_slope, split_val_for_calibration, temporal_split_by_series,
)
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

logging.basicConfig(level=logging.ERROR)
D = str(DATA_DIR)
NAMES = list(PRE_MAP_FEATURE_NAMES)

history = pd.read_parquet(f"{D}/cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history["match_date"])
priors = pd.read_parquet(f"{D}/priors_full.parquet")
model = CS2PreMapModel(f"{D}/cs2_l1_harvested.xgb")

cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
mf = priors.merge(history[cols], on=["match_id", "map_name"], how="inner")
mf = mf[mf.team_a_score != mf.team_b_score].sort_values(["match_date", "match_id"]).reset_index(drop=True)
mtr, mval, mte = temporal_split_by_series(mf)

sf = load_series_frame(priors=priors, history=history)
sf = sf[sf.bo_type == 3].reset_index(drop=True)
str_, sval, ste = temporal_split_by_series(sf, group_col="match_id")
ses, scal = split_val_for_calibration(sf, sval, group_col="match_id")

print("split boundaries (first date of each block)")
for name, f, parts in (("map  frame", mf, (mtr, mval, mte)), ("series BO3", sf, (str_, sval, ste))):
    print(f"  {name}: train {f.iloc[parts[0]].match_date.min().date()}  val {f.iloc[parts[1]].match_date.min().date()}"
          f"  test {f.iloc[parts[2]].match_date.min().date()}")
l1_train_ids = set(mf.iloc[mtr].match_id)
print(f"  series-val series that sit inside L1's TRAIN block: "
      f"{np.mean(sf.iloc[sval].match_id.isin(l1_train_ids)):.1%}")
print(f"  series cal set size: {len(scal)}   es: {len(ses)}   test: {len(ste)}")

mt = mf.iloc[mte]
p_map = model.predict_batch(_matrix(mt, NAMES))
y_map = mt.team_a_won.to_numpy().astype(int)
print(f"\nL1 map-level on its test block: slope {calibration_slope(y_map, p_map):.3f}  "
      f"ll {log_loss(y_map, p_map):.4f}  (n={len(mt)})")

mt = mt.assign(p=p_map, resid=y_map - p_map)
both = mt[mt.bo_type == 3].pivot_table(index="match_id", columns="map_position_in_series", values="resid")
if 1 in both and 2 in both:
    pair = both[[1, 2]].dropna()
    print(f"residual correlation map1 vs map2 within a series: {np.corrcoef(pair[1], pair[2])[0,1]:+.3f}  (n={len(pair)})")

by_mid = {k: v for k, v in priors.groupby("match_id")}


def raw_dp(ids, shrink=1.0):
    frames, idx = [], []
    for mid in ids:
        nf, nodes = build_node_frame(by_mid[mid], 2, NAMES)
        frames.append(nf); idx.append(nodes)
    pr = model.predict_batch(_matrix(pd.concat(frames, ignore_index=True), NAMES))
    if shrink != 1.0:
        lg = np.log(pr / (1 - pr)) / shrink
        pr = 1 / (1 + np.exp(-lg))
    out, cur = [], 0
    for f, nodes in zip(frames, idx):
        ch = pr[cur:cur + len(f)]; cur += len(f)
        out.append(series_win_prob(dict(zip(nodes, ch)), 2, n_veto_maps=3))
    return np.array(out)


y = sf.team_a_won_series.to_numpy().astype(int)
blocks = {"cal": scal, "es": ses, "test": ste}
raw = {k: raw_dp(sf.iloc[v].match_id.tolist()) for k, v in blocks.items()}
yb = {k: y[v] for k, v in blocks.items()}
print("\nraw series DP (no series calibration):")
for k in blocks:
    print(f"  {k:5s} slope {calibration_slope(yb[k], raw[k]):.3f}  ll {log_loss(yb[k], np.clip(raw[k],.01,.99)):.4f}")

cal = fit_calibrator(raw["cal"], yb["cal"])
print(f"\nseries calibrator fit on cal ({len(scal)}): kind={cal.kind} T={cal.temperature:.3f}")
pt = cal.predict(raw["test"])
print(f"  test after calibration: slope {calibration_slope(yb['test'], pt):.3f}  ll {log_loss(yb['test'], pt):.4f}")

pd.DataFrame({"match_id": sf.iloc[ste].match_id.to_numpy(), "raw": raw["test"], "y": yb["test"]}).to_parquet(
    f"{D}/series_raw_test.parquet", index=False)
