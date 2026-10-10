"""
Feature importance for the pre-map model, three ways, averaged over column orders.

Single-feature importance is unreliable here: the features come in correlated
groups (A/B pairs plus a diff, HLTV and VRS rank, global and per-map Elo), so
permuting one member barely hurts while the model leans on its twin. Grouped
permutation - shuffling a whole block jointly - measures what the model would
lose without that information at all. TreeSHAP gives the per-feature view, and
total gain is shown only because it is what XGBoost reports by default.

Each measure is averaged over several fits with shuffled column order, since
colsample_bytree makes any one fit's importances order-dependent.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from gnomepy_research.pipelines.hltv_cs2.context_features import (
    EVENT_FEATURES, PAGE_H2H_FEATURES, RANK_FEATURES, SCHEDULE_FEATURES, VETO_FEATURES,
)
from gnomepy_research.pipelines.hltv_cs2.player_form import PLAYER_FORM_FEATURES
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR
from gnomepy_research.sessions.cs2_win_probability.pre_map_features import _MAP_OHE_NAMES, PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import _fit, _matrix, fit_calibrator
from gnomepy_research.sessions.cs2_win_probability.splits import split_val_for_calibration, temporal_split_by_series
from gnomepy_research.sessions.cs2_win_probability.symmetry import augment, build_swap_plan, symmetric_predict_proba

N_ORDERS = 3
N_PERMS = 5
BLOCKS = {
    "Elo": ["elo_diff", "elo_map_diff", "elo_map_resid_diff", "elo_side_asym", "elo_games_min"],
    "rank (HLTV + VRS)": RANK_FEATURES + ["rank_diff", "team_a_rating_diff", "ranking_age_days"],
    "player form": PLAYER_FORM_FEATURES,
    "team results history": ["team_a_map_winrate_long", "team_b_map_winrate_long", "team_a_map_winrate_short",
                             "team_b_map_winrate_short", "team_a_overall_winrate", "team_b_overall_winrate",
                             "recent_form_diff"],
    "head-to-head": ["h2h_win_rate", "own_h2h_maps", "own_h2h_rate", "own_h2h_never_met"] + PAGE_H2H_FEATURES,
    "schedule / rest": [f for f in SCHEDULE_FEATURES if not f.startswith("own_h2h")],
    "veto": VETO_FEATURES + ["team_a_picked_map"],
    "series state": ["team_a_series_score", "team_b_series_score", "map_position_in_series", "is_decider", "bo_type"],
    "event context": EVENT_FEATURES + ["is_lan", "event_tier"],
    "map identity": sorted(_MAP_OHE_NAMES),
}
assert sorted(sum(BLOCKS.values(), [])) == sorted(PRE_MAP_FEATURE_NAMES), "blocks must partition the feature set"

history = pd.read_parquet(DATA_DIR / "cs2_match_history.parquet")
priors = pd.read_parquet(DATA_DIR / "priors_full.parquet")
cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
mf = priors.merge(history[cols], on=["match_id", "map_name"])
mf = mf[mf.team_a_score != mf.team_b_score].sort_values(["match_date", "match_id"]).reset_index(drop=True)
tr, val, te = temporal_split_by_series(mf)
es, cal = split_val_for_calibration(mf, val)
y = mf.team_a_won.to_numpy().astype(int)
print(f"train {len(tr)}  early-stop {len(es)}  calibrate {len(cal)}  test {len(te)} maps "
      f"(test {mf.iloc[te].match_date.min().date()} -> {mf.iloc[te].match_date.max().date()})", flush=True)

rng = np.random.default_rng(0)
shap_runs, gain_runs, perm_runs, base_lls = [], [], [], []
for k in range(N_ORDERS):
    names = list(PRE_MAP_FEATURE_NAMES)
    rng.shuffle(names)
    col_names = [n for n in names if n not in _MAP_OHE_NAMES] + [n for n in names if n in _MAP_OHE_NAMES]
    plan = build_swap_plan(names)
    X = _matrix(mf, names)
    Xtr, ytr = augment(X[tr], y[tr], plan)
    model = _fit(Xtr, ytr, X[es], y[es], 1000, 5)
    calib = fit_calibrator(symmetric_predict_proba(model, X[cal], plan), y[cal])
    predict = lambda M: calib.predict(symmetric_predict_proba(model, M, plan))

    booster = model.get_booster()
    contrib = booster.predict(xgb.DMatrix(X[te]), pred_contribs=True)[:, :-1]
    shap_runs.append(pd.Series(np.abs(contrib).mean(axis=0), index=col_names))
    gain = booster.get_score(importance_type="total_gain")
    gain_runs.append(pd.Series({col_names[int(f[1:])]: v for f, v in gain.items()}).reindex(col_names).fillna(0.0))

    base = log_loss(y[te], predict(X[te]))
    base_lls.append(base)
    perm = {}
    for block, feats in BLOCKS.items():
        idx = [col_names.index(f) for f in feats]
        deltas = []
        for _ in range(N_PERMS):
            Xp = X[te].copy()
            Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
            deltas.append(log_loss(y[te], predict(Xp)) - base)
        perm[block] = (np.mean(deltas), np.std(deltas))
    perm_runs.append(perm)
    print(f"order {k}: test log-loss {base:.4f}, best iteration {model.best_iteration}", flush=True)

shap = pd.concat(shap_runs, axis=1)
gain = pd.concat(gain_runs, axis=1)
gain = gain / gain.sum()
block_of = {f: b for b, fs in BLOCKS.items() for f in fs}

print(f"\nmean test log-loss across orders: {np.mean(base_lls):.4f}")
print("\n=== grouped permutation importance (rise in test log-loss when the whole block is shuffled) ===")
rows = []
for block in BLOCKS:
    means = [p[block][0] for p in perm_runs]
    rows.append({"block": block, "n_features": len(BLOCKS[block]), "delta_logloss": np.mean(means),
                 "across_orders_sd": np.std(means), "within_perm_sd": np.mean([p[block][1] for p in perm_runs])})
P = pd.DataFrame(rows).sort_values("delta_logloss", ascending=False)
print(P.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n=== top 25 features by mean |SHAP| (log-odds), with total-gain share ===")
T = pd.DataFrame({"block": [block_of[f] for f in shap.index], "mean_abs_shap": shap.mean(axis=1),
                  "shap_sd_across_orders": shap.std(axis=1), "gain_share": gain.mean(axis=1).reindex(shap.index)})
T = T.sort_values("mean_abs_shap", ascending=False)
T["shap_rank"] = range(1, len(T) + 1)
T["gain_rank"] = T.gain_share.rank(ascending=False).astype(int)
print(T.head(25).to_string(float_format=lambda v: f"{v:.4f}"))
print("\nfeatures with ~zero SHAP:", list(T[T.mean_abs_shap < 1e-3].index))
print("\nSHAP share by block:")
print((T.groupby("block").mean_abs_shap.sum() / T.mean_abs_shap.sum()).sort_values(ascending=False)
      .to_string(float_format=lambda v: f"{v:.1%}"))
T.to_parquet(DATA_DIR / "feature_importance.parquet")
P.to_parquet(DATA_DIR / "feature_importance_blocks.parquet", index=False)
