"""Residual CatBoost (model F): trees learn corrections on top of model C's log-odds.

C's log-odds enter as CatBoost's baseline, so the trees only model what C misses, fed with engineered
linear edges and interactions. Swapped rows get the negated baseline (C has no intercept and venue flips
sign on swap), and predictions average the original and swapped views so F is symmetric in team order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT
from gnomepy_research.sessions.cfb_win_probability.models.catboost_wp import MATCHUPS, PARAMS, TEAM_FEATURES, swap
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import N_FIT_DRAWS, fit, game_draws

LINEAR = ["pass_edge", "rush_edge", "net_edge", "fp_diff", "early_diff", "finish_diff", "late_diff"]
INTERACTIONS = ["unc", "edge_x_unc", "edge_x_week", "abs_edge", "pace_sum", "edge_x_pace", "style_diff",
                "run_heavy_vs_run_d", "finish_x_fp"]
SYMMETRIC = {"unc", "abs_edge", "pace_sum"}
SD_COLUMNS = [f"{s}_{r}_sd" for s in ("h", "a") for r in ("o_pass", "o_rush", "d_pass", "d_rush")]
GRID_DEPTHS = [2, 3, 4]
GRID_PARAMS = dict(learning_rate=0.03, l2_leaf_reg=10)
EPS = 1e-4


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pass_edge"] = df["pass_matchup_h"] - df["pass_matchup_a"]
    out["rush_edge"] = df["rush_matchup_h"] - df["rush_matchup_a"]
    out["net_edge"] = out["pass_edge"] + out["rush_edge"]
    out["fp_diff"] = df["h_field_position"] - df["a_field_position"]
    out["early_diff"] = df["h_early_success"] - df["a_early_success"]
    out["finish_diff"] = df["h_finishing_luck"] - df["a_finishing_luck"]
    out["late_diff"] = df["h_late_down_luck"] - df["a_late_down_luck"]
    out["unc"] = np.sqrt((df[SD_COLUMNS] ** 2).mean(axis=1))
    out["edge_x_unc"] = out["net_edge"] * out["unc"]
    out["edge_x_week"] = out["net_edge"] * df["week"]
    out["abs_edge"] = out["net_edge"].abs()
    out["pace_sum"] = df["h_pace"] + df["a_pace"]
    out["edge_x_pace"] = out["net_edge"] * out["pace_sum"]
    out["style_diff"] = df["h_rush_share"] - df["a_rush_share"]
    out["run_heavy_vs_run_d"] = df["h_rush_share"] * df["rush_matchup_h"] - df["a_rush_share"] * df["rush_matchup_a"]
    out["finish_x_fp"] = (df["h_finishing_luck"] * df["h_field_position"]
                          - df["a_finishing_luck"] * df["a_field_position"])
    return out


RAW = [*(f"{s}_{f}" for s in ("h", "a") for f in TEAM_FEATURES), *MATCHUPS, "venue", "postseason", "week"]
FEATURE_SETS = {"linear": [*LINEAR, "venue"], "interactions": [*LINEAR, "venue", *INTERACTIONS]}
FEATURE_SETS["full"] = [*FEATURE_SETS["interactions"], *(c for c in RAW if c not in FEATURE_SETS["interactions"])]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def attach_baseline(df: pd.DataFrame, baseline: pd.Series) -> np.ndarray:
    b = df["game_id"].map(baseline)
    if b.isna().any():
        raise ValueError(f"{int(b.isna().sum())} rows have no C baseline, e.g. game {df.loc[b.isna(), 'game_id'].iloc[0]}")
    return b.to_numpy(float)


@dataclass
class ResidualWP:
    model: CatBoostClassifier
    features: list[str]

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(engineer(df)[self.features], prediction_type="RawFormulaVal")

    def predict(self, df: pd.DataFrame, baseline: np.ndarray) -> np.ndarray:
        baseline = np.asarray(baseline, float)
        return 0.5 * (sigmoid(baseline + self.raw(df)) + 1 - sigmoid(-baseline + self.raw(swap(df))))

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.get_feature_importance(), index=self.features).sort_values(ascending=False)


def _pool(df: pd.DataFrame, baseline: np.ndarray, features: list[str]) -> Pool:
    aug = pd.concat([engineer(df), engineer(swap(df))], ignore_index=True)
    return Pool(aug[features], aug["home_win"], baseline=np.concatenate([baseline, -baseline]))


def fit_residual(train: pd.DataFrame, baseline: np.ndarray, features: list[str], params: dict,
                 seed: int = 0) -> ResidualWP:
    baseline = np.asarray(baseline, float)
    if len(baseline) != len(train) or np.isnan(baseline).any():
        raise ValueError("baseline must be one finite value per training row")
    full = {**PARAMS, **params}
    train = train.reset_index(drop=True)
    if "season" in train and train["season"].nunique() > 1:
        last = train["season"].max()
        is_val = (train["season"] == last).to_numpy()
        early = CatBoostClassifier(**full, random_seed=seed)
        early.fit(_pool(train[~is_val], baseline[~is_val], features),
                  eval_set=_pool(train[is_val], baseline[is_val], features), early_stopping_rounds=200)
        iterations = max(early.get_best_iteration() + 1, 1)
    else:
        iterations = 300
    final = CatBoostClassifier(**{**full, "iterations": iterations}, random_seed=seed)
    final.fit(_pool(train, baseline, features))
    return ResidualWP(final, features)


def _c_logodds(games: pd.DataFrame, fit_seasons: list[int], predict_seasons: list[int],
               n_draws: int | None) -> pd.Series:
    model = fit(game_draws(games, fit_seasons, n_draws=N_FIT_DRAWS, units=UNITS_SPLIT, form=True), seed=0)
    gd = game_draws(games, predict_seasons, n_draws=n_draws, units=UNITS_SPLIT, form=True)
    return pd.Series(logit(model.predict(gd.x)["p_home"].to_numpy()), index=gd.games["game_id"].to_numpy())


def c_baseline(games: pd.DataFrame, train_seasons: list[int], test_seasons: list[int]) -> pd.Series:
    """Model C's log-odds by game_id. Each training season is cross-fitted (C fitted on the other training
    seasons) so its residuals are out-of-sample like the test rows'; test seasons use C fitted on all of them."""
    if len(train_seasons) < 2:
        raise ValueError("cross-fitted baselines need at least two training seasons")
    parts = [_c_logodds(games, [o for o in train_seasons if o != s], [s], N_FIT_DRAWS) for s in train_seasons]
    parts.append(_c_logodds(games, train_seasons, test_seasons, None))
    out = pd.concat(parts)
    if out.index.duplicated().any():
        raise ValueError("a game_id has more than one C baseline")
    return out
