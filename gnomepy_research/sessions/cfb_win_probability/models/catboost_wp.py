"""Gradient-boosted pregame win probability on the model-C inputs (split ratings, rush share, team form).

Each game is one row of home and away team features: posterior mean and sd of the four split ratings at
the game's weekly cutoff, rush share, and form. Training on the original rows plus their home/away swapped
copies, and averaging predict(df) with 1 - predict(swap(df)), makes the model symmetric in team order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT
from gnomepy_research.models.cfb.features.team_form import FORM_FEATURES
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import DRAWS_DIR
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import load_season_draws, split_columns

RATINGS = ["o_pass", "o_rush", "d_pass", "d_rush"]
TEAM_FEATURES = [*RATINGS, *(f"{r}_sd" for r in RATINGS), "rush_share", *FORM_FEATURES]
MATCHUPS = ["pass_matchup_h", "rush_matchup_h", "pass_matchup_a", "rush_matchup_a"]
PARAMS = dict(loss_function="Logloss", depth=4, learning_rate=0.03, l2_leaf_reg=10, iterations=3000, verbose=False,
              allow_writing_files=False)


def game_features(games: pd.DataFrame, seasons: list[int], directory: Path = DRAWS_DIR) -> pd.DataFrame:
    g = games[games["season"].isin(seasons) & (games["home_classification"] == "fbs")
              & (games["away_classification"] == "fbs")]
    parts = []
    for season in seasons:
        draws = load_season_draws(season, directory, UNITS_SPLIT)
        for (st, wk), gg in g[g["season"] == season].groupby(["season_type", "week"]):
            key = f"{st}_{wk}"
            if f"{key}__off" not in draws:
                continue
            if f"{key}__form" not in draws:
                raise ValueError(f"{key}: draws have no form; snapshot with --form")
            off, de = draws[f"{key}__off"], draws[f"{key}__def"]
            p_cols, r_cols, hi, ai = split_columns(key, draws, gg)
            table = {"o_pass": off[:, p_cols], "o_rush": off[:, r_cols], "d_pass": de[:, p_cols], "d_rush": de[:, r_cols]}
            team = {f: v.mean(0) for f, v in table.items()}
            team.update({f"{f}_sd": v.std(0) for f, v in table.items()})
            team["rush_share"] = draws[f"{key}__rush_share"][p_cols]
            form = draws[f"{key}__form"][p_cols]
            team.update({f: form[:, i] for i, f in enumerate(FORM_FEATURES)})
            out = gg[["game_id", "season", "week", "season_type"]].reset_index(drop=True)
            for f in TEAM_FEATURES:
                out[f"h_{f}"] = team[f][hi]
                out[f"a_{f}"] = team[f][ai]
            parts.append(pd.concat([out, _derived(gg)], axis=1))
    df = pd.concat(parts, ignore_index=True)
    df["pass_matchup_h"] = df["h_o_pass"] - df["a_d_pass"]
    df["rush_matchup_h"] = df["h_o_rush"] - df["a_d_rush"]
    df["pass_matchup_a"] = df["a_o_pass"] - df["h_d_pass"]
    df["rush_matchup_a"] = df["a_o_rush"] - df["h_d_rush"]
    return df


def _derived(gg: pd.DataFrame) -> pd.DataFrame:
    gg = gg.reset_index(drop=True)
    return pd.DataFrame({
        "home_win": np.where(gg["completed"], (gg["home_points"] > gg["away_points"]).astype(float), np.nan),
        "venue": np.where(gg["neutral_site"], 0, 1),
        "postseason": (gg["season_type"] == "postseason").astype(int),
    })


def swap(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c.startswith("h_"):
            rename[c] = "a_" + c[2:]
        elif c.startswith("a_"):
            rename[c] = "h_" + c[2:]
        elif c.endswith("_h"):
            rename[c] = c[:-2] + "_a"
        elif c.endswith("_a"):
            rename[c] = c[:-2] + "_h"
    out = df.rename(columns=rename)[list(df.columns)].copy()
    out["home_win"] = 1 - out["home_win"]
    out["venue"] = -out["venue"]
    return out


def _model_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("h_", "a_")) or c in (*MATCHUPS, "venue", "postseason", "week")]


@dataclass
class CatBoostWP:
    model: CatBoostClassifier
    columns: list[str]

    def _p(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.columns])[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return 0.5 * (self._p(df) + (1 - self._p(swap(df))))

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.get_feature_importance(), index=self.columns).sort_values(ascending=False)


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, swap(df)], ignore_index=True)


def fit_catboost(train: pd.DataFrame, seed: int = 0) -> CatBoostWP:
    columns = _model_columns(train)
    last = train["season"].max() if "season" in train else None
    if last is not None and train["season"].nunique() > 1:
        fit_rows, val_rows = train[train["season"] < last], train[train["season"] == last]
        early = CatBoostClassifier(**PARAMS, random_seed=seed)
        early.fit(_augment(fit_rows)[columns], _augment(fit_rows)["home_win"],
                  eval_set=(_augment(val_rows)[columns], _augment(val_rows)["home_win"]),
                  early_stopping_rounds=200)
        iterations = max(early.get_best_iteration() + 1, 1)
    else:
        iterations = 300
    aug = _augment(train)
    final = CatBoostClassifier(**{**PARAMS, "iterations": iterations}, random_seed=seed)
    final.fit(aug[columns], aug["home_win"])
    return CatBoostWP(final, columns)
