"""In-game win-probability model (v1): monotone gradient boosting on game state + market prior.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.live.wp_model train
    poetry run python -m gnomepy_research.sessions.cfb_win_probability.live.wp_model evaluate 2023 2024

Trained on 2018-2022, tuned on 2023-2024; 2025 is the holdout and needs --holdout. Baselines:
the pregame prior held constant, and a logistic regression on the same features.

The last training season is held out inside training for early stopping and a Platt
recalibration of the trees' log-odds: every tree configuration tried was overconfident
(calibration slope 0.83-0.93 on 2022). Overtime uses the pregame prior until the drive simulator,
which implements NCAA overtime rules, replaces it: the trees scored worse than the prior there.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import paired_bootstrap, reliability, summarize
from gnomepy_research.sessions.cfb_win_probability.live.wp_data import FEATURES, MONOTONE, build_states, pregame_prior

TRAIN_SEASONS = list(range(2018, 2023))
HOLDOUT = 2025
OUT = Path(__file__).resolve().parent.parent / "results" / "ingame"
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 6, "eta": 0.05,
          "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 200, "tree_method": "hist"}


@dataclass
class WinProbModel:
    booster: xgb.Booster
    best_iteration: int
    calib_a: float
    calib_b: float

    def predict(self, states: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(xgb.DMatrix(states[FEATURES]), iteration_range=(0, self.best_iteration + 1))
        raw = np.clip(raw, 1e-6, 1 - 1e-6)
        p = 1 / (1 + np.exp(-(self.calib_a + self.calib_b * np.log(raw / (1 - raw)))))
        return np.where(states["is_ot"].to_numpy() == 1, states["prior"].to_numpy(), p)


def load_states(seasons: list[int]) -> pd.DataFrame:
    ds = DatasetStore()
    plays, games, lines = ds.load("cfb_plays"), ds.load("cfb_games"), ds.load("cfb_lines")
    s = build_states(plays[plays["season"].isin(seasons)], games, pregame_prior(lines))
    return s[(s["home_classification"] == "fbs") & (s["away_classification"] == "fbs")].reset_index(drop=True)


def fit_xgb(train: pd.DataFrame) -> WinProbModel:
    """Early-stop and recalibrate on the last training season, train on the rest."""
    val_season = train["season"].max()
    fit_rows, val = train[train["season"] < val_season], train[train["season"] == val_season]
    params = {**PARAMS, "monotone_constraints": "(" + ",".join(str(MONOTONE.get(f, 0)) for f in FEATURES) + ")"}
    dval = xgb.DMatrix(val[FEATURES], label=val["home_win"])
    booster = xgb.train(params, xgb.DMatrix(fit_rows[FEATURES], label=fit_rows["home_win"]), num_boost_round=3000,
                        evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=False)
    raw = np.clip(booster.predict(dval, iteration_range=(0, booster.best_iteration + 1)), 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e6).fit(np.log(raw / (1 - raw))[:, None], val["home_win"])
    return WinProbModel(booster, booster.best_iteration, float(lr.intercept_[0]), float(lr.coef_[0, 0]))


def fit_logistic(train: pd.DataFrame):
    x = train[FEATURES].fillna(0)
    scaler = StandardScaler().fit(x)
    return scaler, LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(x), train["home_win"])


def quarter(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["is_ot"] == 1, "OT", "Q" + df["period"].clip(upper=4).astype(str)), index=df.index)


def evaluate(train_seasons: list[int], test_seasons: list[int]) -> str:
    train, test = load_states(train_seasons), load_states(test_seasons)
    model = fit_xgb(train)
    scaler, logit = fit_logistic(train)
    test = test.assign(
        p_xgb=model.predict(test),
        p_logistic=logit.predict_proba(scaler.transform(test[FEATURES].fillna(0)))[:, 1],
        p_prior=test["prior"],
        q=quarter(test),
    )
    y = test["home_win"].to_numpy()
    lines = [f"train {train_seasons} ({len(train):,} plays), test {test_seasons} ({len(test):,} plays, "
             f"{test['game_id'].nunique()} games)"]
    lines.append("overall:\n" + pd.DataFrame({m: summarize(y, test[m].to_numpy()) for m in ("p_xgb", "p_logistic", "p_prior")}).T.round(4).to_string())
    by_q = {q: {m: summarize(d["home_win"].to_numpy(), d[m].to_numpy())["log_loss"] for m in ("p_xgb", "p_logistic", "p_prior")} | {"n": len(d)}
            for q, d in test.groupby("q")}
    lines.append("log loss by quarter:\n" + pd.DataFrame(by_q).round(4).to_string())
    # Plays within a game are correlated: bootstrap whole games for the comparison.
    per_game = test.groupby("game_id").apply(
        lambda d: pd.Series({m: summarize(d["home_win"].to_numpy(), d[m].to_numpy())["log_loss"] for m in ("p_xgb", "p_logistic")}),
        include_groups=False)
    d = (per_game["p_xgb"] - per_game["p_logistic"]).to_numpy()
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    lines.append(f"xgb - logistic per-game log loss: {d.mean():+.4f} (95% {np.quantile(boots, .025):+.4f} .. {np.quantile(boots, .975):+.4f})")
    lines.append("xgb reliability:\n" + reliability(y, test["p_xgb"].to_numpy()).round(3).to_string())
    late = test[(test["sec_game"] <= 300) & (test["is_ot"] == 0)]
    lines.append("last 5 minutes, xgb reliability:\n" + reliability(late["home_win"].to_numpy(), late["p_xgb"].to_numpy(), 5).round(3).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    test.to_parquet(OUT / f"eval_{'_'.join(map(str, test_seasons))}.parquet", index=False)
    return "\n\n".join(lines)


def train_final(seasons: list[int]) -> Path:
    model = fit_xgb(load_states(seasons))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"wp_xgb_{min(seasons)}_{max(seasons)}.json"
    model.booster.save_model(path)
    (OUT / f"wp_xgb_{min(seasons)}_{max(seasons)}.meta.json").write_text(json.dumps({
        "features": FEATURES, "monotone": MONOTONE, "params": PARAMS, "seasons": seasons,
        "best_iteration": model.best_iteration, "calib_a": model.calib_a, "calib_b": model.calib_b}, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evaluate", "train"])
    ap.add_argument("seasons", type=int, nargs="*")
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    if args.cmd == "evaluate":
        test = args.seasons or [2023, 2024]
        if HOLDOUT in test and not args.holdout:
            raise SystemExit(f"{HOLDOUT} is the holdout; pass --holdout to score it")
        train = [s for s in range(2018, min(test))]
        text = evaluate(train, test)
        (OUT / f"report_{'_'.join(map(str, test))}.txt").write_text(text)
        print(text)
    else:
        print(train_final(args.seasons or TRAIN_SEASONS))


if __name__ == "__main__":
    main()
