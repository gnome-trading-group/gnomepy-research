"""Where does the model beat or lose to the market, and do those games share traits?

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.edge_analysis 2023_2024

Uses the backtest predictions (tuning seasons only; the holdout stays untouched) and the
sportsbook OPENING spread, which is public before our cutoff. For each subgroup:
- d = model log loss - market log loss (negative = model better), game-bootstrap 95% CI
- blend weight: logit(y) ~ a*logit(model) + b*logit(market); a > 0 means the model adds
  information the market lacks in that subgroup
Many subgroups are compared, so anything that looks like an edge is a hypothesis to confirm on
the holdout, not a finding.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss

BACKTEST = Path(__file__).resolve().parent.parent / "results" / "backtest"
POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "FBS Independents"}
SPREAD_SD = 15.5


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load(tag: str) -> pd.DataFrame:
    p = pd.read_parquet(BACKTEST / f"predictions_{tag}.parquet")
    ds = DatasetStore()
    lines, games, pregame = ds.load("cfb_lines"), ds.load("cfb_games"), ds.load("cfb_pregame_features")
    cons = lines.groupby("game_id").agg(open=("spread_open", "median"), close=("spread", "median"))
    p = p.merge(cons, left_on="game_id", right_index=True, how="left")
    p["p_open"] = norm.cdf(-p["open"] / SPREAD_SD)
    p = p.dropna(subset=["p_open"])
    g = games.set_index("game_id")
    p["home_conf"] = p["game_id"].map(g["home_conference"])
    p["away_conf"] = p["game_id"].map(g["away_conference"])
    pg = pregame.set_index("game_id")

    def col(c: str) -> pd.Series:
        return p["game_id"].map(pg[c])

    p["any_new_coach"] = col("home_pre_coach_new").astype(bool) | col("away_pre_coach_new").astype(bool)
    p["any_qb_not_returning"] = ~col("home_pre_qb_returning_starter").astype(bool) | ~col("away_pre_qb_returning_starter").astype(bool)
    # In-season QB change: the last game's starter isn't the preseason expected starter.
    home_change = (col("home_std_qb_current_starter").astype(str) != col("home_pre_qb_expected_starter").astype(str)) & col("home_std_qb_current_starter").notna()
    away_change = (col("away_std_qb_current_starter").astype(str) != col("away_pre_qb_expected_starter").astype(str)) & col("away_std_qb_current_starter").notna()
    p["any_qb_changed"] = home_change | away_change
    p["min_talent"] = np.minimum(col("home_pre_talent"), col("away_pre_talent"))
    p["fcs_free_games_seen"] = np.minimum(col("home_std_games").fillna(0), col("away_std_games").fillna(0))
    return p


def enrich(p: pd.DataFrame) -> pd.DataFrame:
    y = p["home_win"].to_numpy()
    p = p.assign(
        d=log_loss(y, p["p_bayes"].to_numpy()) - log_loss(y, p["p_open"].to_numpy()),
        disagree=p["p_bayes"] - p["p_open"],
        line_move=(p["close"] - p["open"]).abs(),
        matchup=np.select([p["home_conf"].isin(POWER) & p["away_conf"].isin(POWER),
                           ~p["home_conf"].isin(POWER) & ~p["away_conf"].isin(POWER)],
                          ["power v power", "G5 v G5"], "power v G5"),
        venue_type=np.where(p["venue"] == 1, "home", "neutral"),
        model_more_confident=np.abs(p["p_bayes"] - 0.5) > np.abs(p["p_open"] - 0.5),
        interval_width=p["p_bayes_hi"] - p["p_bayes_lo"],
    )
    fav = np.maximum(p["p_open"], 1 - p["p_open"])
    p["market_fav"] = pd.cut(fav, [0.5, 0.65, 0.8, 0.9, 1.0], include_lowest=True).astype(str)
    p["abs_disagree"] = pd.cut(p["disagree"].abs(), [0, 0.05, 0.10, 0.15, 1.0], include_lowest=True).astype(str)
    p["line_move_bin"] = pd.cut(p["line_move"], [-0.01, 0.5, 2.0, 4.0, 100], labels=["<=0.5", "0.5-2", "2-4", ">4"]).astype(str)
    p["width_tercile"] = pd.qcut(p["interval_width"], 3, labels=["narrow", "mid", "wide"]).astype(str)
    return p


def blend_weight(sub: pd.DataFrame) -> float:
    X = np.column_stack([logit(sub["p_bayes"].to_numpy()), logit(sub["p_open"].to_numpy())])
    if len(sub) < 40 or sub["home_win"].nunique() < 2:
        return np.nan
    return float(LogisticRegression(C=1e6, fit_intercept=False).fit(X, sub["home_win"]).coef_[0, 0])


def group_table(p: pd.DataFrame, by: str, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, sub in p.groupby(by):
        d = sub["d"].to_numpy()
        boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        rows.append({by: key, "n": len(sub), "model_ll": log_loss(sub["home_win"].to_numpy(), sub["p_bayes"].to_numpy()).mean(),
                     "market_ll": log_loss(sub["home_win"].to_numpy(), sub["p_open"].to_numpy()).mean(),
                     "d": d.mean(), "lo95": np.quantile(boots, 0.025), "hi95": np.quantile(boots, 0.975),
                     "model_better_share": (d < 0).mean(), "blend_w_model": blend_weight(sub)})
    return pd.DataFrame(rows).set_index(by)


def who_was_right(p: pd.DataFrame) -> pd.DataFrame:
    """When they disagree by >10 points, how often does the side the model leans toward win?"""
    big = p[p["disagree"].abs() > 0.10]
    model_side_home = big["p_bayes"] > big["p_open"]
    won = np.where(model_side_home, big["home_win"], 1 - big["home_win"])
    implied = np.where(model_side_home, big["p_open"], 1 - big["p_open"])
    modeled = np.where(model_side_home, big["p_bayes"], 1 - big["p_bayes"])
    return pd.DataFrame({"n": [len(big)], "model_side_win_rate": [won.mean()],
                         "market_implied": [implied.mean()], "model_implied": [modeled.mean()]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="2023_2024")
    args = ap.parse_args()
    p = enrich(load(args.tag))
    pd.set_option("display.width", 200)
    print(f"games: {len(p)}; overall model - market (open) log loss {p['d'].mean():+.4f}; "
          f"overall blend weight on model {blend_weight(p):.3f}\n")
    for by in ("phase", "market_fav", "matchup", "venue_type", "any_new_coach", "any_qb_not_returning",
               "any_qb_changed", "line_move_bin", "abs_disagree", "model_more_confident", "width_tercile", "season"):
        print(group_table(p, by).round(4).to_string(), "\n")
    print("big disagreements (>10 pts): who was right?\n", who_was_right(p).round(3).to_string(index=False), "\n")
    cols = ["season", "week", "away_team", "home_team", "p_bayes", "p_open", "home_win", "d", "line_move"]
    g = DatasetStore().load("cfb_games").set_index("game_id")
    p["away_team"] = p["game_id"].map(g["away_team"])
    p["home_team"] = p["game_id"].map(g["home_team"])
    print("model's best calls vs market:\n", p.nsmallest(8, "d")[cols].round(3).to_string(index=False), "\n")
    print("model's worst calls vs market:\n", p.nlargest(8, "d")[cols].round(3).to_string(index=False))
    out = BACKTEST / f"edge_analysis_{args.tag}.parquet"
    p.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
