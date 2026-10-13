"""Comparison models, all point-in-time and fitted only on training seasons.

1. home_field: training-set home win rate for true home games, 0.5 at neutral sites.
2. elo: MOV-adjusted Elo, ratings frozen at each weekly cutoff (a week's games are predicted
   before any of them update ratings), regressed a third toward the mean between seasons.
3. logistic: non-Bayesian logistic regression on the same point-in-time features the Bayesian
   model's priors and ratings come from (cfb_pregame_features), no intercept (symmetric).
4. market: no-vig closing moneyline where both sides exist, else the closing spread converted
   with a normal margin model. CLOSING prices are only public at kickoff, after our cutoff: this
   is a benchmark, not a fair competitor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss


def home_field(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    rate = train.loc[train["venue"] == 1, "home_win"].mean()
    return np.where(test["venue"] == 1, rate, 0.5)


@dataclass
class EloParams:
    k: float = 25.0
    hfa: float = 55.0
    carryover: float = 2 / 3
    fcs_start: float = 1200.0


def elo_predictions(games: pd.DataFrame, params: EloParams) -> pd.Series:
    """P(home) for every game, from ratings frozen at its week's start. All games (FCS too) update
    ratings; seasons before the first prediction season act as burn-in."""
    g = games.sort_values("kickoff")
    rating: dict[int, float] = {}
    out = {}
    season = None
    for (s, st, wk), week in g.groupby(["season", "season_type", "week"], sort=False):
        if s != season:
            for t in rating:
                rating[t] = 1500 + params.carryover * (rating[t] - 1500)
            season = s
        week_ratings = {}
        for r in week.itertuples():
            h = rating.setdefault(r.home_id, 1500.0 if r.home_classification == "fbs" else params.fcs_start)
            a = rating.setdefault(r.away_id, 1500.0 if r.away_classification == "fbs" else params.fcs_start)
            diff = h - a + (0.0 if r.neutral_site else params.hfa)
            out[r.game_id] = 1 / (1 + 10 ** (-diff / 400))
            week_ratings[r.game_id] = diff
        for r in week.itertuples():
            if not r.completed or pd.isna(r.home_points):
                continue
            mov = r.home_points - r.away_points
            result = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
            diff = week_ratings[r.game_id]
            mult = np.log(abs(mov) + 1) * 2.2 / (0.001 * diff * np.sign(mov) + 2.2) if mov != 0 else 1.0
            shift = params.k * mult * (result - out[r.game_id])
            rating[r.home_id] += shift
            rating[r.away_id] -= shift
    return pd.Series(out, name="p_elo")


def tune_elo(games: pd.DataFrame, train_ids: set) -> EloParams:
    best, best_ll = EloParams(), np.inf
    labels = games.set_index("game_id")
    for k in (15, 20, 25, 30, 40):
        for hfa in (40, 55, 70, 85):
            p = elo_predictions(games, EloParams(k=k, hfa=hfa))
            ids = [i for i in p.index if i in train_ids]
            y = (labels.loc[ids, "home_points"] > labels.loc[ids, "away_points"]).to_numpy(float)
            ll = log_loss(y, p.loc[ids].to_numpy()).mean()
            if ll < best_ll:
                best, best_ll = EloParams(k=k, hfa=hfa), ll
    return best


LOGISTIC_DIFFS = [
    "pre_prior_off_rating", "pre_prior_def_rating", "pre_recruiting_points", "pre_talent_offense",
    "pre_talent_defense", "pre_returning_pct_ppa", "pre_qb_prior_ppa", "pre_coach_srs_resid", "pre_portal_in_stars",
    "std_off_rating", "std_def_rating", "std_off_ppa", "std_def_ppa_allowed",
]


def logistic_design(pregame: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=pregame.index)
    for c in LOGISTIC_DIFFS:
        x[c] = (pregame[f"home_{c}"] - pregame[f"away_{c}"]).astype(float)
    # Season-to-date evidence should matter more as games accumulate.
    games = (pregame["home_std_games"].fillna(0) + pregame["away_std_games"].fillna(0)) / 2
    for c in ("std_off_rating", "std_def_rating"):
        x[f"{c}_x_games"] = x[c] * games
    x["venue"] = np.where(pregame["neutral_site"], 0.0, 1.0)
    return x.fillna(0.0)


def logistic(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    xtr, xte = logistic_design(train), logistic_design(test)
    # Scale without centring so a swapped neutral game still mirrors exactly.
    scaler = StandardScaler(with_mean=False).fit(xtr)
    m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000)
    m.fit(scaler.transform(xtr), train["home_win"].to_numpy(int))
    return m.predict_proba(scaler.transform(xte))[:, 1]


def market_close(lines: pd.DataFrame, train_games: pd.DataFrame) -> tuple[pd.Series, float]:
    """Consensus closing no-vig P(home) per game, plus the spread->probability sd fitted on training games."""
    cons = lines.groupby("game_id").agg(spread=("spread", "median"), p_ml=("home_prob_novig", "median"))
    t = train_games.merge(cons, left_on="game_id", right_index=True).dropna(subset=["spread"])
    margin = t["home_points"] - t["away_points"]
    sd = float((margin + t["spread"]).std())
    p = cons["p_ml"].copy()
    from_spread = pd.Series(norm.cdf(-cons["spread"] / sd), index=cons.index)
    p = p.fillna(from_spread)
    return p.rename("p_market"), sd
