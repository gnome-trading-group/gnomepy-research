"""In-game win-probability training rows: one per scrimmage play, as the game stood at the snap.

CFBD play rows carry the score AFTER the play (a touchdown row already shows the 7), so the
pre-snap score is the previous play's post-play score. Down, distance, yards-to-goal and
timeouts on a row are pre-snap. The CFBD live feed has the same semantics: the latest play's
score is the current score, and the top-level situation is the next snap.

Everything is from the home team's perspective; the label is the home team winning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Converts a closing spread to a pregame probability when no moneyline exists (2018-2020).
SPREAD_SD = 15.5
FEATURES = [
    "prior_logit", "score_diff", "sec_game", "sec_half", "period", "is_ot", "home_has_ball", "yards_to_goal",
    "down", "distance", "home_timeouts", "away_timeouts", "home_receives_2h", "prior_x_remaining",
    "diff_per_sqrt_time",
]
# +1: win probability may only rise with the feature; 0: unconstrained.
MONOTONE = {"prior_logit": 1, "score_diff": 1, "prior_x_remaining": 1, "diff_per_sqrt_time": 1}


def pregame_prior(lines: pd.DataFrame) -> pd.Series:
    """Consensus closing no-vig P(home) per game: moneyline where both sides exist, else spread."""
    c = lines.groupby("game_id").agg(spread=("spread", "median"), p_ml=("home_prob_novig", "median"))
    return c["p_ml"].fillna(pd.Series(norm.cdf(-c["spread"] / SPREAD_SD), index=c.index)).rename("prior")


def game_seconds_left(period: pd.Series, clock_s: pd.Series) -> pd.Series:
    return np.where(period <= 4, (4 - period) * 900 + clock_s, 0.0)


def half_seconds_left(period: pd.Series, clock_s: pd.Series) -> pd.Series:
    return np.where(period.isin([1, 3]), clock_s + 900, np.where(period <= 4, clock_s, 0.0))


def build_states(plays: pd.DataFrame, games: pd.DataFrame, prior: pd.Series) -> pd.DataFrame:
    g = games[games["completed"] & games["home_points"].notna()].set_index("game_id")
    p = plays[plays["game_id"].isin(g.index)].sort_values(["game_id", "play_seq"]).copy()
    p["home_team"] = p["game_id"].map(g["home_team"])
    p["home_has_ball"] = (p["offense"] == p["home_team"]).astype(int)
    post_home = np.where(p["home_has_ball"] == 1, p["offense_score"], p["defense_score"])
    post_away = np.where(p["home_has_ball"] == 1, p["defense_score"], p["offense_score"])
    p["post_home"], p["post_away"] = post_home, post_away
    # Pre-snap score = the previous play's post-play score (0-0 before the first play).
    p["home_score"] = p.groupby("game_id")["post_home"].shift(1).fillna(0)
    p["away_score"] = p.groupby("game_id")["post_away"].shift(1).fillna(0)

    # Kickoff rows list the kicking team as the offense, so the opening receiver is the offense
    # of the first scrimmage play. Whoever received the opening kickoff kicks off the second half.
    first_offense = p[p["down"].between(1, 4)].groupby("game_id")["home_has_ball"].first()
    p["home_receives_2h"] = 1 - p["game_id"].map(first_offense)

    p = p[p["down"].between(1, 4)].copy()
    p["period"] = p["period"].astype(int)
    p["is_ot"] = (p["period"] >= 5).astype(int)
    p["sec_game"] = game_seconds_left(p["period"], p["clock_s"].fillna(0))
    p["sec_half"] = half_seconds_left(p["period"], p["clock_s"].fillna(0))
    p["score_diff"] = p["home_score"] - p["away_score"]
    p["home_timeouts"] = np.where(p["home_has_ball"] == 1, p["offense_timeouts"], p["defense_timeouts"])
    p["away_timeouts"] = np.where(p["home_has_ball"] == 1, p["defense_timeouts"], p["offense_timeouts"])

    pr = p["game_id"].map(prior).clip(0.005, 0.995)
    p["prior"] = pr
    p["prior_logit"] = np.log(pr / (1 - pr))
    remaining = p["sec_game"] / 3600.0
    p["prior_x_remaining"] = p["prior_logit"] * remaining
    # A lead matters more the less time is left; the +60 s floor keeps the last snap finite.
    p["diff_per_sqrt_time"] = p["score_diff"] / np.sqrt((p["sec_game"] + 60) / 60)

    p["home_win"] = (p["game_id"].map(g["home_points"]) > p["game_id"].map(g["away_points"])).astype(int)
    p["neutral_site"] = p["game_id"].map(g["neutral_site"])
    p["home_classification"] = p["game_id"].map(g["home_classification"])
    p["away_classification"] = p["game_id"].map(g["away_classification"])
    keep = ["game_id", "season", "play_id", "play_seq", "home_team", "home_score", "away_score", "home_win",
            "prior", "neutral_site", "home_classification", "away_classification", *FEATURES]
    return p.dropna(subset=["prior_logit"])[keep].reset_index(drop=True)
