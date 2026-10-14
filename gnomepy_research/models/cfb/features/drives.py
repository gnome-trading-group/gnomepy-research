"""Points per drive per team-game offense: a second, drive-level observation of team strength.

Kneel-downs at the end of a half or game say nothing about the offense, so they are dropped.
Garbage-time drives still carry information, just less, so they are down-weighted rather than dropped.
Overtime drives start at the 25 and are dropped as unrepresentative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.features.team_form import is_garbage

KNEEL_RESULTS = ("END OF HALF", "END OF GAME", "END OF 4TH QUARTER")
KNEEL_MAX_PLAYS = 3
GARBAGE_WEIGHT = 0.3
REGULATION_PERIODS = (1, 2, 3, 4)
DRIVE_COLUMNS = ["game_id", "season", "season_type", "week", "team_id", "opponent_id", "team_classification",
                 "opponent_classification", "venue", "ppd", "n_eff", "available_at"]


def team_game_drives(drives: pd.DataFrame, games: pd.DataFrame, garbage_weight: float = GARBAGE_WEIGHT) -> pd.DataFrame:
    d = drives.dropna(subset=["offense_id", "defense_id"])
    kneel = d["drive_result"].isin(KNEEL_RESULTS) & (d["n_plays"] <= KNEEL_MAX_PLAYS)
    d = d[~kneel & d["start_period"].isin(REGULATION_PERIODS)]
    garbage = is_garbage(d["start_period"].to_numpy(), d["start_offense_score"].to_numpy(),
                         d["start_defense_score"].to_numpy())
    d = d.assign(w=np.where(garbage, garbage_weight, 1.0))
    d = d.assign(wp=d["w"] * d["off_points"].astype(float))
    agg = d.groupby(["game_id", "offense_id", "defense_id"], as_index=False).agg(n_eff=("w", "sum"), wp=("wp", "sum"))
    agg = agg.rename(columns={"offense_id": "team_id", "defense_id": "opponent_id"})
    agg["ppd"] = agg["wp"] / agg["n_eff"]
    g = games.set_index("game_id")
    out = agg.join(g[["season", "season_type", "week", "kickoff", "neutral_site", "home_id",
                      "home_classification", "away_classification"]], on="game_id", how="inner")
    home = out["team_id"] == out["home_id"]
    out["venue"] = np.where(out["neutral_site"].astype(bool), 0, np.where(home, 1, -1))
    out["team_classification"] = np.where(home, out["home_classification"], out["away_classification"])
    out["opponent_classification"] = np.where(home, out["away_classification"], out["home_classification"])
    out["available_at"] = out["kickoff"] + pd.Timedelta(days=1)
    out[["team_id", "opponent_id"]] = out[["team_id", "opponent_id"]].astype("int64")
    return out[DRIVE_COLUMNS].sort_values(["game_id", "team_id"]).reset_index(drop=True)
