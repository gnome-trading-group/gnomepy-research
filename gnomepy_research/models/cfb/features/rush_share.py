"""Point-in-time run/pass mix per team, shrunk toward the league so a team's first game doesn't swing it."""
from __future__ import annotations

import pandas as pd

RUSH_SHARE_SHRINK_PLAYS = 150.0


def league_rush_share(efficiency: pd.DataFrame, season: int) -> float:
    e = efficiency[efficiency["season"] == season]
    total = float(e["rush_plays"].sum() + e["pass_plays"].sum())
    return float(e["rush_plays"].sum()) / total if total > 0 else 0.5


def rush_share(efficiency: pd.DataFrame, season: int, shrink_plays: float = RUSH_SHARE_SHRINK_PLAYS) -> pd.Series:
    e = efficiency[efficiency["season"] == season]
    lg = league_rush_share(efficiency, season)
    g = e.groupby("team_id")[["rush_plays", "pass_plays"]].sum()
    return ((g["rush_plays"] + shrink_plays * lg) / (g["rush_plays"] + g["pass_plays"] + shrink_plays)).rename("rush_share")
