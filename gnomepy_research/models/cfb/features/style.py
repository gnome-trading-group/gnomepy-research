"""Offensive style flags. Option offenses (service academies, etc.) run far more than anyone else,
which the efficiency ratings don't capture as a matchup trait. Point-in-time: callers pass only
rows visible before their cutoff."""
from __future__ import annotations

import pandas as pd

OPTION_RUSH_SHARE = 0.65
MIN_SEASON_GAMES = 3


def _share(e: pd.DataFrame) -> pd.Series:
    g = e.groupby("team_id")[["rush_plays", "pass_plays"]].sum()
    return g["rush_plays"] / (g["rush_plays"] + g["pass_plays"])


def option_offense(efficiency: pd.DataFrame, season: int) -> pd.Series:
    """team_id -> 1 if the previous season's rush share is >= 0.65; teams without a previous season
    fall back to this season's rows once they have at least 3 games."""
    prev = _share(efficiency[efficiency["season"] == season - 1])
    cur = efficiency[efficiency["season"] == season]
    games = cur.groupby("team_id")["game_id"].nunique()
    cur_share = _share(cur)[lambda s: s.index.isin(games[games >= MIN_SEASON_GAMES].index)]
    share = pd.concat([prev, cur_share[~cur_share.index.isin(prev.index)]])
    return (share >= OPTION_RUSH_SHARE).astype(int).rename("option")
