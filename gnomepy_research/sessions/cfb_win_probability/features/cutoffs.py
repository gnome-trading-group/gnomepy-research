"""Prediction cutoffs: one per (season, season_type, week), at that week's first kickoff.

Every game in a week is predicted from the same snapshot, so a Saturday game never uses the
Thursday game of its own week. That costs a little freshness and makes one model fit per week enough.
"""
from __future__ import annotations

import pandas as pd

WEEK_KEYS = ["season", "season_type", "week"]


def week_cutoffs(games: pd.DataFrame) -> pd.DataFrame:
    c = games.groupby(WEEK_KEYS, as_index=False)["kickoff"].min().rename(columns={"kickoff": "cutoff"})
    return c.sort_values("cutoff").reset_index(drop=True)


def game_cutoffs(games: pd.DataFrame) -> pd.DataFrame:
    """game_id -> cutoff."""
    return games[["game_id", *WEEK_KEYS]].merge(week_cutoffs(games), on=WEEK_KEYS)[["game_id", *WEEK_KEYS, "cutoff"]]
