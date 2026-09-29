"""
Compute point-in-time team priors from cs2_match_history and cs2_team_rankings datasets.

For each map in cs2_match_history, computes prior features using only
data strictly before that match's date — no lookahead bias.

Output dataset: cs2_match_priors
  match_id, map_name, team_a_name, team_b_name,
  team_a_rating_diff,
  team_a_map_winrate_long, team_b_map_winrate_long,   (last 20 maps)
  team_a_map_winrate_short, team_b_map_winrate_short, (last 5 maps)
  h2h_win_rate, recent_form_diff, team_a_picked_map, is_lan
"""
from __future__ import annotations

import argparse
import datetime
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MAP_WIN_RATE_WINDOW_LONG = 20
_MAP_WIN_RATE_WINDOW_SHORT = 5
_RECENT_FORM_WINDOW = 10
_H2H_WINDOW = 10


def _team_map_winrate(history: pd.DataFrame, team_name: str, map_name: str, before_date: datetime.date, window: int) -> float:
    """Win rate for team_name on map_name using the last `window` maps played strictly before before_date."""
    mask = (
        ((history["team_a_name"] == team_name) | (history["team_b_name"] == team_name)) &
        (history["map_name"] == map_name) &
        (history["match_date"] < before_date)
    )
    sub = history[mask].tail(window)
    if len(sub) == 0:
        return float("nan")
    wins = sub.apply(
        lambda r: r["team_a_won"] if r["team_a_name"] == team_name else (1 - r["team_a_won"]),
        axis=1,
    ).sum()
    return float(wins / len(sub))


def _team_recent_form(history: pd.DataFrame, team_name: str, before_date: datetime.date) -> float:
    """Win rate across all maps in the last N maps played before before_date."""
    mask = (
        ((history["team_a_name"] == team_name) | (history["team_b_name"] == team_name)) &
        (history["match_date"] < before_date)
    )
    sub = history[mask].tail(_RECENT_FORM_WINDOW)
    if len(sub) == 0:
        return float("nan")
    wins = sub.apply(
        lambda r: r["team_a_won"] if r["team_a_name"] == team_name else (1 - r["team_a_won"]),
        axis=1,
    ).sum()
    return float(wins / len(sub))


def _h2h_winrate(history: pd.DataFrame, team_a: str, team_b: str, before_date: datetime.date) -> float:
    """team_a's win rate vs team_b in maps played strictly before before_date."""
    mask = (
        (
            ((history["team_a_name"] == team_a) & (history["team_b_name"] == team_b)) |
            ((history["team_a_name"] == team_b) & (history["team_b_name"] == team_a))
        ) &
        (history["match_date"] < before_date)
    )
    sub = history[mask].tail(_H2H_WINDOW)
    if len(sub) == 0:
        return float("nan")
    wins = sub.apply(
        lambda r: r["team_a_won"] if r["team_a_name"] == team_a else (1 - r["team_a_won"]),
        axis=1,
    ).sum()
    return float(wins / len(sub))


def _ranking_points(rankings: pd.DataFrame, team_name: str, before_date: datetime.date) -> float:
    """Most recent ranking points for team_name from rankings published before before_date."""
    mask = (
        (rankings["team_name"].str.lower() == team_name.lower()) &
        (rankings["date"] < before_date)
    )
    sub = rankings[mask]
    if len(sub) == 0:
        return float("nan")
    return float(sub.sort_values("date").iloc[-1]["points"])


def compute_priors_for_row(
    row: pd.Series,
    history_sorted: pd.DataFrame,
    rankings: pd.DataFrame,
) -> dict:
    """Compute all prior features for a single map row."""
    team_a = row["team_a_name"]
    team_b = row["team_b_name"]
    map_name = row["map_name"]
    if pd.isna(row["match_date"]):
        return None
    match_date = pd.Timestamp(row["match_date"])

    pts_a = _ranking_points(rankings, team_a, match_date)
    pts_b = _ranking_points(rankings, team_b, match_date)
    is_lan = int("(LAN)" in str(row.get("format", "")))

    return {
        "match_id": row["match_id"],
        "map_name": map_name,
        "match_date": pd.Timestamp(match_date),
        "team_a_name": team_a,
        "team_b_name": team_b,
        "team_a_rating_diff": pts_a - pts_b,
        "team_a_map_winrate_long": _team_map_winrate(history_sorted, team_a, map_name, match_date, _MAP_WIN_RATE_WINDOW_LONG),
        "team_b_map_winrate_long": _team_map_winrate(history_sorted, team_b, map_name, match_date, _MAP_WIN_RATE_WINDOW_LONG),
        "team_a_map_winrate_short": _team_map_winrate(history_sorted, team_a, map_name, match_date, _MAP_WIN_RATE_WINDOW_SHORT),
        "team_b_map_winrate_short": _team_map_winrate(history_sorted, team_b, map_name, match_date, _MAP_WIN_RATE_WINDOW_SHORT),
        "h2h_win_rate": _h2h_winrate(history_sorted, team_a, team_b, match_date),
        "recent_form_diff": (
            _team_recent_form(history_sorted, team_a, match_date) -
            _team_recent_form(history_sorted, team_b, match_date)
        ),
        "team_a_picked_map": float(row.get("team_a_picked_map", 0)),
        "is_lan": is_lan,
    }


def build_priors(
    match_history: pd.DataFrame,
    team_rankings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute point-in-time priors for every row in match_history.
    Returns a DataFrame with one row per map, keyed by match_id + map_name.
    """
    history_sorted = match_history.sort_values("match_date").reset_index(drop=True)
    rankings = team_rankings.sort_values("date").reset_index(drop=True)

    logger.info("Computing priors for %d map rows...", len(history_sorted))
    rows = []
    for i, row in history_sorted.iterrows():
        result = compute_priors_for_row(row, history_sorted, rankings)
        if result is not None:
            rows.append(result)
        if (i + 1) % 500 == 0:
            logger.info("Processed %d / %d rows", i + 1, len(history_sorted))

    return pd.DataFrame(rows)


def compute_live_priors(
    team_a_name: str,
    team_b_name: str,
    map_name: str,
    team_a_picked_map: bool,
    is_lan: bool,
    match_history: pd.DataFrame,
    team_rankings: pd.DataFrame,
    as_of_date: datetime.date | None = None,
) -> dict:
    """
    Compute priors for a live/upcoming match using the most recent available data.
    as_of_date defaults to today. Used by the live strategy at startup.
    """
    if as_of_date is None:
        as_of_date = datetime.date.today()

    history_sorted = match_history.sort_values("match_date").reset_index(drop=True)
    rankings = team_rankings.sort_values("date").reset_index(drop=True)

    future_date = pd.Timestamp(as_of_date + datetime.timedelta(days=1))

    pts_a = _ranking_points(rankings, team_a_name, future_date)
    pts_b = _ranking_points(rankings, team_b_name, future_date)

    return {
        "team_a_avg_rating": pts_a,
        "team_b_avg_rating": pts_b,
        "team_a_map_winrate_long": _team_map_winrate(history_sorted, team_a_name, map_name, future_date, _MAP_WIN_RATE_WINDOW_LONG),
        "team_b_map_winrate_long": _team_map_winrate(history_sorted, team_b_name, map_name, future_date, _MAP_WIN_RATE_WINDOW_LONG),
        "team_a_map_winrate_short": _team_map_winrate(history_sorted, team_a_name, map_name, future_date, _MAP_WIN_RATE_WINDOW_SHORT),
        "team_b_map_winrate_short": _team_map_winrate(history_sorted, team_b_name, map_name, future_date, _MAP_WIN_RATE_WINDOW_SHORT),
        "h2h_win_rate": _h2h_winrate(history_sorted, team_a_name, team_b_name, future_date),
        "team_a_recent_form": _team_recent_form(history_sorted, team_a_name, future_date),
        "team_b_recent_form": _team_recent_form(history_sorted, team_b_name, future_date),
        "team_a_picked_map": team_a_picked_map,
        "is_lan": int(is_lan),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-history", required=True)
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    history = pd.read_parquet(args.match_history)
    rankings_df = pd.read_parquet(args.rankings)
    priors_df = build_priors(history, rankings_df)
    priors_df.to_parquet(args.out, index=False)
    logger.info("Saved %d rows to %s", len(priors_df), args.out)
