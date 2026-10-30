"""HLTV player-rating form: each lineup's mean map rating over the 90 days before a day, shrunk toward 1.0.

Ratings are counted only for maps dated before the day in question, so a series sees the same ratings whether the
bundle holds the whole history or only the rows before its week's cutoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load

WINDOW_DAYS = 90
PRIOR_MAPS = 10
PRIOR_RATING = 1.0
EPOCH = pd.Timestamp("1970-01-01")


def day_index(s: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(s)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(None)
    return (dates - EPOCH).dt.days.to_numpy(np.int64)


class RatingTimelines:
    def __init__(self, data_dir: str | Path, max_cut: pd.Timestamp):
        df = load(data_dir, "hltv/cs2_player_map_stats", max_cut, ["match_date", "player_id", "side", "rating"])
        df = df[(df["side"] == "all") & df["rating"].notna() & df["player_id"].notna()]
        pid = df["player_id"].to_numpy(np.int64)
        days = day_index(df["match_date"])
        rating = df["rating"].to_numpy(float)
        order = np.lexsort((days, pid))
        pid, days, rating = pid[order], days[order], rating[order]
        uniq, first = np.unique(pid, return_index=True)
        last = np.append(first[1:], len(pid))
        self._days: dict[int, np.ndarray] = {}
        self._cum: dict[int, np.ndarray] = {}
        for u, lo, hi in zip(uniq, first, last):
            self._days[int(u)] = days[lo:hi]
            self._cum[int(u)] = np.concatenate([[0.0], np.cumsum(rating[lo:hi])])

    def block(self, player_ids: np.ndarray, asof_day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.full(len(player_ids), PRIOR_RATING)
        n_maps = np.zeros(len(player_ids))
        if len(player_ids) == 0:
            return mean, n_maps
        positions = pd.Series(np.arange(len(player_ids))).groupby(player_ids).indices
        for pid, idx in positions.items():
            days = self._days.get(int(pid))
            if days is None:
                continue
            cum = self._cum[int(pid)]
            t = asof_day[idx]
            hi = np.searchsorted(days, t, side="left")
            lo = np.searchsorted(days, t - WINDOW_DAYS, side="left")
            n = hi - lo
            mean[idx] = (cum[hi] - cum[lo] + PRIOR_MAPS * PRIOR_RATING) / (n + PRIOR_MAPS)
            n_maps[idx] = n
        return mean, n_maps


def lineup_ratings(data_dir: str | Path, max_cut: pd.Timestamp, x: pd.DataFrame) -> pd.Series:
    """Rating difference (team a minus team b) per row of x; rows need lineup_a, lineup_b and match_date."""
    tl = RatingTimelines(data_dir, max_cut)
    days = day_index(x["match_date"])
    frames = []
    for side, offset in (("a", 0), ("b", 1)):
        lineups = x[["row", f"lineup_{side}"]].rename(columns={f"lineup_{side}": "lineup"})
        lineups = lineups.assign(key=lineups["row"] * 2 + offset, asof_day=days[x["row"].to_numpy()])
        frames.append(lineups.explode("lineup").dropna(subset=["lineup"]))
    players = pd.concat(frames, ignore_index=True)
    mean, _ = tl.block(players["lineup"].to_numpy(np.int64), players["asof_day"].to_numpy(np.int64))
    team = pd.Series(mean, index=players["key"].to_numpy()).groupby(level=0).mean()
    keys = np.arange(2 * len(x))
    team = team.reindex(keys).fillna(PRIOR_RATING).to_numpy()
    return pd.Series(team[0::2] - team[1::2], index=x["row"].to_numpy())
