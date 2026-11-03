"""
Point-in-time reads of the tables e12 fits on.

A row counts only once it was known before the cutoff: HLTV rows by match date (and match time for maps), rankings by
ranking date, FACEIT rows by their match's `finished_at` (some ESEA maps are finalised weeks after they are played),
and matchmaking players through their matches. Every source applies the same rules, so a fit at a cutoff is the same
wherever its tables come from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATE_COLUMN = {"match_history": "match_date", "match_lineups": "match_date", "player_map_stats": "match_date",
               "team_rankings": "date"}
BY_FACEIT_MATCH = {"faceit_match_history", "faceit_rosters", "faceit_exclusions"}
FINISHED_AT = {"faceit_matches", "faceit_history_matches"}


class Tables(Protocol):
    def load(self, name: str, before: pd.Timestamp | None = None, columns: list[str] | None = None) -> pd.DataFrame:
        """Rows of one table known before `before` (naive UTC); every row when None."""

    def history_players(self) -> pa.Table:
        """FACEIT matchmaking players (match_id, faction, faceit_id), dictionary-encoded and uncut: it has tens of millions
        of rows, so callers cut it through `faceit_history_matches` instead of materialising its id strings."""


def naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def cut(name: str, d: pd.DataFrame, before: pd.Timestamp, faceit_known: set | None = None) -> pd.DataFrame:
    """Keep the rows of table `name` known before `before`; FACEIT match tables need the ids of matches finished by then."""
    if name in DATE_COLUMN:
        d = d[naive(d[DATE_COLUMN[name]]) < before]
        if name == "match_history":
            d = d[naive(d.match_time.fillna(d.match_date)) < before]
        return d
    if name in FINISHED_AT:
        return d[d.finished_at < int(before.tz_localize("UTC").timestamp())]
    if name in BY_FACEIT_MATCH:
        return d[d.match_id.isin(faceit_known)]
    if name == "hltv_faceit_map":
        return d
    raise KeyError(f"no point-in-time rule for {name}")


class BundleTables:
    """The research bundle layout (`hltv/cs2_*.parquet`, `faceit/*.parquet`), as the model competitions shipped it."""

    PATHS = {"match_history": "hltv/cs2_match_history", "match_lineups": "hltv/cs2_match_lineups",
             "player_map_stats": "hltv/cs2_player_map_stats", "team_rankings": "hltv/cs2_team_rankings",
             "hltv_faceit_map": "faceit/hltv_faceit_map", "faceit_matches": "faceit/faceit_matches",
             "faceit_match_history": "faceit/faceit_match_history", "faceit_rosters": "faceit/faceit_rosters",
             "faceit_exclusions": "faceit/faceit_exclusions", "faceit_history_matches": "faceit/history_matches"}

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        return self.root / f"{self.PATHS[name]}.parquet"

    def load(self, name: str, before: pd.Timestamp | None = None, columns: list[str] | None = None) -> pd.DataFrame:
        path = self._path(name)
        if before is None:
            return pd.read_parquet(path, columns=columns)
        before = pd.Timestamp(before)
        if name in FINISHED_AT:
            secs = int(before.tz_localize("UTC").timestamp())
            return pq.read_table(path, columns=columns, filters=[("finished_at", "<", secs)]).to_pandas()
        known = None
        if name in BY_FACEIT_MATCH:
            known = set(self.load("faceit_matches", before, ["match_id"]).match_id)
        need = columns
        if columns is not None and name in DATE_COLUMN:
            need = list(dict.fromkeys(columns + [DATE_COLUMN[name]] + (["match_time"] if name == "match_history" else [])))
        d = cut(name, pd.read_parquet(path, columns=need), before, known)
        return d if columns is None else d[columns]

    def history_players(self) -> pa.Table:
        return pq.read_table(self.root / "faceit/history_players.parquet", columns=["match_id", "faction", "faceit_id"],
                             read_dictionary=["match_id", "faction", "faceit_id"])

