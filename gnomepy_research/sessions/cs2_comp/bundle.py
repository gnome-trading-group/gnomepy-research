"""Read the competition bundle with a point-in-time cut.

`load(data_dir, table, before)` returns the rows of one bundle table dated strictly before `before` (a naive UTC
timestamp, normally a target's week_cutoff). Tables without a date of their own are cut through the FACEIT matches
they belong to; every FACEIT table is cut by its match's finished_at. Using this for every read is the simplest way to
stay point-in-time."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATE_COLUMN = {
    "hltv/cs2_match_history": "match_date",
    "hltv/cs2_match_lineups": "match_date",
    "hltv/cs2_player_map_stats": "match_date",
    "hltv/cs2_h2h_history": "match_date",
    "hltv/cs2_team_recent_form": "match_date",
    "hltv/cs2_team_rankings": "date",
    "hltv/cs2_roster_moves": "event_date",
}
# A FACEIT match counts as known once FACEIT finalised it (finished_at), not when it was played: some April ESEA maps
# were only finalised in late May. Every FACEIT table follows its match, so all of them agree on what was known.
BY_FACEIT_MATCH = {"faceit/faceit_match_history", "faceit/faceit_lineups", "faceit/faceit_rosters",
                   "faceit/faceit_player_map_stats", "faceit/faceit_exclusions"}
UNIX_SECONDS = {"faceit/faceit_matches": "finished_at", "faceit/history_matches": "finished_at"}


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def load(data_dir: str | Path, table: str, before: pd.Timestamp | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(data_dir) / f"{table}.parquet"
    if before is None or table == "faceit/hltv_faceit_map":
        return pd.read_parquet(path, columns=columns)
    before = pd.Timestamp(before)
    secs = int(before.tz_localize("UTC").timestamp())
    if table in UNIX_SECONDS:
        t = pq.read_table(path, columns=columns, filters=[(UNIX_SECONDS[table], "<", secs)])
        return t.to_pandas()
    if table in DATE_COLUMN:
        d = pd.read_parquet(path, columns=columns)
        d = d[_naive(d[DATE_COLUMN[table]]) < before]
        if table == "hltv/cs2_match_history" and "match_time" in d:
            d = d[_naive(d.match_time.fillna(d.match_date)) < before]
        return d
    if table in BY_FACEIT_MATCH:
        ids = load(data_dir, "faceit/faceit_matches", before, ["match_id"]).match_id
        d = pd.read_parquet(path, columns=columns)
        return d[d.match_id.isin(ids)]
    if table == "faceit/faceit_hltv_series_link":
        f = load(data_dir, "faceit/faceit_matches", before, ["match_id"]).match_id
        h = load(data_dir, "hltv/cs2_match_history", before, ["match_id", "match_date", "match_time"]).match_id
        d = pd.read_parquet(path, columns=columns)
        return d[d.faceit_match_id.isin(f) & d.hltv_match_id.isin(h)]
    if table == "faceit/history_players":
        ids = pq.read_table(Path(data_dir) / "faceit/history_matches.parquet", columns=["match_id"],
                            filters=[("finished_at", "<", secs)]).column("match_id")
        t = pq.read_table(path, columns=columns)
        return t.filter(pc.is_in(t["match_id"], value_set=ids)).to_pandas()
    raise KeyError(f"no point-in-time rule for {table}")
