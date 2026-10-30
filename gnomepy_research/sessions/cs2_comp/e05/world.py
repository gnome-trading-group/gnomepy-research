"""Everything a run reads from the bundle, loaded once at the latest cutoff it will need."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e05 import data as D
from gnomepy_research.sessions.cs2_comp.e05.pricing import PriceConfig, infer_lineups, roster_changes
from gnomepy_research.sessions.cs2_comp.e05.stream import Config, Filter, build_stream


@dataclass
class World:
    data_dir: Path
    last_cutoff: pd.Timestamp
    maps: D.Maps
    levels: pd.Series
    rankings: pd.DataFrame
    faceit_players: frozenset
    changes: pd.DataFrame
    links: dict
    lineup_rows: pd.DataFrame
    standins: frozenset
    streams: dict = field(default_factory=dict)
    lineup_cache: dict = field(default_factory=dict)

    def new_filter(self, cfg: Config) -> Filter:
        if cfg.faceit not in self.streams:
            self.streams[cfg.faceit] = build_stream(self.maps, cfg.faceit, self.links)
        stream = self.streams[cfg.faceit]
        levels = self.levels if cfg.faceit else self.levels.iloc[:0]
        return Filter(stream, cfg, self.rankings, levels)

    def history(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        return load(self.data_dir, "hltv/cs2_match_history", cutoff)

    def lineups(self, cutoff: pd.Timestamp, teams: set[int], price: PriceConfig) -> dict:
        key = (cutoff, frozenset(teams), price.lineup_series, price.lineup_decay, price.use_moves, price.faceit_lineups, price.standin_weight)
        if key not in self.lineup_cache:
            changes = self.changes[self.changes.date < cutoff] if price.use_moves else None
            extra = self.lineup_rows[self.lineup_rows.time < cutoff.tz_localize("UTC")] if price.faceit_lineups else None
            self.lineup_cache[key] = infer_lineups(self.history(cutoff), teams, price.lineup_series, price.lineup_decay, changes, extra, self.standins, price.standin_weight)
        return self.lineup_cache[key]


def load_world(data_dir: str | Path, last_cutoff: pd.Timestamp, faceit: bool = True) -> World:
    data_dir = Path(data_dir)
    stats = load(data_dir, "hltv/cs2_player_map_stats", last_cutoff, ["mapstatsid", "player_id", "side", "rating", "match_date"])
    hltv = D.hltv_maps(load(data_dir, "hltv/cs2_match_history", last_cutoff), stats)
    rankings = load(data_dir, "hltv/cs2_team_rankings", last_cutoff)
    changes = roster_changes(load(data_dir, "hltv/cs2_roster_moves", last_cutoff))
    lineups = load(data_dir, "hltv/cs2_match_lineups", last_cutoff, ["match_id", "player_id", "is_standin", "match_date"])
    lineups = lineups[lineups.is_standin.fillna(False).astype(bool)]
    standins = frozenset(zip(lineups.match_id.tolist(), lineups.player_id.astype(int).tolist()))
    if not faceit:
        return World(data_dir, last_cutoff, hltv, pd.Series(dtype=float), rankings, frozenset(), changes, {}, pd.DataFrame(), standins)
    link = pd.read_parquet(data_dir / "faceit/hltv_faceit_map.parquet")
    face, keys = D.faceit_maps(data_dir, last_cutoff)
    levels = D.faceit_first_levels(data_dir, last_cutoff, keys)
    links = {int(h): keys[f] for h, f in zip(link.hltv_id, link.faceit_id) if f in keys}
    return World(data_dir, last_cutoff, D.concat_maps([hltv, face]), levels, rankings, frozenset(links), changes, links, D.faceit_lineup_rows(data_dir, last_cutoff), standins)
