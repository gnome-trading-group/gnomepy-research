"""Map events for the skill filter: HLTV and FACEIT maps on one clock, each with its players and optional ratings.

Everything is read through bundle.load so a run only sees rows before its latest cutoff. A FACEIT map takes its
series' finish time (the bundle cuts FACEIT matches by finished_at), an HLTV map its match time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

EPOCH = pd.Timestamp("2020-01-01")
LINEUP = 5


def day_number(ts: pd.Series | pd.Timestamp) -> np.ndarray | int:
    if isinstance(ts, pd.Timestamp):
        return int((ts.normalize() - EPOCH).days)
    return ((ts.dt.normalize() - EPOCH).dt.days).to_numpy()


def naive_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


@dataclass
class PlayerIndex:
    """Dense ids for every player. A FACEIT player linked to HLTV is the HLTV player; the rest keep their own id."""

    hltv: dict[int, int]
    faceit_only: dict[str, int]
    faceit_to_hltv: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.hltv) + len(self.faceit_only)

    def from_hltv(self, pid: int) -> int:
        i = self.hltv.get(pid)
        if i is None:
            i = self.hltv[pid] = len(self.hltv) + len(self.faceit_only)
        return i

    def from_faceit(self, fid: str) -> int:
        h = self.faceit_to_hltv.get(fid)
        if h is not None:
            return self.from_hltv(h)
        i = self.faceit_only.get(fid)
        if i is None:
            i = self.faceit_only[fid] = len(self.hltv) + len(self.faceit_only)
        return i


def new_index(data_dir: Path) -> PlayerIndex:
    m = load(data_dir, "faceit/hltv_faceit_map")
    return PlayerIndex({}, {}, dict(zip(m.faceit_id, m.hltv_id.astype(int))))


def hltv_maps(data_dir: Path, before: pd.Timestamp, ranks_cols: bool = False) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", before)
    h = h[~showmatch_mask(h) & (h.team_a_score != h.team_b_score)].copy()
    ok = h.team_a_player_ids.map(lambda x: x is not None and len(x) == LINEUP) \
        & h.team_b_player_ids.map(lambda x: x is not None and len(x) == LINEUP)
    h = h[ok]
    h["time"] = naive_utc(h.match_time.fillna(h.match_date))
    return h.sort_values(["time", "match_id", "map_position_in_series"]).reset_index(drop=True)


def hltv_rating_dev(data_dir: Path, before: pd.Timestamp) -> pd.DataFrame:
    """Per player-map rating minus the team's mean on that map, only where all five players have a rating."""
    s = load(data_dir, "hltv/cs2_player_map_stats", before, ["match_id", "match_date", "map_name", "player_id", "team_id", "side", "rating"])
    s = s[(s.side == "all") & s.rating.notna()]
    g = s.groupby(["match_id", "map_name", "team_id"]).rating
    s = s.assign(n=g.transform("size"), dev=s.rating - g.transform("mean"))
    s = s[s.n == LINEUP]
    s["player_id"] = s.player_id.astype(int)
    return s[["match_id", "map_name", "team_id", "player_id", "rating", "dev"]]


FACEIT_STAT_KEYS = [
    "kills", "deaths", "assists", "headshots", "mvps", "k_d_ratio", "k_r_ratio", "adr", "damage", "first_kills",
    "entry_count", "entry_wins", "clutch_kills", "1v1count", "1v1wins", "1v2count", "1v2wins", "double_kills",
    "triple_kills", "quadro_kills", "penta_kills", "pistol_kills", "sniper_kills", "utility_damage", "enemies_flashed",
    "flash_count", "flash_successes", "utility_count", "utility_successes", "utility_enemies", "knife_kills", "zeus_kills",
]


def faceit_maps(data_dir: Path, before: pd.Timestamp) -> pd.DataFrame:
    """Cleaned FACEIT maps with both fives as dense ids; one row per map, time = series finish."""
    fm = load(data_dir, "faceit/faceit_matches", before, ["match_id", "finished_at"])
    ex = load(data_dir, "faceit/faceit_exclusions", before)
    whole = set(ex[ex.map_name.isna()].match_id)
    part = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
    cols = ["match_id", "match_date", "map_name", "map_position_in_series", "team_a_faceit_ids", "team_b_faceit_ids",
            "team_a_score", "team_b_score", "team_a_id", "team_b_id"]
    h = load(data_dir, "faceit/faceit_match_history", before, cols)
    h = h[~h.match_id.isin(whole)]
    h = h[[(m, n) not in part for m, n in zip(h.match_id, h.map_name)]]
    h = h[h.team_a_score != h.team_b_score]
    ok = h.team_a_faceit_ids.map(lambda x: x is not None and len(x) == LINEUP and len(set(x)) == LINEUP) \
        & h.team_b_faceit_ids.map(lambda x: x is not None and len(x) == LINEUP and len(set(x)) == LINEUP)
    h = h[ok].merge(fm, on="match_id")
    h["time"] = pd.to_datetime(h.finished_at, unit="s")
    return h.sort_values(["time", "match_id", "map_position_in_series"]).reset_index(drop=True)


def faceit_stats(data_dir: Path, before: pd.Timestamp, match_ids: set[str]) -> pd.DataFrame:
    """Player-map stat rows of the given matches, numeric columns as float32 (the full table is 0.8 GB as float64)."""
    cut = set(load(data_dir, "faceit/faceit_matches", before, ["match_id"]).match_id) & set(match_ids)
    cols = ["match_id", "map_name", "faceit_id", "hltv_id", "faction"] + FACEIT_STAT_KEYS
    t = pq.read_table(Path(data_dir) / "faceit/faceit_player_map_stats.parquet", columns=cols)
    t = t.filter(pc.is_in(t["match_id"], value_set=pa.array(sorted(cut))))
    s = t.to_pandas()
    for c in FACEIT_STAT_KEYS:
        s[c] = s[c].astype("float32")
    return s


def first_skill_level(data_dir: Path, before: pd.Timestamp) -> dict[str, int]:
    r = load(data_dir, "faceit/faceit_rosters", before, ["match_id", "faceit_id", "skill_level"])
    fm = load(data_dir, "faceit/faceit_matches", before, ["match_id", "finished_at"])
    r = r.merge(fm, on="match_id").sort_values("finished_at").drop_duplicates("faceit_id")
    return dict(zip(r.faceit_id, r.skill_level.astype(int)))
