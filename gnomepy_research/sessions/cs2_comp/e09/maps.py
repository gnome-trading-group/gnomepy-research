"""One chronological table of maps from both sources, point in time.

HLTV maps and FACEIT organised-competition maps become the same row shape: two fives of integer player keys and the
rounds each side won. A player linked through hltv_faceit_map has one key (the HLTV id) on both sources; a FACEIT
player with no link gets a negative key of its own, numbered in order of first appearance, so unlinked opponents
still connect the rating graph."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

EPOCH = pd.Timestamp("2023-01-01")
# A FACEIT series and its HLTV copy start up to 26 hours apart. Holding FACEIT maps back this long means a series is
# deduplicated against its HLTV copy the same way whether the bundle is cut at this cutoff or at a later one.
FACEIT_LAG = pd.Timedelta(hours=36)
LINEUP_SIZE = 5
FACEIT_PERF = (0.864, -0.533, 0.0996)
NON_DUPLICATE_REASONS = ("no stats (forfeit)", "wrong map count for best-of", "implausible map score",
                         "winner disagrees with score")

CLASS_PATTERNS = (
    ("esea_premier", r"premier|invite"),
    ("esea_advanced", r"advanced"),
    ("esea_main", r"\bmain\b"),
    ("esea_intermediate", r"intermediate"),
    ("esea_open", r"\bopen"),
    ("esea_entry", r"\bentry\b"),
    ("ecl", r"\becl\b|challenger league"),
    ("cct", r"\bcct\b|cct "),
    ("qualifier", r"qualif|open qual"),
)


def competition_class(name: str | None) -> str:
    n = (name or "").lower()
    for label, pattern in CLASS_PATTERNS:
        if re.search(pattern, n):
            return label
    return "other"


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _days(t: pd.Series) -> np.ndarray:
    return ((t - EPOCH).dt.total_seconds() / 86400.0).to_numpy()


def _five(ids) -> tuple[int, ...] | None:
    if ids is None or isinstance(ids, float):
        return None
    five = tuple(int(p) for p in ids if not pd.isna(p))
    return five if len(five) == LINEUP_SIZE and len(set(five)) == LINEUP_SIZE else None


def _hltv_perf(data_dir: str | Path, before: pd.Timestamp, m: pd.DataFrame) -> pd.DataFrame:
    st = load(data_dir, "hltv/cs2_player_map_stats", before, ["match_id", "match_date", "map_name", "player_id", "side", "rating"])
    st = st[(st.side == "all") & st.rating.notna()]
    table = {(str(mid), mp, int(p)): r for mid, mp, p, r in zip(st.match_id, st.map_name, st.player_id, st.rating)}
    for side in ("a", "b"):
        m[f"s{side}"] = [tuple(table.get((mid, mp, p), np.nan) for p in five) for mid, mp, five in zip(m.match_id, m.map_name, m[f"p{side}"])]
    return m


def _faceit_perf(data_dir: str | Path, before: pd.Timestamp, fm: pd.DataFrame) -> pd.DataFrame:
    st = load(data_dir, "faceit/faceit_player_map_stats", before, ["match_id", "map_name", "faceit_id", "kills", "deaths", "assists"])
    rounds = (fm.team_a_score + fm.team_b_score).to_numpy(float)
    n = len(fm)
    for side in ("a", "b"):
        ids = np.concatenate([np.asarray(list(x), dtype=object) for x in fm[f"team_{side}_faceit_ids"]])
        long = pd.DataFrame({"row": np.repeat(np.arange(n), LINEUP_SIZE), "slot": np.tile(np.arange(LINEUP_SIZE), n), "faceit_id": ids,
                             "match_id": np.repeat(fm.match_id.to_numpy(), LINEUP_SIZE), "map_name": np.repeat(fm.map_name.to_numpy(), LINEUP_SIZE)})
        long = long.merge(st, on=["match_id", "map_name", "faceit_id"], how="left")
        r = rounds[long.row.to_numpy()]
        long["perf"] = FACEIT_PERF[0] * long.kills / r + FACEIT_PERF[1] * long.deaths / r + FACEIT_PERF[2] * long.assists / r
        grid = np.full((n, LINEUP_SIZE), np.nan)
        grid[long.row.to_numpy(), long.slot.to_numpy()] = long.perf.to_numpy()
        fm[f"s{side}"] = [tuple(row) for row in grid]
    return fm


def hltv_maps(data_dir: str | Path, before: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", before)
    h = h[~showmatch_mask(h) & h.team_a_score.notna() & h.team_b_score.notna() & (h.team_a_score + h.team_b_score > 0)].copy()
    h["time"] = _naive(h.match_time.fillna(h.match_date))
    h["pa"] = h.team_a_player_ids.map(_five)
    h["pb"] = h.team_b_player_ids.map(_five)
    h = h[h.pa.notna() & h.pb.notna()].copy()
    h["match_id"] = h.match_id.astype(str)
    h = _hltv_perf(data_dir, before, h)
    return pd.DataFrame({
        "src": "hltv", "map_name": h.map_name.to_numpy(), "sa": h.sa.to_numpy(), "sb": h.sb.to_numpy(), "time": h.time.to_numpy(), "match_id": h.match_id.to_numpy(),
        "map_pos": h.map_position_in_series.fillna(0).astype(int).to_numpy(), "bo": h.bo_type.fillna(0).astype(int).to_numpy(),
        "team_a": h.team_a_id.astype(int).to_numpy(),
        "team_b": h.team_b_id.astype(int).to_numpy(), "pa": h.pa.to_numpy(), "pb": h.pb.to_numpy(),
        "ra": h.team_a_score.astype(int).to_numpy(), "rb": h.team_b_score.astype(int).to_numpy(),
        "comp": np.where(h.is_lan.fillna(0).astype(int) == 1, "hltv_lan", "hltv_online"),
        "tier": h.event_tier.fillna(0).astype(int).to_numpy(),
    })


def faceit_maps(data_dir: str | Path, before: pd.Timestamp) -> pd.DataFrame:
    fm = load(data_dir, "faceit/faceit_match_history", before)
    ex = load(data_dir, "faceit/faceit_exclusions", before)
    link = load(data_dir, "faceit/faceit_hltv_series_link", before)
    pair = load(data_dir, "faceit/hltv_faceit_map")
    to_hltv = dict(zip(pair.faceit_id, pair.hltv_id.astype(int)))

    whole = set(ex[ex.map_name.isna() & ex.reason.isin(NON_DUPLICATE_REASONS)].match_id) | set(link.faceit_match_id)
    maps_out = ex[ex.map_name.notna() & ex.reason.isin(NON_DUPLICATE_REASONS)]
    drop_map = set(zip(maps_out.match_id, maps_out.map_name))
    fm = fm[~fm.match_id.isin(whole)]
    fm = fm[[(m, n) not in drop_map for m, n in zip(fm.match_id, fm.map_name)]]
    fm = fm[(fm.team_a_players == LINEUP_SIZE) & (fm.team_b_players == LINEUP_SIZE) & fm.team_a_score.notna()
            & fm.team_b_score.notna() & (fm.team_a_score + fm.team_b_score > 0)].copy()
    fm["time"] = _naive(fm.match_time)
    fm = fm[fm.time + FACEIT_LAG < before]

    meta = load(data_dir, "faceit/faceit_matches", before, ["match_id", "competition_name", "region"]).set_index("match_id")
    cls = {m: competition_class(n) for m, n in meta.competition_name.items()}
    fm["comp"] = fm.match_id.map(cls).fillna("other")

    unlinked: dict[str, int] = {}

    def key(fid: str) -> int:
        h = to_hltv.get(fid)
        if h is not None:
            return h
        return unlinked.setdefault(fid, -(len(unlinked) + 1))

    fm = fm.sort_values(["time", "match_id", "map_position_in_series"], kind="stable")
    fm["pa"] = [_five([key(f) for f in ids]) for ids in fm.team_a_faceit_ids]
    fm["pb"] = [_five([key(f) for f in ids]) for ids in fm.team_b_faceit_ids]
    fm = fm[fm.pa.notna() & fm.pb.notna()].copy()
    fm = _faceit_perf(data_dir, before, fm)
    return pd.DataFrame({
        "src": "faceit", "map_name": fm.map_name.to_numpy(), "sa": fm.sa.to_numpy(), "sb": fm.sb.to_numpy(), "time": fm.time.to_numpy(), "match_id": fm.match_id.to_numpy(),
        "map_pos": fm.map_position_in_series.fillna(0).astype(int).to_numpy(), "bo": fm.bo_type.fillna(0).astype(int).to_numpy(),
        "team_a": fm.team_a_id.fillna(0).astype(int).to_numpy(), "team_b": fm.team_b_id.fillna(0).astype(int).to_numpy(),
        "pa": fm.pa.to_numpy(), "pb": fm.pb.to_numpy(), "ra": fm.team_a_score.astype(int).to_numpy(),
        "rb": fm.team_b_score.astype(int).to_numpy(), "comp": fm.comp.to_numpy(), "tier": 0,
    })


def build_maps(data_dir: str | Path, before: pd.Timestamp, use_faceit: bool = True) -> pd.DataFrame:
    parts = [hltv_maps(data_dir, before)]
    if use_faceit:
        parts.append(faceit_maps(data_dir, before))
    m = pd.concat(parts, ignore_index=True)
    m["avail"] = m.time.where(m.src == "hltv", m.time + FACEIT_LAG)
    m = m.sort_values(["avail", "src", "match_id", "map_pos"], kind="stable").reset_index(drop=True)
    m["day"] = _days(m.time)
    m["avail_day"] = _days(m.avail)
    return m
