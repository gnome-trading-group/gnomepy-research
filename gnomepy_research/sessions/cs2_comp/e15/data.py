"""
Map observations from HLTV and FACEIT on one player and team id space, point in time.

Every map is a paired comparison between two fives: regulation rounds won by A out of rounds played (an overtime map
counts as 12-12 of 24). HLTV ids are kept as they are; a FACEIT player is his HLTV id when hltv_faceit_map links him,
otherwise a negative id hashed from his FACEIT id, so unlinked FACEIT players still connect the rating graph. FACEIT
teams take the HLTV team FACEIT inferred for the faction, otherwise a hashed negative id.

`load_raw` reads the bundle once; `observations(raw, cutoff)` keeps only what was known before the cutoff, using the
same rules as the bundle's point-in-time loader (HLTV by match_date and match_time, FACEIT by finished_at), so a fit
on the full bundle and one on a bundle cut at the cutoff see identical rows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.bayes_series import event_region
from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

LINEUP_SIZE = 5
HLTV, FACEIT = 0, 1
# ESEA league ladder, lowest to highest: the division a FACEIT-only player competes in is his prior class.
DIVISIONS = (("open", 1), ("entry", 2), ("intermediate", 3), ("main", 4), ("advanced", 5), ("challenger", 5), ("premier", 6))
N_DIVISIONS = 7
# FACEIT server regions on the region codes of bayes_series.event_region (1 Europe .. 6 CIS, 0 unknown).
FACEIT_REGION = {"EU": 1, "NA": 2, "SA": 3, "SEA": 4, "OCE": 5}


def hashed_id(faceit_id: str) -> int:
    """A stable negative id for an unlinked FACEIT player or team, independent of which rows are loaded."""
    return -(int(faceit_id.replace("-", "")[:15], 16) + 1)


def division(competition_name: str | None) -> int:
    n = (competition_name or "").lower()
    if "esea" not in n:
        return 0
    level = 0
    for word, lvl in DIVISIONS:
        if re.search(rf"\b{word}", n):
            level = max(level, lvl)
    return level


@dataclass
class Raw:
    maps: pd.DataFrame        # one row per map, both sources, sorted by (known_at, match_id, map position)
    rankings: pd.DataFrame
    ratings: pd.DataFrame     # match_id, map_name, player, src, perf: one row per player per map


def _rounds(score_a: np.ndarray, score_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    overtime = score_a + score_b > 24
    return np.where(overtime, 12, score_a), np.where(overtime, 24, score_a + score_b)


def _hltv_maps(data_dir: Path) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history")
    h = h[~showmatch_mask(h) & (h.team_a_score != h.team_b_score)].copy()
    lu = load(data_dir, "hltv/cs2_match_lineups", columns=["match_id", "team_id", "player_id", "slot"])
    fives = {k: tuple(int(p) for p in g.sort_values("slot").player_id.tolist()[:LINEUP_SIZE])
             for k, g in lu.groupby(["match_id", "team_id"])}

    def five(played, match_id, team_id):
        if played is not None and len(played) == LINEUP_SIZE:
            return tuple(int(p) for p in played)
        return fives.get((match_id, team_id))

    h["five_a"] = [five(p, m, t) for p, m, t in zip(h.team_a_player_ids, h.match_id, h.team_a_id)]
    h["five_b"] = [five(p, m, t) for p, m, t in zip(h.team_b_player_ids, h.match_id, h.team_b_id)]
    h = h[h.five_a.map(lambda x: x is not None and len(set(x)) == LINEUP_SIZE)
          & h.five_b.map(lambda x: x is not None and len(set(x)) == LINEUP_SIZE)]
    match_date = pd.to_datetime(h.match_date)
    match_time = pd.to_datetime(h.match_time)
    match_time = match_time.dt.tz_convert(None) if match_time.dt.tz is not None else match_time
    ka, n = _rounds(h.team_a_score.to_numpy(int), h.team_b_score.to_numpy(int))
    return pd.DataFrame({
        "match_id": h.match_id.astype(str).to_numpy(), "src": HLTV,
        "known_at": np.maximum(match_date.to_numpy(), match_time.fillna(match_date).to_numpy()),
        "day": match_date.dt.normalize().to_numpy(), "pos": h.map_position_in_series.fillna(0).to_numpy(int),
        "map_name": h.map_name.astype(str).to_numpy(),
        "team_a": h.team_a_id.astype(np.int64).to_numpy(), "team_b": h.team_b_id.astype(np.int64).to_numpy(),
        "five_a": h.five_a.to_numpy(), "five_b": h.five_b.to_numpy(), "rounds_a": ka, "rounds": n,
        "bo_type": h.bo_type.fillna(0).to_numpy(int), "tier": h.event_tier.fillna(0).to_numpy(int),
        "is_lan": h.is_lan.fillna(0).to_numpy(int), "division": 0,
        "region": np.array([event_region(e) for e in h.event_name]),
    })


def _faceit_maps(data_dir: Path) -> pd.DataFrame:
    fm = load(data_dir, "faceit/faceit_matches", columns=["match_id", "finished_at", "competition_name", "has_stats", "region"])
    fm = fm[fm.has_stats.fillna(False)]
    ex = load(data_dir, "faceit/faceit_exclusions")
    dead_series = set(ex.loc[ex.map_name.isna(), "match_id"])
    dead_maps = set(zip(ex.loc[ex.map_name.notna(), "match_id"], ex.loc[ex.map_name.notna(), "map_name"]))
    fh = pd.read_parquet(Path(data_dir) / "faceit/faceit_match_history.parquet")
    fh = fh[fh.match_id.isin(set(fm.match_id)) & ~fh.match_id.isin(dead_series)]
    fh = fh[[(m, n) not in dead_maps for m, n in zip(fh.match_id, fh.map_name)]]
    fh = fh[(fh.team_a_players == LINEUP_SIZE) & (fh.team_b_players == LINEUP_SIZE) & (fh.team_a_score != fh.team_b_score)]
    link = dict(zip(pd.read_parquet(Path(data_dir) / "faceit/hltv_faceit_map.parquet").faceit_id,
                    pd.read_parquet(Path(data_dir) / "faceit/hltv_faceit_map.parquet").hltv_id.astype(np.int64)))

    def five(ids):
        if ids is None or len(ids) != LINEUP_SIZE:
            return None
        out = tuple(int(link[f]) if f in link else hashed_id(f) for f in ids)
        return out if len(set(out)) == LINEUP_SIZE else None

    fh = fh.assign(five_a=fh.team_a_faceit_ids.map(five), five_b=fh.team_b_faceit_ids.map(five))
    fh = fh[fh.five_a.notna() & fh.five_b.notna()]
    meta = fm.set_index("match_id")
    finished = pd.to_datetime(meta.finished_at.reindex(fh.match_id).to_numpy(), unit="s")
    comp = meta.competition_name.reindex(fh.match_id).to_numpy()
    region = meta.region.reindex(fh.match_id).map(FACEIT_REGION).fillna(0).astype(int).to_numpy()

    def team(hltv_team, faceit_team):
        return np.array([int(h) if pd.notna(h) else hashed_id(f) for h, f in zip(hltv_team, faceit_team)], dtype=np.int64)

    ka, n = _rounds(fh.team_a_score.to_numpy(int), fh.team_b_score.to_numpy(int))
    return pd.DataFrame({
        "match_id": fh.match_id.to_numpy(), "src": FACEIT, "known_at": finished.to_numpy(),
        "day": pd.to_datetime(fh.match_date).dt.normalize().to_numpy(), "pos": fh.map_position_in_series.fillna(0).to_numpy(int),
        "map_name": fh.map_name.astype(str).to_numpy(),
        "team_a": team(fh.team_a_id, fh.team_a_faceit_id), "team_b": team(fh.team_b_id, fh.team_b_faceit_id),
        "five_a": fh.five_a.to_numpy(), "five_b": fh.five_b.to_numpy(), "rounds_a": ka, "rounds": n,
        "bo_type": fh.bo_type.fillna(0).to_numpy(int), "tier": -1, "is_lan": fh.is_lan.fillna(0).to_numpy(int),
        "division": np.array([division(c) for c in comp]), "region": region,
    })


def _hltv_ratings(data_dir: Path) -> pd.DataFrame:
    r = load(data_dir, "hltv/cs2_player_map_stats", columns=["match_id", "map_name", "player_id", "side", "rating"])
    r = r[(r.side == "all") & r.rating.notna()]
    return pd.DataFrame({"match_id": r.match_id.astype(str).to_numpy(), "map_name": r.map_name.astype(str).to_numpy(),
                         "player": r.player_id.astype(np.int64).to_numpy(), "src": HLTV, "perf": r.rating.to_numpy(float)})


def _faceit_ratings(data_dir: Path, link: dict) -> pd.DataFrame:
    """A per-map performance score from FACEIT stats: kill-death margin per round plus damage per round."""
    r = pd.read_parquet(data_dir / "faceit/faceit_player_map_stats.parquet",
                        columns=["match_id", "map_name", "faceit_id", "kills", "deaths", "adr"])
    fh = pd.read_parquet(data_dir / "faceit/faceit_match_history.parquet", columns=["match_id", "map_name", "team_a_score", "team_b_score"])
    rounds = dict(zip(zip(fh.match_id, fh.map_name), fh.team_a_score + fh.team_b_score))
    n = np.array([rounds.get(k, np.nan) for k in zip(r.match_id, r.map_name)], dtype=float)
    perf = (r.kills.to_numpy(float) - r.deaths.to_numpy(float)) / n + r.adr.to_numpy(float) / 100.0
    out = pd.DataFrame({"match_id": r.match_id.to_numpy(), "map_name": r.map_name.astype(str).to_numpy(),
                        "player": [int(link[f]) if f in link else hashed_id(f) for f in r.faceit_id],
                        "src": FACEIT, "perf": perf})
    return out[np.isfinite(out.perf)]


def load_raw(data_dir: str | Path, use_faceit: bool, ratings: bool = True, faceit_stats: bool = False) -> Raw:
    data_dir = Path(data_dir)
    parts = [_hltv_maps(data_dir)] + ([_faceit_maps(data_dir)] if use_faceit else [])
    maps = pd.concat(parts, ignore_index=True).sort_values(["known_at", "match_id", "pos"], kind="stable").reset_index(drop=True)
    rankings = load(data_dir, "hltv/cs2_team_rankings")
    rankings = rankings.assign(date=pd.to_datetime(rankings["date"]))
    if rankings.date.dt.tz is not None:
        rankings["date"] = rankings.date.dt.tz_convert(None)
    rating_parts = []
    if ratings:
        rating_parts.append(_hltv_ratings(data_dir))
        if use_faceit and faceit_stats:
            m = pd.read_parquet(data_dir / "faceit/hltv_faceit_map.parquet")
            rating_parts.append(_faceit_ratings(data_dir, dict(zip(m.faceit_id, m.hltv_id.astype(np.int64)))))
    rt = pd.concat(rating_parts, ignore_index=True) if rating_parts else \
        pd.DataFrame({"match_id": [], "map_name": [], "player": np.array([], np.int64), "src": [], "perf": []})
    return Raw(maps=maps, rankings=rankings, ratings=rt)


def before(raw: Raw, cutoff: pd.Timestamp, window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Maps known before the cutoff and inside the window, and the rankings published before it."""
    c = np.datetime64(pd.Timestamp(cutoff))
    m = raw.maps
    keep = (m.known_at.to_numpy() < c) & (m.day.to_numpy() < c) & (m.day.to_numpy() >= c - np.timedelta64(window_days, "D"))
    return m[keep].reset_index(drop=True), raw.rankings[raw.rankings.date.to_numpy() < c]
