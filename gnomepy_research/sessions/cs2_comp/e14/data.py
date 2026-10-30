"""
HLTV and FACEIT maps as one table of paired comparisons between fives, point in time.

Every map carries an `avail` time (the latest of its date, kickoff and FACEIT finish): a fit at a cutoff uses only
maps with avail before it. Players are HLTV ids where linked through hltv_faceit_map and otherwise a stable negative
id hashed from the FACEIT id, so unlinked FACEIT players stay in the rating graph. Teams are HLTV team ids where
FACEIT inferred one, otherwise a hashed FACEIT team id.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.bayes_series import event_region
from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

LINEUP = 5
HLTV, FACEIT = 0, 1
# ESEA ladder, bottom to top; anything else on FACEIT (cups, qualifiers, leagues) is division 0.
DIVISIONS = ("entry", "open", "intermediate", "main", "advanced", "challenger", "pro")
FACEIT_REGION = {"EU": 1, "NA": 2, "SA": 3, "SEA": 4, "OCE": 5}
_DIV_RE = re.compile(r"\b(" + "|".join(DIVISIONS) + r")", re.IGNORECASE)


def stable_id(key: str) -> int:
    """A negative int64 from a FACEIT uuid, identical on every machine and run."""
    return -int.from_bytes(hashlib.md5(key.encode()).digest()[:7], "big") - 1


def division(name: str | None) -> int:
    m = _DIV_RE.search(name or "")
    return DIVISIONS.index(m.group(1).lower()) + 1 if m else 0


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


@dataclass
class Bundle:
    maps: pd.DataFrame       # one row per map, both sources
    ratings: pd.DataFrame    # one row per player per map: map row, side (0 = A), player, rating deviation from own five
    ranks: pd.DataFrame      # HLTV rankings (date, team_id, rank)
    levels: pd.DataFrame     # FACEIT skill level per player per series (player, avail, level)
    moves: pd.DataFrame      # HLTV roster moves (event_date, person_id, kind, from/to team and status)


def _rounds(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regulation rounds and rounds won by A; an overtime map counts as 12-12 of 24."""
    ot = a + b > 24
    return np.where(ot, 24, a + b), np.where(ot, 12, a)


def _hltv_maps(data: Path, before: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = load(data, "hltv/cs2_match_history", before)
    h = h[(h.team_a_score != h.team_b_score) & ~showmatch_mask(h)].copy()
    lu = load(data, "hltv/cs2_match_lineups", before, ["match_id", "match_date", "team_id", "player_id", "slot"])
    fives = {k: g.sort_values("slot").player_id.astype(int).tolist()[:LINEUP] for k, g in lu.groupby(["match_id", "team_id"])}

    def five(played, match, team):
        if played is not None and len(played) == LINEUP:
            return [int(p) for p in played]
        return fives.get((match, team))

    h["la"] = [five(p, m, t) for p, m, t in zip(h.team_a_player_ids, h.match_id, h.team_a_id)]
    h["lb"] = [five(p, m, t) for p, m, t in zip(h.team_b_player_ids, h.match_id, h.team_b_id)]
    ok = h.la.map(lambda x: x is not None and len(x) == LINEUP and len(set(x)) == LINEUP) \
        & h.lb.map(lambda x: x is not None and len(x) == LINEUP and len(set(x)) == LINEUP)
    h = h[ok].copy()
    day = _naive(h.match_date)
    avail = np.maximum(day, _naive(h.match_time.fillna(h.match_date)))
    n, k = _rounds(h.team_a_score.to_numpy(int), h.team_b_score.to_numpy(int))
    maps = pd.DataFrame({
        "match_id": h.match_id.astype(str).to_numpy(), "map_name": h.map_name.to_numpy(), "source": HLTV,
        "day": day.dt.normalize().to_numpy(), "avail": avail.to_numpy(),
        "team_a": h.team_a_id.astype(np.int64).to_numpy(), "team_b": h.team_b_id.astype(np.int64).to_numpy(),
        "region": [event_region(e) for e in h.event_name], "division": -1, "is_lan": h.is_lan.astype(int).to_numpy(),
        "bo_type": h.bo_type.astype(int).to_numpy(), "rounds": n, "rounds_a": k,
        "la": h.la.to_numpy(), "lb": h.lb.to_numpy(),
    })
    st = load(data, "hltv/cs2_player_map_stats", before, ["match_id", "match_date", "map_name", "player_id", "team_id", "side", "rating"])
    st = st[(st.side == "all") & st.rating.notna()]
    st = st.assign(match_id=st.match_id.astype(str), player_id=st.player_id.astype(np.int64), value=st.rating.astype(float))
    return maps, st[["match_id", "map_name", "player_id", "team_id", "value"]]


def _faceit_maps(data: Path, before: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fm = load(data, "faceit/faceit_matches", before,
              ["match_id", "has_stats", "region", "competition_name", "finished_at"])
    fm = fm[fm.has_stats.fillna(False).astype(bool)]
    ex = load(data, "faceit/faceit_exclusions", before)
    link = load(data, "faceit/faceit_hltv_series_link", before, ["faceit_match_id", "hltv_match_id"])
    drop_series = set(ex[ex.map_name.isna()].match_id) | set(link.faceit_match_id)
    drop_maps = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
    h = load(data, "faceit/faceit_match_history", before,
             ["match_id", "map_name", "map_position_in_series", "match_date", "match_time", "bo_type", "team_a_id",
              "team_b_id", "team_a_faceit_id", "team_b_faceit_id", "team_a_faceit_ids", "team_b_faceit_ids",
              "team_a_players", "team_b_players", "team_a_score", "team_b_score"])
    h = h[h.match_id.isin(set(fm.match_id) - drop_series)]
    h = h[[(m, n) not in drop_maps for m, n in zip(h.match_id, h.map_name)]]
    h = h[(h.team_a_players == LINEUP) & (h.team_b_players == LINEUP) & (h.team_a_score != h.team_b_score)]
    h = h[h.team_a_faceit_ids.map(lambda x: x is not None and len(set(x)) == LINEUP)
          & h.team_b_faceit_ids.map(lambda x: x is not None and len(set(x)) == LINEUP)].copy()
    link_map = load(data, "faceit/hltv_faceit_map")
    to_hltv = dict(zip(link_map.faceit_id, link_map.hltv_id.astype(np.int64)))
    pid: dict[str, int] = {}

    def player(f: str) -> int:
        if f not in pid:
            pid[f] = to_hltv.get(f, stable_id(f))
        return pid[f]

    def team(hltv, fid):
        return int(hltv) if pd.notna(hltv) else stable_id(str(fid))

    meta = fm.set_index("match_id")
    fin = pd.to_datetime(meta.finished_at.reindex(h.match_id).to_numpy(), unit="s")
    day = _naive(h.match_date)
    avail = np.maximum(np.maximum(day.to_numpy(), _naive(h.match_time).to_numpy()), fin.to_numpy())
    n, k = _rounds(h.team_a_score.to_numpy(int), h.team_b_score.to_numpy(int))
    maps = pd.DataFrame({
        "match_id": h.match_id.to_numpy(), "map_name": h.map_name.to_numpy(), "source": FACEIT,
        "day": day.dt.normalize().to_numpy(), "avail": avail,
        "team_a": [team(a, f) for a, f in zip(h.team_a_id, h.team_a_faceit_id)],
        "team_b": [team(b, f) for b, f in zip(h.team_b_id, h.team_b_faceit_id)],
        "region": meta.region.reindex(h.match_id).map(FACEIT_REGION).fillna(0).astype(int).to_numpy(),
        "division": [division(c) for c in meta.competition_name.reindex(h.match_id)],
        "is_lan": 0, "bo_type": h.bo_type.astype(int).to_numpy(), "rounds": n, "rounds_a": k,
        "la": [[player(f) for f in x] for x in h.team_a_faceit_ids],
        "lb": [[player(f) for f in x] for x in h.team_b_faceit_ids],
        "map_pos": h.map_position_in_series.astype(int).to_numpy(),
    })
    st = load(data, "faceit/faceit_player_map_stats", before,
              ["match_id", "map_name", "map_position_in_series", "faceit_id", "faction", "k_r_ratio", "adr"])
    st = st[st.match_id.isin(set(maps.match_id)) & st.k_r_ratio.notna()].copy()
    # A rating-like number on the HLTV rating's scale: kills per round and ADR, each relative to its typical value.
    st["value"] = 0.5 * st.k_r_ratio / 0.69 + 0.5 * st.adr.fillna(st.k_r_ratio / 0.69 * 75.5) / 75.5
    st["player_id"] = [player(f) for f in st.faceit_id]
    ro = load(data, "faceit/faceit_rosters", before, ["match_id", "faceit_id", "skill_level"])
    ro = ro[ro.skill_level.fillna(0) > 0]
    levels = pd.DataFrame({"player": [to_hltv.get(f, stable_id(f)) for f in ro.faceit_id],
                           "avail": pd.to_datetime(meta.finished_at.reindex(ro.match_id).to_numpy(), unit="s"),
                           "level": ro.skill_level.astype(int).to_numpy()}).dropna(subset=["avail"])
    return maps, st, levels


def _rating_rows(maps: pd.DataFrame, stats: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Each stat row matched to its map row and side; value minus the mean of the five on that side."""
    idx = maps.reset_index()[["index", *keys, "la", "lb"]]
    r = stats.merge(idx, on=keys)
    in_a = np.array([p in a for p, a in zip(r.player_id, r.la)])
    in_b = np.array([p in b for p, b in zip(r.player_id, r.lb)])
    r = r[in_a ^ in_b].assign(side=np.where(in_a, 0, 1)[in_a ^ in_b])
    r = r.drop_duplicates(["index", "player_id"])
    r["dev"] = r.value - r.groupby(["index", "side"]).value.transform("mean")
    full = r.groupby(["index", "side"]).player_id.transform("size") == LINEUP
    r = r[full]
    return pd.DataFrame({"map_row": r["index"].to_numpy(np.int64), "side": r.side.to_numpy(np.int64),
                         "player": r.player_id.to_numpy(np.int64), "dev": r.dev.to_numpy(float)})


def load_bundle(data: str | Path, before: pd.Timestamp, faceit: bool = True) -> Bundle:
    """Every map and rating row available before `before` (the latest cutoff of the run)."""
    data = Path(data)
    hm, hst = _hltv_maps(data, before)
    parts, ratings = [hm], []
    levels = pd.DataFrame({"player": pd.Series(dtype=np.int64), "avail": pd.Series(dtype="datetime64[ns]"),
                           "level": pd.Series(dtype=np.int64)})
    if faceit:
        fmaps, fst, levels = _faceit_maps(data, before)
        parts.append(fmaps)
    maps = pd.concat(parts, ignore_index=True)
    maps = maps.sort_values(["avail", "source", "match_id", "map_name"], kind="mergesort").reset_index(drop=True)
    ratings.append(_rating_rows(maps[maps.source == HLTV], hst, ["match_id", "map_name"]).assign(source=HLTV))
    if faceit:
        fst = fst.rename(columns={"map_position_in_series": "map_pos"})
        ratings.append(_rating_rows(maps[maps.source == FACEIT], fst, ["match_id", "map_name", "map_pos"]).assign(source=FACEIT))
    ranks = load(data, "hltv/cs2_team_rankings", before, ["date", "team_id", "rank"])
    moves = load(data, "hltv/cs2_roster_moves", before,
                 ["event_date", "person_id", "kind", "from_team_id", "from_status", "to_team_id", "to_status"])
    moves = moves.assign(event_date=_naive(moves.event_date)).sort_values(["event_date", "person_id"], kind="mergesort")
    return Bundle(maps=maps, ratings=pd.concat(ratings, ignore_index=True), ranks=ranks, levels=levels,
                  moves=moves.reset_index(drop=True))
