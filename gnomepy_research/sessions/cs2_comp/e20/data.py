"""
Point-in-time map rows from both sources on one player id space.

HLTV players keep their HLTV id. A FACEIT player linked through hltv_faceit_map takes the HLTV id; an unlinked one
gets a negative id of its own, so FACEIT-only teams still connect the rating graph through their opponents.

Each map yields (lineup_a, lineup_b, rounds_a, rounds, source, weight, prior group). Overtime maps count as 12-12
of 24 as in the Bayes model; the legacy 16-round format keeps its own regulation count.

The rating rows are the second measurement: a player's per-map stat minus the mean of the teammates who have
one (HLTV rating 2.0; FACEIT ADR per 100), with the five whose mean skill it is measured against.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.bayes_series import RankStrength, UNRANKED
from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

LINEUP_SIZE = 5
HLTV, FACEIT = 0, 1
DIVISIONS = ("open", "intermediate", "main", "advanced", "entry", "challenger")
HLTV_RANK_BINS = (10, 30, 100)
GROUPS = [f"hltv_rank_le{b}" for b in HLTV_RANK_BINS] + ["hltv_unranked"] + [f"faceit_{d}" for d in DIVISIONS] + ["faceit_other"]
GROUP_INDEX = {g: i for i, g in enumerate(GROUPS)}


@dataclass
class MapRows:
    lineup_a: np.ndarray    # (N, 5) unified player ids
    lineup_b: np.ndarray
    rounds_a: np.ndarray    # (N,)
    rounds: np.ndarray      # (N,)
    source: np.ndarray      # (N,) HLTV or FACEIT
    day: np.ndarray         # (N,) datetime64[ns]
    group: np.ndarray       # (N,) prior group index the map would give a debutant
    match_id: np.ndarray    # (N,) object
    team_a: np.ndarray      # (N,) HLTV team id or -1
    team_b: np.ndarray
    map_name: np.ndarray    # (N,) object
    # FACEIT id -> unified id, built once so the synthetic ids of unlinked players agree across tables.
    faceit_lookup: dict = None


@dataclass
class RatingRows:
    player: np.ndarray      # (R,) unified player id
    lineup: np.ndarray      # (R, 5) the player's own five on that map
    dev: np.ndarray         # (R,)
    source: np.ndarray      # (R,)
    day: np.ndarray         # (R,) datetime64[ns]


def hltv_rank_group(rank_strength: RankStrength, team: int, day: pd.Timestamp) -> int:
    rank = np.exp(-rank_strength(team, day))
    for b in HLTV_RANK_BINS:
        if rank <= b:
            return GROUP_INDEX[f"hltv_rank_le{b}"]
    return GROUP_INDEX["hltv_unranked"]


def faceit_group(event_name: str | None) -> int:
    n = (event_name or "").lower()
    for d in DIVISIONS:
        if re.search(rf"\b{d}", n):
            return GROUP_INDEX[f"faceit_{d}"]
    return GROUP_INDEX["faceit_other"]


def _regulation(score_a: np.ndarray, score_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rounds_a, rounds and a keep mask: overtime is 12-12 of 24, unfinished maps are dropped."""
    w, l = np.maximum(score_a, score_b), np.minimum(score_a, score_b)
    overtime = (w >= 16) & (l >= 12)
    rounds_a = np.where(overtime, 12, score_a)
    rounds = np.where(overtime, 24, score_a + score_b)
    keep = (w >= 13) & (score_a != score_b)
    return rounds_a, rounds, keep


def _ids(values, lookup: dict | None) -> np.ndarray:
    out = np.full((len(values), LINEUP_SIZE), 0, dtype=np.int64)
    ok = np.zeros(len(values), dtype=bool)
    for i, v in enumerate(values):
        if v is None or len(v) != LINEUP_SIZE:
            continue
        try:
            row = [lookup[x] for x in v] if lookup is not None else [int(x) for x in v]
        except (KeyError, ValueError, TypeError):
            continue
        if len(set(row)) == LINEUP_SIZE:
            out[i] = row
            ok[i] = True
    return out, ok


def faceit_player_lookup(data_dir, faceit_ids: pd.Series) -> dict[str, int]:
    """FACEIT id -> unified id: the HLTV id where linked, otherwise a negative id of its own."""
    link = load(data_dir, "faceit/hltv_faceit_map")
    lookup = dict(zip(link.faceit_id, link.hltv_id.astype(int)))
    unlinked = sorted(set(faceit_ids) - set(lookup))
    lookup.update({f: -(i + 1) for i, f in enumerate(unlinked)})
    return lookup


def load_maps(data_dir, cutoff: pd.Timestamp, use_faceit: bool, rank_strength: RankStrength) -> MapRows:
    h = load(data_dir, "hltv/cs2_match_history", cutoff,
             ["match_id", "match_date", "event_name", "format", "team_a_id", "team_b_id", "team_a_player_ids",
              "team_b_player_ids", "team_a_score", "team_b_score", "map_position_in_series", "map_name"])
    h = h[~showmatch_mask(h)].sort_values(["match_date", "match_id", "map_position_in_series"])
    lookup: dict[str, int] = {}
    la, oka = _ids(h.team_a_player_ids.tolist(), None)
    lb, okb = _ids(h.team_b_player_ids.tolist(), None)
    ra, n, keep = _regulation(h.team_a_score.to_numpy(int), h.team_b_score.to_numpy(int))
    keep &= oka & okb
    day = pd.to_datetime(h.match_date).to_numpy()
    group = np.array([hltv_rank_group(rank_strength, int(t), d) for t, d in zip(h.team_a_id, h.match_date)])
    parts = [(la[keep], lb[keep], ra[keep], n[keep], np.full(keep.sum(), HLTV), day[keep], group[keep],
              h.match_id.to_numpy(object)[keep], h.team_a_id.to_numpy(int)[keep], h.team_b_id.to_numpy(int)[keep],
              h.map_name.to_numpy(object)[keep])]
    if use_faceit:
        f = load(data_dir, "faceit/faceit_match_history", cutoff,
                 ["match_id", "match_date", "event_name", "team_a_id", "team_b_id", "team_a_faceit_ids", "team_b_faceit_ids",
                  "team_a_score", "team_b_score", "team_a_players", "team_b_players", "map_name", "map_position_in_series"])
        ex = load(data_dir, "faceit/faceit_exclusions", cutoff)
        whole = set(ex[ex.map_name.isna()].match_id)
        bad_map = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
        f = f[~f.match_id.isin(whole) & (f.team_a_players == LINEUP_SIZE) & (f.team_b_players == LINEUP_SIZE)]
        f = f[[(m, mp) not in bad_map for m, mp in zip(f.match_id, f.map_name)]]
        f = f.sort_values(["match_date", "match_id", "map_position_in_series"])
        all_ids = pd.Series([x for v in f.team_a_faceit_ids for x in v] + [x for v in f.team_b_faceit_ids for x in v])
        lookup = faceit_player_lookup(data_dir, all_ids)
        la, oka = _ids(f.team_a_faceit_ids.tolist(), lookup)
        lb, okb = _ids(f.team_b_faceit_ids.tolist(), lookup)
        ra, n, keep = _regulation(f.team_a_score.to_numpy(int), f.team_b_score.to_numpy(int))
        keep &= oka & okb
        day = pd.to_datetime(f.match_date).to_numpy()
        group = np.array([faceit_group(e) for e in f.event_name])
        parts.append((la[keep], lb[keep], ra[keep], n[keep], np.full(keep.sum(), FACEIT), day[keep], group[keep],
                      f.match_id.to_numpy(object)[keep], f.team_a_id.fillna(-1).to_numpy(int)[keep],
                      f.team_b_id.fillna(-1).to_numpy(int)[keep], f.map_name.to_numpy(object)[keep]))
    cols = [np.concatenate([p[i] for p in parts]) for i in range(11)]
    order = np.lexsort((cols[7].astype(str), cols[5]))
    return MapRows(*[c[order] for c in cols], faceit_lookup=lookup)


def _rating_rows(stats: pd.DataFrame, maps: MapRows, source: int) -> RatingRows:
    """stats: match_id, map_name, side (0 = team A), player (unified id), stat; keeps players of the map's five."""
    idx = pd.DataFrame({"match_id": maps.match_id, "map_name": maps.map_name, "row": np.arange(len(maps.rounds))})
    idx = idx[maps.source == source]
    r = stats.merge(idx, on=["match_id", "map_name"])
    lineup = np.where(r.side.to_numpy()[:, None] == 0, maps.lineup_a[r.row], maps.lineup_b[r.row])
    r = r[(lineup == r.player.to_numpy()[:, None]).any(axis=1)]
    lineup = np.where(r.side.to_numpy()[:, None] == 0, maps.lineup_a[r.row], maps.lineup_b[r.row])
    dev = (r.stat - r.groupby(["row", "side"]).stat.transform("mean")).to_numpy(float)
    return RatingRows(player=r.player.to_numpy(np.int64), lineup=lineup, dev=dev, source=np.full(len(r), source),
                      day=maps.day[r.row.to_numpy()])


def load_ratings(data_dir, cutoff: pd.Timestamp, maps: MapRows, use_faceit: bool) -> RatingRows:
    st = load(data_dir, "hltv/cs2_player_map_stats", cutoff,
              ["match_id", "match_date", "map_name", "player_id", "team_id", "side", "rating"])
    st = st[(st.side == "all") & st.rating.notna()]
    team_a = pd.Series(maps.team_a, index=pd.MultiIndex.from_arrays([maps.match_id, maps.map_name]))
    team_a = team_a[~team_a.index.duplicated()]
    ta = team_a.reindex(pd.MultiIndex.from_arrays([st.match_id, st.map_name])).to_numpy()
    h = pd.DataFrame({"match_id": st.match_id.to_numpy(), "map_name": st.map_name.to_numpy(),
                      "side": np.where(st.team_id.to_numpy() == ta, 0, 1), "player": st.player_id.to_numpy(int),
                      "stat": st.rating.to_numpy(float)})
    parts = [_rating_rows(h, maps, HLTV)]
    if use_faceit:
        fs = load(data_dir, "faceit/faceit_player_map_stats", cutoff, ["match_id", "map_name", "faceit_id", "faction", "adr"])
        fs = fs[fs.adr.notna() & fs.faceit_id.isin(maps.faceit_lookup)]
        f = pd.DataFrame({"match_id": fs.match_id.to_numpy(), "map_name": fs.map_name.to_numpy(),
                          "side": np.where(fs.faction.to_numpy() == "faction1", 0, 1),
                          "player": fs.faceit_id.map(maps.faceit_lookup).to_numpy(int), "stat": fs.adr.to_numpy(float) / 100})
        parts.append(_rating_rows(f, maps, FACEIT))
    return RatingRows(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in RatingRows.__dataclass_fields__})


def latest_lineups(maps: MapRows) -> dict[int, tuple[np.ndarray, np.datetime64]]:
    """Each HLTV team's five from its most recent map in either source."""
    out: dict[int, tuple[np.ndarray, np.datetime64]] = {}
    for team_col, lineup in ((maps.team_a, maps.lineup_a), (maps.team_b, maps.lineup_b)):
        for i in np.flatnonzero(team_col >= 0):
            t = int(team_col[i])
            if t not in out or maps.day[i] >= out[t][1]:
                out[t] = (lineup[i], maps.day[i])
    return out


def unranked_strength() -> float:
    return -float(np.log(UNRANKED))
