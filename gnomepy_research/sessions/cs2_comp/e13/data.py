"""
One time-ordered stream of maps from HLTV and FACEIT, keyed by HLTV ids where a link exists.

Each map carries the time its result became known, its source, both teams and both fives as integer ids, the rounds
each side won, and (FACEIT only) the ESEA division of its competition, which sets the prior for players debuting in it.
FACEIT players without an HLTV link keep their own ids (negative, from their FACEIT uuid) so that FACEIT opponents
still connect the rating graph; FACEIT teams without an HLTV team do the same.

HLTV maps are dated by match_time, which is what the bundle cut uses. FACEIT maps are dated by their series'
finished_at, which is when the bundle cut admits them, so a run on a truncated bundle replays the same stream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

HLTV, FACEIT = 0, 1
DIVISIONS = (("entry", 0), ("open", 1), ("intermediate", 2), ("main", 3), ("advanced", 4), ("premier", 5), ("pro league", 5))
UNKNOWN_DIVISION = -1


def synthetic_id(uuid: str) -> int:
    return -(int(str(uuid).replace("-", "")[:13], 16) + 1)


def division(competition_name: str | None) -> int:
    n = (competition_name or "").lower()
    if "esea" not in n and not re.search(r"\bs\d+ (eu|na|sa|oce|sea) ", n):
        return UNKNOWN_DIVISION
    for word, level in DIVISIONS:
        if re.search(rf"\b{word}\b", n):
            return level
    return UNKNOWN_DIVISION


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _ids(value) -> tuple[int, ...]:
    if value is None or isinstance(value, float):
        return ()
    return tuple(int(p) for p in value)


@dataclass
class Bundle:
    maps: pd.DataFrame          # t, src, match_id, map_pos, team_a, team_b, players_a, players_b, rounds_a, rounds_b, division
    series: pd.DataFrame        # HLTV series: match_id, t, team_a, team_b, bo_type, is_lan, event_name, y
    rankings: pd.DataFrame
    moves: pd.DataFrame
    faceit_level: dict[int, float]
    devs: dict[tuple[int, int, int], float]   # (map row, 0 for side a / 1 for b, player) -> score relative to his five


def hltv_maps(data: Path, before: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = load(data, "hltv/cs2_match_history", before)
    h = h[~showmatch_mask(h)].copy()
    h["t"] = _naive(h.match_time).fillna(_naive(h.match_date))
    h["players_a"] = h.team_a_player_ids.map(_ids)
    h["players_b"] = h.team_b_player_ids.map(_ids)
    ok = (h.players_a.map(len) == 5) & (h.players_b.map(len) == 5) \
        & (h.players_a.map(lambda x: len(set(x))) == 5) & (h.players_b.map(lambda x: len(set(x))) == 5) \
        & (h.team_a_score + h.team_b_score > 0) & (h.team_a_score != h.team_b_score)
    m = h[ok]
    maps = pd.DataFrame({
        "t": m.t, "src": HLTV, "match_id": m.match_id.astype(str), "map_name": m.map_name.astype(str),
        "map_pos": m.map_position_in_series.fillna(0).astype(int),
        "team_a": m.team_a_id.astype(np.int64), "team_b": m.team_b_id.astype(np.int64),
        "players_a": m.players_a, "players_b": m.players_b,
        "rounds_a": m.team_a_score.astype(int), "rounds_b": m.team_b_score.astype(int), "division": UNKNOWN_DIVISION,
    })
    s = h.groupby("match_id").agg(t=("t", "min"), team_a=("team_a_id", "first"), team_b=("team_b_id", "first"),
                                  bo_type=("bo_type", "first"), is_lan=("is_lan", "first"), event_name=("event_name", "first"),
                                  sa=("team_a_series_score", "max"), sb=("team_b_series_score", "max"),
                                  won=("team_a_won", "sum"), n=("team_a_won", "size")).reset_index()
    s["y"] = np.where(s.sa != s.sb, (s.sa > s.sb).astype(float), np.where(s.won * 2 != s.n, (s.won * 2 > s.n).astype(float), np.nan))
    return maps, s.drop(columns=["sa", "sb", "won", "n"])


def faceit_maps(data: Path, before: pd.Timestamp) -> tuple[pd.DataFrame, dict[int, float]]:
    matches = load(data, "faceit/faceit_matches", before,
                   ["match_id", "competition_name", "finished_at", "has_stats", "faction1_id", "faction2_id"])
    matches = matches[matches.has_stats]
    fm = load(data, "faceit/faceit_match_history", before)
    fm = fm[fm.match_id.isin(matches.match_id)]
    ex = load(data, "faceit/faceit_exclusions", before)
    whole = set(ex.match_id[ex.map_name.isna()])
    some = set(zip(ex.match_id[ex.map_name.notna()], ex.map_name[ex.map_name.notna()]))
    fm = fm[~fm.match_id.isin(whole)]
    fm = fm[[(m, n) not in some for m, n in zip(fm.match_id, fm.map_name)]]
    linked = set(load(data, "faceit/faceit_hltv_series_link", before).faceit_match_id)
    fm = fm[~fm.match_id.isin(linked)]
    link = pd.read_parquet(Path(data) / "faceit/hltv_faceit_map.parquet")
    to_hltv = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))

    def five(uuids) -> tuple[int, ...]:
        if uuids is None or len(uuids) != 5 or len(set(uuids)) != 5:
            return ()
        return tuple(int(to_hltv[u]) if u in to_hltv else synthetic_id(u) for u in uuids)

    fm = fm.copy()
    fm["players_a"] = fm.team_a_faceit_ids.map(five)
    fm["players_b"] = fm.team_b_faceit_ids.map(five)
    fm = fm[(fm.players_a.map(len) == 5) & (fm.players_b.map(len) == 5) & (fm.team_a_score != fm.team_b_score)
            & (fm.team_a_score + fm.team_b_score > 0)]
    fm = fm[[len(set(a + b)) == 10 for a, b in zip(fm.players_a, fm.players_b)]]
    meta = matches.set_index("match_id")
    finished = pd.to_datetime(fm.match_id.map(meta.finished_at), unit="s")
    team = lambda hl, uuid: np.where(hl.notna(), hl.fillna(0).astype(np.int64), [synthetic_id(u) for u in uuid])
    maps = pd.DataFrame({
        "t": finished, "src": FACEIT, "match_id": fm.match_id.astype(str), "map_name": fm.map_name.astype(str),
        "map_pos": fm.map_position_in_series.fillna(0).astype(int),
        "team_a": team(fm.team_a_id, fm.team_a_faceit_id), "team_b": team(fm.team_b_id, fm.team_b_faceit_id),
        "players_a": fm.players_a, "players_b": fm.players_b,
        "rounds_a": fm.team_a_score.astype(int), "rounds_b": fm.team_b_score.astype(int),
        "division": fm.match_id.map(meta.competition_name).map(division).astype(int),
    })
    ro = load(data, "faceit/faceit_rosters", before, ["faceit_id", "match_id", "skill_level", "played"])
    ro = ro[ro.played & ro.skill_level.notna()]
    ro = ro.assign(t=ro.match_id.map(meta.finished_at)).sort_values(["t", "match_id"])
    first = ro.drop_duplicates("faceit_id")
    level = {int(to_hltv[u]) if u in to_hltv else synthetic_id(u): float(s) for u, s in zip(first.faceit_id, first.skill_level)}
    return maps, level


def player_stats(data: Path, before: pd.Timestamp, maps: pd.DataFrame, use_faceit: bool) -> pd.DataFrame:
    """One row per (map row, player) with a per-map performance score relative to his four teammates on that map:
    HLTV rating on HLTV maps, (kills - deaths) per round on FACEIT maps."""
    rows = []
    for side in ("a", "b"):
        e = maps[["match_id", "map_name", "map_pos", "src", f"players_{side}", "rounds_a", "rounds_b"]].rename(
            columns={f"players_{side}": "player"}).explode("player")
        e["player"] = e.player.astype(np.int64)
        e["side"] = side
        e["map_row"] = e.index.to_numpy()
        rows.append(e)
    e = pd.concat(rows, ignore_index=True)
    h = load(data, "hltv/cs2_player_map_stats", before, ["match_id", "map_name", "player_id", "side", "rating", "match_date"])
    h = h[(h.side == "all") & h.rating.notna()]
    h = pd.DataFrame({"match_id": h.match_id.astype(str), "map_name": h.map_name.astype(str),
                      "player": h.player_id.astype(np.int64), "value": h.rating.astype(float)}).drop_duplicates(["match_id", "map_name", "player"])
    out = [e[e.src == HLTV].merge(h, on=["match_id", "map_name", "player"])]
    if use_faceit:
        f = load(data, "faceit/faceit_player_map_stats", before, ["match_id", "map_position_in_series", "faceit_id", "kills", "deaths"])
        link = pd.read_parquet(Path(data) / "faceit/hltv_faceit_map.parquet")
        to_hltv = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))
        f = f[f.kills.notna() & f.deaths.notna()]
        f = pd.DataFrame({"match_id": f.match_id.astype(str), "map_pos": f.map_position_in_series.fillna(0).astype(int),
                          "player": [int(to_hltv[u]) if u in to_hltv else synthetic_id(u) for u in f.faceit_id],
                          "kd": (f.kills - f.deaths).astype(float)}).drop_duplicates(["match_id", "map_pos", "player"])
        fe = e[e.src == FACEIT].merge(f, on=["match_id", "map_pos", "player"])
        fe["value"] = fe.kd / (fe.rounds_a + fe.rounds_b)
        out.append(fe.drop(columns="kd"))
    s = pd.concat(out, ignore_index=True)
    g = s.groupby(["map_row", "side"]).value
    s["n"] = g.transform("size")
    s["dev"] = s.value - g.transform("mean")
    return s[s.n == 5][["map_row", "side", "player", "src", "dev"]].reset_index(drop=True)


def build(data: Path, before: pd.Timestamp, use_faceit: bool) -> Bundle:
    hm, series = hltv_maps(data, before)
    parts, level = [hm], {}
    if use_faceit:
        fmaps, level = faceit_maps(data, before)
        parts.append(fmaps)
    maps = pd.concat(parts, ignore_index=True).sort_values(["t", "src", "match_id", "map_pos"], kind="mergesort").reset_index(drop=True)
    rankings = load(data, "hltv/cs2_team_rankings", before)
    moves = load(data, "hltv/cs2_roster_moves", before)
    moves = moves.assign(event_date=_naive(moves.event_date)).sort_values(["event_date", "person_id"], kind="mergesort")
    st = player_stats(data, before, maps, use_faceit)
    devs = dict(zip(zip(st.map_row.tolist(), (st.side == "b").astype(int).tolist(), st.player.tolist()), st.dev.tolist()))
    return Bundle(maps=maps, series=series, rankings=rankings, moves=moves, faceit_level=level, devs=devs)
