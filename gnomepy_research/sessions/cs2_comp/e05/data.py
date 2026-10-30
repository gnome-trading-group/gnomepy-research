"""Point-in-time inputs of the joint HLTV + FACEIT skill model.

Every map from both sources becomes one row between two fives of players. A player is one id across sources: the
HLTV id where the HLTV-FACEIT map links them, otherwise a FACEIT-only id that lives above FACEIT_ONLY_BASE, so FACEIT
opponents still connect the rating graph. Everything is cut at the fit cutoff exactly as the competition's
point-in-time check cuts a bundle (FACEIT rows by the finish time of their match, HLTV rows by date and kickoff)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.bayes_series import RankStrength
from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

FACEIT_ONLY_BASE = 100_000_000
ORIGIN = pd.Timestamp("2023-10-01")
TEAM_SIZE = 5
ADR_MEAN = 75.0
SOURCES = ("hltv", "entry", "open", "inter", "main", "adv", "prem", "ecl", "cct", "dach", "qual", "other")
SOURCE_INDEX = {s: i for i, s in enumerate(SOURCES)}
FACEIT_SOURCES = SOURCES[1:]
_TIER_PATTERNS = (
    ("prem", r"premier|invite"), ("adv", r"advanced"), ("main", r"\bmain\b"), ("inter", r"intermediate"),
    ("open", r"\bopen"), ("entry", r"\bentry"), ("ecl", r"challenger league|\becl\b"), ("cct", r"\bcct\b"),
    ("dach", r"dach|spielklasse"), ("qual", r"qualifier"),
)


def faceit_source(event_name: str | None) -> str:
    name = (event_name or "").lower()
    for source, pattern in _TIER_PATTERNS:
        if re.search(pattern, name):
            return source
    return "other"


def regulation(score_a: np.ndarray, score_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regulation rounds won by A and played: an overtime map counts as 12-12 of 24."""
    overtime = score_a + score_b > 24
    return np.where(overtime, 12, score_a), np.where(overtime, 24, score_a + score_b)


@dataclass
class Maps:
    day: np.ndarray            # (N,) datetime64[D]: the day a point-in-time cut sees the map from
    source: np.ndarray         # (N,) index into SOURCES
    rounds_a: np.ndarray       # (N,)
    rounds: np.ndarray         # (N,)
    win_a: np.ndarray          # (N,) 1 when A won the map, overtime included
    team_a: np.ndarray         # (N,) HLTV team id or -1
    team_b: np.ndarray
    side_a: np.ndarray         # (N, 5) player key
    side_b: np.ndarray
    perf_a: np.ndarray         # (N, 5) rating-like performance, centred on the map's ten players; NaN where there is no stat line
    perf_b: np.ndarray


def _fives(column: pd.Series) -> np.ndarray:
    return np.array([list(x) for x in column], dtype=np.int64).reshape(-1, TEAM_SIZE)


def _centred(perf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an (N, 10) performance array into the two sides after subtracting each map's mean; maps with any gap are all NaN."""
    full = np.isfinite(perf).all(axis=1)
    perf = np.where(full[:, None], perf - perf.mean(axis=1, keepdims=True), np.nan)
    return perf[:, :TEAM_SIZE], perf[:, TEAM_SIZE:]


def _lookup(n: int, keys: dict[str, np.ndarray], stats: pd.DataFrame, value: str) -> np.ndarray:
    """(N, 10) values of `value`, looked up by the key columns of the stats table for each of the ten slots of each map."""
    long = pd.DataFrame({"row": np.repeat(np.arange(n), 2 * TEAM_SIZE), "slot": np.tile(np.arange(2 * TEAM_SIZE), n),
                         **{k: v for k, v in keys.items()}})
    merged = long.merge(stats, on=list(keys), how="left")
    out = np.full((n, 2 * TEAM_SIZE), np.nan)
    out[merged.row.to_numpy(), merged.slot.to_numpy()] = merged[value].to_numpy(float)
    return out


def hltv_perf(h: pd.DataFrame, side_a: np.ndarray, side_b: np.ndarray, stats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = stats[stats.side == "all"].drop_duplicates(["mapstatsid", "player_id"])[["mapstatsid", "player_id", "rating"]]
    s = s.astype({"mapstatsid": "float64", "player_id": "int64"})
    keys = {"mapstatsid": np.repeat(h.mapstatsid.to_numpy(float), 2 * TEAM_SIZE), "player_id": np.concatenate([side_a, side_b], axis=1).ravel()}
    return _centred(_lookup(len(h), keys, s, "rating"))


def faceit_perf(f: pd.DataFrame, stats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """HLTV-style rating without KAST and impact: 0.36 kills/round - 0.53 deaths/round + 0.0032 ADR."""
    rounds = f[["match_id", "map_name"]].assign(rounds=(f.team_a_score + f.team_b_score).to_numpy(float))
    s = stats.drop_duplicates(["match_id", "map_name", "faceit_id"]).merge(rounds, on=["match_id", "map_name"])
    s = s.assign(perf=0.36 * s.kills / s.rounds - 0.53 * s.deaths / s.rounds + 0.0032 * s.adr.fillna(ADR_MEAN))[["match_id", "map_name", "faceit_id", "perf"]]
    n = len(f)
    ids = np.array([list(a) + list(b) for a, b in zip(f.team_a_faceit_ids, f.team_b_faceit_ids)], dtype=object).reshape(n, 2 * TEAM_SIZE)
    keys = {"match_id": np.repeat(f.match_id.to_numpy(), 2 * TEAM_SIZE), "map_name": np.repeat(f.map_name.to_numpy(), 2 * TEAM_SIZE),
            "faceit_id": ids.ravel()}
    return _centred(_lookup(n, keys, s, "perf"))


def hltv_maps(history: pd.DataFrame, stats: pd.DataFrame) -> Maps:
    h = history[~showmatch_mask(history) & (history.team_a_score != history.team_b_score)]
    ok = h.team_a_player_ids.map(lambda x: x is not None and len(x) == TEAM_SIZE) \
        & h.team_b_player_ids.map(lambda x: x is not None and len(x) == TEAM_SIZE)
    h = h[ok]
    ra, n = regulation(h.team_a_score.to_numpy(), h.team_b_score.to_numpy())
    kickoff = pd.to_datetime(h.match_time.fillna(h.match_date), utc=True).dt.tz_convert(None).dt.normalize()
    day = np.maximum(pd.to_datetime(h.match_date).dt.normalize().to_numpy("datetime64[D]"), kickoff.to_numpy("datetime64[D]"))
    side_a, side_b = _fives(h.team_a_player_ids), _fives(h.team_b_player_ids)
    perf_a, perf_b = hltv_perf(h, side_a, side_b, stats)
    return Maps(day=day, source=np.zeros(len(h), np.int64), rounds_a=ra, rounds=n,
                win_a=(h.team_a_score > h.team_b_score).to_numpy(np.int64), team_a=h.team_a_id.fillna(-1).to_numpy(np.int64),
                team_b=h.team_b_id.fillna(-1).to_numpy(np.int64), side_a=side_a, side_b=side_b, perf_a=perf_a, perf_b=perf_b)


def faceit_keys(face_ids: pd.Series) -> dict[str, int]:
    """Player key of every FACEIT id seen. A linked person keeps a FACEIT key of their own: their FACEIT skill is a
    separate state that only seeds their HLTV skill at its debut."""
    only = sorted({i for ids in face_ids for i in ids})
    return {i: FACEIT_ONLY_BASE + k for k, i in enumerate(only)}


def faceit_maps(data_dir, cutoff: pd.Timestamp) -> tuple[Maps, dict[str, int]]:
    finished = load(data_dir, "faceit/faceit_matches", cutoff, ["match_id", "finished_at"]).set_index("match_id").finished_at
    ids = set(finished.index)
    cols = ["match_id", "map_name", "match_date", "team_a_faceit_ids", "team_b_faceit_ids", "team_a_score", "team_b_score",
            "team_a_players", "team_b_players", "event_name", "team_a_id", "team_b_id"]
    f = pd.read_parquet(f"{data_dir}/faceit/faceit_match_history.parquet", columns=cols)
    f = f[f.match_id.isin(ids)]
    ex = pd.read_parquet(f"{data_dir}/faceit/faceit_exclusions.parquet")
    ex = ex[ex.match_id.isin(ids)]
    f = f[~f.match_id.isin(ex[ex.map_name.isna()].match_id)]
    dropped = pd.MultiIndex.from_frame(ex[ex.map_name.notna()][["match_id", "map_name"]])
    f = f[~pd.MultiIndex.from_frame(f[["match_id", "map_name"]]).isin(dropped)]
    f = f[(f.team_a_players == TEAM_SIZE) & (f.team_b_players == TEAM_SIZE) & (f.team_a_score != f.team_b_score)]
    keys = faceit_keys(pd.concat([f.team_a_faceit_ids, f.team_b_faceit_ids]))
    side = {c: np.array([[keys[i] for i in ids_] for ids_ in f[c]], dtype=np.int64).reshape(-1, TEAM_SIZE)
            for c in ("team_a_faceit_ids", "team_b_faceit_ids")}
    names = f.event_name.drop_duplicates()
    tier = dict(zip(names, names.map(faceit_source)))
    stats = load(data_dir, "faceit/faceit_player_map_stats", cutoff, ["match_id", "map_name", "faceit_id", "kills", "deaths", "adr"])
    perf_a, perf_b = faceit_perf(f, stats)
    ra, n = regulation(f.team_a_score.to_numpy(), f.team_b_score.to_numpy())
    day = (f.match_id.map(finished).to_numpy(np.int64) // 86400).astype("datetime64[D]")
    return Maps(day=day, source=f.event_name.map(tier).map(SOURCE_INDEX).to_numpy(np.int64), rounds_a=ra, rounds=n,
                win_a=(f.team_a_score > f.team_b_score).to_numpy(np.int64), team_a=np.full(len(f), -1, np.int64), team_b=np.full(len(f), -1, np.int64),
                side_a=side["team_a_faceit_ids"], side_b=side["team_b_faceit_ids"], perf_a=perf_a, perf_b=perf_b), keys


def faceit_lineup_rows(data_dir, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Fives that FACEIT matches of an HLTV team fielded, with all five linked to HLTV ids: (team, match, time, players)."""
    cols = ["match_id", "match_date", "match_time", "team_a_id", "team_b_id", "team_a_player_ids", "team_b_player_ids"]
    f = load(data_dir, "faceit/faceit_match_history", cutoff, cols)
    time = pd.to_datetime(f.match_time.fillna(f.match_date), utc=True)
    parts = []
    for side in "ab":
        team, players = f[f"team_{side}_id"], f[f"team_{side}_player_ids"]
        ok = team.notna() & players.map(lambda p: p is not None and len(p) == TEAM_SIZE and bool(np.isfinite(np.asarray(p, float)).all()))
        parts.append(pd.DataFrame({"team": team[ok], "match": f.match_id[ok], "time": time[ok], "players": players[ok]}))
    return pd.concat(parts, ignore_index=True)


def concat_maps(parts: list[Maps]) -> Maps:
    cat = {k: np.concatenate([getattr(p, k) for p in parts]) for k in Maps.__dataclass_fields__}
    order = np.argsort(cat["day"], kind="stable")
    return Maps(**{k: v[order] for k, v in cat.items()})


def faceit_first_levels(data_dir, cutoff: pd.Timestamp, keys: dict[str, int]) -> pd.Series:
    """FACEIT skill level (1-10) of each player key at the first roster record it appears in; unrated is NaN. Taking the
    first record, not the first rated one, keeps a level from ever coming from after the player's first match."""
    r = load(data_dir, "faceit/faceit_rosters", cutoff, ["match_id", "faceit_id", "skill_level"])
    finished = load(data_dir, "faceit/faceit_matches", cutoff, ["match_id", "finished_at"])
    r = r.merge(finished, on="match_id").sort_values(["finished_at", "match_id"], kind="stable").drop_duplicates("faceit_id")
    r = r.assign(key=r.faceit_id.map(keys)).dropna(subset=["key"])
    level = r.skill_level.to_numpy(float)
    return pd.Series(np.where(level > 0, level, np.nan), index=r.key.astype(np.int64).to_numpy())
