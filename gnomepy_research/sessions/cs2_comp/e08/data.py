"""
One time-ordered table of maps from both sources, with every player on a single id scale.

HLTV maps carry the five players who played, the HLTV rating of each, and the points of the team's latest ranking.
FACEIT maps (cleaned by the bundle's exclusion list, 5v5 only) carry the same, with the rating replaced by a
kills/deaths/assists performance score and the prior feature replaced by the player's FACEIT skill level. A FACEIT
player linked to an HLTV player takes the HLTV id; an unlinked one gets an id of its own, so FACEIT opponents still
connect the rating graph.

A map is known from `time`: the match start for HLTV, the match finish for FACEIT.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

HLTV, FACEIT = 0, 1
UNLINKED_BASE = 10_000_000
FIVE = 5
COMP_CLASSES = ("esea_open", "esea_mid", "esea_main", "ecl_cct", "qualifier", "other")
# HLTV rating-style weights on kills, deaths and assists per round; ADR and KAST are missing for many FACEIT rows.
PERF_KPR, PERF_DPR, PERF_APR = 0.8644, -0.5329, 0.0996
FACEIT_LEVEL_DEFAULT = 5.0


@dataclass
class Maps:
    time: np.ndarray
    src: np.ndarray
    comp: np.ndarray
    players_a: np.ndarray
    players_b: np.ndarray
    rounds_a: np.ndarray
    rounds: np.ndarray
    dev_a: np.ndarray
    dev_b: np.ndarray
    feat_a: np.ndarray
    feat_b: np.ndarray
    team_a: np.ndarray
    team_b: np.ndarray

    def __len__(self) -> int:
        return len(self.time)


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def classify_competition(name: str | None) -> int:
    n = (name or "").lower()
    if "qualif" in n:
        return COMP_CLASSES.index("qualifier")
    if "ecl" in n or "cct" in n:
        return COMP_CLASSES.index("ecl_cct")
    if "open" in n or "entry" in n:
        return COMP_CLASSES.index("esea_open")
    if "intermediate" in n:
        return COMP_CLASSES.index("esea_mid")
    if "main" in n or "advanced" in n or "esea" in n:
        return COMP_CLASSES.index("esea_main")
    return COMP_CLASSES.index("other")


def regulation(score_a: np.ndarray, score_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regulation rounds won by A and played: an overtime map counts as 12-12 of 24."""
    total = score_a + score_b
    overtime = total > 24
    return np.where(overtime, 12, score_a), np.where(overtime, 24, total)


class RankPoints:
    """log1p of a team's HLTV ranking points at the latest ranking strictly before a day."""

    def __init__(self, rankings: pd.DataFrame):
        r = rankings[["date", "team_id", "points"]].dropna().copy()
        r["date"] = _naive(r["date"]).astype("datetime64[ns]")
        self.r = r.sort_values("date")

    def lookup(self, team_ids: np.ndarray, days: np.ndarray) -> np.ndarray:
        q = pd.DataFrame({"team_id": np.asarray(team_ids, dtype=float), "date": pd.to_datetime(days).astype("datetime64[ns]")})
        q["i"] = np.arange(len(q))
        q = q.dropna(subset=["team_id"]).sort_values("date")
        m = pd.merge_asof(q, self.r.assign(team_id=self.r.team_id.astype(float)), on="date", by="team_id",
                          allow_exact_matches=False)
        out = np.zeros(len(team_ids))
        out[m.i.to_numpy()] = np.log1p(m.points.fillna(0.0).to_numpy())
        return out


def _five(ids) -> list[int] | None:
    if ids is None:
        return None
    ids = [int(p) for p in ids if p == p]
    return ids if len(ids) == FIVE and len(set(ids)) == FIVE else None


def hltv_maps(data_dir: Path, before: pd.Timestamp, ranks: RankPoints) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", before)
    h = h[(h.team_a_score != h.team_b_score) & ~showmatch_mask(h)].copy()
    h["pa"] = [_five(x) for x in h.team_a_player_ids]
    h["pb"] = [_five(x) for x in h.team_b_player_ids]
    h = h[h.pa.notna() & h.pb.notna()].copy()
    h["time"] = _naive(h.match_time.fillna(h.match_date))
    ra, n = regulation(h.team_a_score.to_numpy(), h.team_b_score.to_numpy())
    h["rounds_a"], h["rounds"] = ra, n
    day = h.time.dt.normalize().to_numpy()
    h["fa"] = ranks.lookup(h.team_a_id.to_numpy(), day)
    h["fb"] = ranks.lookup(h.team_b_id.to_numpy(), day)
    stats = load(data_dir, "hltv/cs2_player_map_stats", before,
                 ["match_id", "match_date", "map_name", "player_id", "team_id", "side", "rating"])
    stats = stats[(stats.side == "all") & stats.rating.notna()]
    stats = stats.assign(dev=stats.rating - stats.groupby(["match_id", "map_name", "team_id"]).rating.transform("mean"))
    dev = {(m, mp, int(p)): d for m, mp, p, d in zip(stats.match_id, stats.map_name, stats.player_id, stats.dev)}
    h["da"] = [[dev.get((m, mp, p), np.nan) for p in five] for m, mp, five in zip(h.match_id, h.map_name, h.pa)]
    h["db"] = [[dev.get((m, mp, p), np.nan) for p in five] for m, mp, five in zip(h.match_id, h.map_name, h.pb)]
    return h.assign(src=HLTV, comp=len(COMP_CLASSES), ta=h.team_a_id, tb=h.team_b_id)[
        ["time", "src", "comp", "pa", "pb", "rounds_a", "rounds", "da", "db", "fa", "fb", "ta", "tb"]]


def faceit_maps(data_dir: Path, before: pd.Timestamp) -> pd.DataFrame:
    fmh = load(data_dir, "faceit/faceit_match_history", before)
    fma = load(data_dir, "faceit/faceit_matches", before, ["match_id", "finished_at", "competition_name"])
    excl = load(data_dir, "faceit/faceit_exclusions", before)
    whole = set(excl[excl.map_name.isna()].match_id)
    partial = set(zip(excl[excl.map_name.notna()].match_id, excl[excl.map_name.notna()].map_name))
    fmh = fmh[fmh.match_id.isin(fma.match_id) & ~fmh.match_id.isin(whole)]
    fmh = fmh[[(m, n) not in partial for m, n in zip(fmh.match_id, fmh.map_name)]]
    fmh = fmh[(fmh.team_a_score != fmh.team_b_score) & (fmh.team_a_faceit_ids.map(len) == FIVE) & (fmh.team_b_faceit_ids.map(len) == FIVE)]
    fmh = fmh.merge(fma.assign(comp=fma.competition_name.map(classify_competition)), on="match_id")
    fmh["time"] = pd.to_datetime(fmh.finished_at, unit="s")

    link = load(data_dir, "faceit/hltv_faceit_map", None)
    to_key = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))
    unlinked = sorted({f for col in ("team_a_faceit_ids", "team_b_faceit_ids") for ids in fmh[col] for f in ids} - set(to_key))
    to_key.update({f: UNLINKED_BASE + i for i, f in enumerate(unlinked)})
    fmh["pa"] = [[to_key[f] for f in ids] for ids in fmh.team_a_faceit_ids]
    fmh["pb"] = [[to_key[f] for f in ids] for ids in fmh.team_b_faceit_ids]
    fmh = fmh[[len(set(a + b)) == 2 * FIVE for a, b in zip(fmh.pa, fmh.pb)]].copy()
    ra, n = regulation(fmh.team_a_score.to_numpy(), fmh.team_b_score.to_numpy())
    fmh["rounds_a"], fmh["rounds"] = ra, n
    fmh["total"] = fmh.team_a_score + fmh.team_b_score

    st = load(data_dir, "faceit/faceit_player_map_stats", before,
              ["match_id", "map_name", "faceit_id", "kills", "deaths", "assists"])
    st = st.merge(fmh[["match_id", "map_name", "total"]], on=["match_id", "map_name"])
    perf = (PERF_KPR * st.kills + PERF_DPR * st.deaths + PERF_APR * st.assists) / st.total
    st["perf"] = perf
    perf_by = {(m, mp, f): v for m, mp, f, v in zip(st.match_id, st.map_name, st.faceit_id, st.perf)}

    def dev_row(m, mp, ids):
        v = np.array([perf_by.get((m, mp, f), np.nan) for f in ids])
        return (v - v.mean()).tolist() if not np.isnan(v).any() else [np.nan] * FIVE

    fmh["da"] = [dev_row(m, mp, ids) for m, mp, ids in zip(fmh.match_id, fmh.map_name, fmh.team_a_faceit_ids)]
    fmh["db"] = [dev_row(m, mp, ids) for m, mp, ids in zip(fmh.match_id, fmh.map_name, fmh.team_b_faceit_ids)]

    ro = load(data_dir, "faceit/faceit_rosters", before, ["match_id", "faceit_id", "skill_level"])
    ro = ro[ro.match_id.isin(fmh.match_id)]
    level = {(m, f): s for m, f, s in zip(ro.match_id, ro.faceit_id, ro.skill_level) if s == s and s > 0}
    fmh["fa"] = [[level.get((m, f), FACEIT_LEVEL_DEFAULT) for f in ids] for m, ids in zip(fmh.match_id, fmh.team_a_faceit_ids)]
    fmh["fb"] = [[level.get((m, f), FACEIT_LEVEL_DEFAULT) for f in ids] for m, ids in zip(fmh.match_id, fmh.team_b_faceit_ids)]
    return fmh.assign(src=FACEIT, ta=-1, tb=-1)[["time", "src", "comp", "pa", "pb", "rounds_a", "rounds", "da", "db", "fa", "fb", "ta", "tb"]]


def build_maps(data_dir: str | Path, before: pd.Timestamp, use_faceit: bool = True) -> Maps:
    data_dir = Path(data_dir)
    ranks = RankPoints(load(data_dir, "hltv/cs2_team_rankings", before))
    parts = [hltv_maps(data_dir, before, ranks)]
    if use_faceit:
        parts.append(faceit_maps(data_dir, before))
    m = pd.concat(parts, ignore_index=True).sort_values(["time", "src"], kind="stable").reset_index(drop=True)

    def five_feature(col) -> np.ndarray:
        return np.array([x if isinstance(x, list) else [x] * FIVE for x in m[col]], dtype=float)

    return Maps(
        time=m.time.to_numpy("datetime64[ns]"), src=m.src.to_numpy(np.int8), comp=m.comp.to_numpy(np.int8),
        players_a=np.array(m.pa.tolist(), dtype=np.int64), players_b=np.array(m.pb.tolist(), dtype=np.int64),
        rounds_a=m.rounds_a.to_numpy(np.int32), rounds=m.rounds.to_numpy(np.int32),
        dev_a=np.array(m.da.tolist(), dtype=float), dev_b=np.array(m.db.tolist(), dtype=float),
        feat_a=five_feature("fa"), feat_b=five_feature("fb"),
        team_a=m.ta.to_numpy(np.int64), team_b=m.tb.to_numpy(np.int64),
    )
