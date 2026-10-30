"""
Point-in-time inputs for e12: HLTV and FACEIT maps on one player id space, plus what each team's five looked like
going into a cutoff.

Everything is read once through `bundle.load` at the latest cutoff of a run, and each fit then keeps the rows that
were known before its own cutoff by the same rules (HLTV by match date and time, FACEIT by the series' finish time),
so a fit never depends on which other weeks are in the run.

A FACEIT player linked to HLTV (`hltv_faceit_map`) takes the HLTV id; an unlinked one keeps its own id (a negative
hash of the FACEIT id), so FACEIT-only opponents still connect the rating graph.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

FIVE = 5
UNRANKED = 300.0
DIVISIONS = ("entry", "open", "intermediate", "main", "advanced", "pro", "qualifier", "other")
_DIVISION_RE = re.compile(r"(entry|open10|open|intermediate|main|advanced|premier|challenger|pro league|qualifier)", re.I)


def faceit_uid(faceit_id: str) -> int:
    return -int(hashlib.blake2b(faceit_id.encode(), digest_size=7).hexdigest(), 16)


def division(competition_name: str | None) -> str:
    m = _DIVISION_RE.search(competition_name or "")
    if m is None:
        return "other"
    d = m.group(1).lower()
    if d == "open10":
        return "open"
    if d in ("premier", "challenger", "pro league"):
        return "pro"
    return d


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _regulation(score_a: np.ndarray, score_b: np.ndarray, overtime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An overtime map counts as 12-12 of 24 regulation rounds."""
    ot = overtime | (score_a + score_b > 24)
    return np.where(ot, 12, score_a), np.where(ot, 24, score_a + score_b)


def _fives(col: pd.Series, mapper=None) -> np.ndarray:
    out = np.zeros((len(col), FIVE), dtype=np.int64)
    ok = np.zeros(len(col), dtype=bool)
    for i, v in enumerate(col):
        if v is None or len(v) != FIVE:
            continue
        ids = [mapper(x) for x in v] if mapper else [int(x) for x in v]
        if len(set(ids)) == FIVE:
            out[i] = ids
            ok[i] = True
    return out, ok


@dataclass
class Maps:
    """Decisive maps, one row each, sorted by time: who played, regulation rounds, and when the row became known."""

    frame: pd.DataFrame     # match_id, time, known, team_a, team_b, rank_a, rank_b, rounds, rounds_a, source, division
    five_a: np.ndarray      # (N, 5) unified player ids
    five_b: np.ndarray

    def before(self, cutoff: pd.Timestamp, start: pd.Timestamp | None = None) -> Maps:
        keep = (self.frame.known < cutoff).to_numpy()
        if start is not None:
            keep &= (self.frame.time >= start).to_numpy()
        return Maps(self.frame[keep].reset_index(drop=True), self.five_a[keep], self.five_b[keep])

    @staticmethod
    def concat(parts: list[Maps]) -> Maps:
        frame = pd.concat([p.frame for p in parts], ignore_index=True)
        order = np.lexsort((frame.match_id.astype(str).to_numpy(), frame.time.to_numpy()))
        return Maps(frame.iloc[order].reset_index(drop=True), np.concatenate([p.five_a for p in parts])[order],
                    np.concatenate([p.five_b for p in parts])[order])


class Inputs:
    """Every table e12 reads, loaded once up to `until` (the run's latest cutoff)."""

    def __init__(self, data_dir: str | Path, until: pd.Timestamp, faceit: bool):
        self.data_dir = Path(data_dir)
        self.until = pd.Timestamp(until)
        self.faceit = faceit
        self._load_hltv()
        if faceit:
            self._load_faceit()

    def _load_hltv(self) -> None:
        h = load(self.data_dir, "hltv/cs2_match_history", self.until)
        h = h[~showmatch_mask(h)].copy()
        h["time"] = _naive(h.match_time.fillna(h.match_date))
        h["known"] = np.maximum(h.time, _naive(h.match_date))
        self.series = self._series(h)
        h = h[(h.team_a_score != h.team_b_score) & h.team_a_id.notna() & h.team_b_id.notna()]
        five_a, ok_a = _fives(h.team_a_player_ids)
        five_b, ok_b = _fives(h.team_b_player_ids)
        ok = ok_a & ok_b
        h, five_a, five_b = h[ok], five_a[ok], five_b[ok]
        ra, n = _regulation(h.team_a_score.to_numpy(int), h.team_b_score.to_numpy(int), np.zeros(len(h), bool))
        frame = pd.DataFrame({
            "match_id": h.match_id.astype(str).to_numpy(), "time": h.time.to_numpy(), "known": h.known.to_numpy(),
            "team_a": h.team_a_id.astype(np.int64).to_numpy(), "team_b": h.team_b_id.astype(np.int64).to_numpy(),
            "rank_a": h.team_a_rank.fillna(UNRANKED).to_numpy(float), "rank_b": h.team_b_rank.fillna(UNRANKED).to_numpy(float),
            "rounds": n, "rounds_a": ra, "source": "hltv", "division": "hltv", "is_lan": h.is_lan.fillna(0).to_numpy(int)})
        self.hltv = Maps.concat([Maps(frame, five_a, five_b)])
        lu = load(self.data_dir, "hltv/cs2_match_lineups", self.until, ["match_id", "match_date", "team_id", "is_standin"])
        lu = lu.assign(known=_naive(lu.match_date)).dropna(subset=["team_id"])
        self.standins = lu.groupby(["match_id", "team_id"]).agg(known=("known", "max"), standins=("is_standin", "sum")).reset_index()
        rk = load(self.data_dir, "hltv/cs2_team_rankings", self.until, ["date", "team_id", "rank"])
        rk = rk.assign(date=_naive(rk["date"])).sort_values("date")
        self.rankings = {int(t): (g.date.to_numpy(), g["rank"].to_numpy(float)) for t, g in rk.groupby("team_id")}

    @staticmethod
    def _series(h: pd.DataFrame) -> pd.DataFrame:
        """HLTV series with an outcome: the walk-forward targets for development."""
        d = h[h.team_a_id.notna() & h.team_b_id.notna()]
        g = d.groupby("match_id")
        s = g.agg(time=("time", "min"), known=("known", "max"), team_a=("team_a_id", "first"), team_b=("team_b_id", "first"),
                  bo_type=("bo_type", "first"), is_lan=("is_lan", "first"), wins=("team_a_won", "sum"), maps=("team_a_won", "size"))
        s = s[(s.wins * 2 != s.maps)].reset_index()
        s["y"] = (s.wins * 2 > s.maps).astype(int)
        return s.astype({"team_a": np.int64, "team_b": np.int64})

    def _load_faceit(self) -> None:
        d = self.data_dir
        link = load(d, "faceit/hltv_faceit_map")
        self.f2h = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))
        self.h2f = {h: f for f, h in self.f2h.items()}
        self.faceit_of: dict[int, str] = {}
        fm = load(d, "faceit/faceit_matches", self.until,
                  ["match_id", "finished_at", "started_at", "has_stats", "competition_name", "faction1_hltv_team", "faction2_hltv_team"])
        fm["known"] = pd.to_datetime(fm.finished_at, unit="s")
        self.faceit_finish = dict(zip(fm.match_id, fm.known))
        ex = load(d, "faceit/faceit_exclusions", self.until)
        whole = set(ex.match_id[ex.map_name.isna()])
        bad_maps = set(zip(ex.match_id[ex.map_name.notna()], ex.map_name[ex.map_name.notna()]))
        cols = ["match_id", "map_name", "match_time", "match_date", "team_a_score", "team_b_score", "team_a_ot_score",
                "team_b_ot_score", "team_a_faceit_ids", "team_b_faceit_ids", "team_a_players", "team_b_players",
                "team_a_faceit_id", "team_b_faceit_id"]
        fh = load(d, "faceit/faceit_match_history", self.until, cols)
        fh = fh.merge(fm[fm.has_stats][["match_id", "known", "competition_name", "faction1_hltv_team", "faction2_hltv_team"]],
                      on="match_id")
        fh = fh[~fh.match_id.isin(whole)]
        fh = fh[[(m, n) not in bad_maps for m, n in zip(fh.match_id, fh.map_name)]]
        fh = fh[(fh.team_a_players == FIVE) & (fh.team_b_players == FIVE) & (fh.team_a_score != fh.team_b_score)]
        uid = self.uid
        five_a, ok_a = _fives(fh.team_a_faceit_ids, uid)
        five_b, ok_b = _fives(fh.team_b_faceit_ids, uid)
        ok = ok_a & ok_b
        fh, five_a, five_b = fh[ok], five_a[ok], five_b[ok]
        ot = (fh.team_a_ot_score.fillna(0) + fh.team_b_ot_score.fillna(0) > 0).to_numpy()
        ra, n = _regulation(fh.team_a_score.to_numpy(int), fh.team_b_score.to_numpy(int), ot)
        frame = pd.DataFrame({
            "match_id": fh.match_id.to_numpy(), "time": _naive(fh.match_time.fillna(fh.match_date)).to_numpy(),
            "known": fh.known.to_numpy(),
            "team_a": fh.faction1_hltv_team.fillna(-1).astype(np.int64).to_numpy(),
            "team_b": fh.faction2_hltv_team.fillna(-1).astype(np.int64).to_numpy(),
            "rank_a": UNRANKED, "rank_b": UNRANKED, "rounds": n, "rounds_a": ra, "source": "faceit",
            "division": [division(c) for c in fh.competition_name], "is_lan": 0})
        self.faceit_maps = Maps.concat([Maps(frame, five_a, five_b)])
        ro = load(d, "faceit/faceit_rosters", self.until, ["match_id", "faceit_id", "played", "skill_level"])
        ro = ro[ro.played & ro.skill_level.notna()]
        known = ro.match_id.map(self.faceit_finish)
        self.levels = pd.DataFrame({"uid": [uid(f) for f in ro.faceit_id], "known": known.to_numpy(),
                                    "level": ro.skill_level.to_numpy(float)}).dropna().sort_values("known").reset_index(drop=True)

    def uid(self, faceit_id: str) -> int:
        h = self.f2h.get(faceit_id)
        if h is not None:
            return int(h)
        u = faceit_uid(faceit_id)
        self.faceit_of[u] = faceit_id
        return u

    def faceit_ids(self, five: tuple[int, ...] | None) -> list[str]:
        """The FACEIT ids of a five's players that have one."""
        out = []
        for p in five or ():
            f = self.h2f.get(p) if p > 0 else self.faceit_of.get(p)
            if f is not None:
                out.append(f)
        return out

    def maps(self, cutoff: pd.Timestamp, start: pd.Timestamp, faceit: bool) -> Maps:
        parts = [self.hltv.before(cutoff, start)]
        if faceit:
            parts.append(self.faceit_maps.before(cutoff, start))
        return Maps.concat(parts)

    def rank(self, team: int, day: pd.Timestamp, max_age_days: int = 120) -> float:
        h = self.rankings.get(int(team))
        if h is None:
            return UNRANKED
        i = np.searchsorted(h[0], np.datetime64(day), side="left") - 1
        if i < 0 or (np.datetime64(day) - h[0][i]) > np.timedelta64(max_age_days, "D"):
            return UNRANKED
        return float(h[1][i])
