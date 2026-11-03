"""Point-in-time map rows for g07: HLTV and FACEIT maps on one player id space, ordered by when each became known.

Every table is read through `bundle.load` at the run's latest cutoff. A map counts as known at max(match_time,
match_date) on HLTV and at the series' finished_at on FACEIT, so a filter that consumes maps in `known` order and stops
at a cutoff sees exactly what `bundle.load(..., before=cutoff)` would return.

A FACEIT player linked to HLTV takes the HLTV id; an unlinked one gets a negative hash of the FACEIT id. FACEIT series
that are also on HLTV (`faceit_hltv_series_link`) are dropped so each series counts once."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

FIVE = 5
_DIVISION_RE = re.compile(r"(entry|open10|open|intermediate|main|advanced|premier|challenger|pro league)", re.I)
DIVISIONS = ("entry", "open", "intermediate", "main", "advanced", "pro", "other")


def faceit_uid(faceit_id: str) -> int:
    return -int(hashlib.blake2b(faceit_id.encode(), digest_size=7).hexdigest(), 16)


def division(name: str | None) -> str:
    m = _DIVISION_RE.search(name or "")
    if m is None:
        return "other"
    d = m.group(1).lower()
    if d == "open10":
        return "open"
    if d in ("premier", "challenger", "pro league"):
        return "pro"
    return d


def naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _regulation(score_a: np.ndarray, score_b: np.ndarray, overtime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An overtime map counts as 12-12 of 24 regulation rounds: the overtime itself is close to a coin flip."""
    ot = overtime | (score_a + score_b > 24)
    return np.where(ot, 12, score_a), np.where(ot, 24, score_a + score_b)


def _fives(col: pd.Series, mapper=None) -> tuple[list, np.ndarray]:
    out, ok = [], np.zeros(len(col), dtype=bool)
    for i, v in enumerate(col):
        five = None
        if v is not None and len(v) == FIVE:
            try:
                ids = tuple(mapper(x) if mapper else int(x) for x in v)
            except (TypeError, ValueError):
                ids = ()
            if len(set(ids)) == FIVE:
                five, ok[i] = ids, True
        out.append(five)
    return out, ok


class Inputs:
    """Every map and side table g07 reads, loaded once up to `until`."""

    def __init__(self, data_dir: str | Path, until: pd.Timestamp, faceit: bool = True):
        self.data_dir = Path(data_dir)
        self.until = pd.Timestamp(until)
        parts = [self._hltv()]
        if faceit:
            parts.append(self._faceit())
        m = pd.concat(parts, ignore_index=True)
        m["pos"] = m.pos.fillna(1).astype(int)
        self.maps = m.sort_values(["known", "time", "match_id", "pos"], kind="stable").reset_index(drop=True)
        rk = load(self.data_dir, "hltv/cs2_team_rankings", self.until, ["date", "team_id", "rank"])
        rk = rk.assign(date=naive(rk["date"])).sort_values("date")
        self.rankings = {int(t): (g.date.to_numpy(), g["rank"].to_numpy(float)) for t, g in rk.groupby("team_id")}

    def _hltv(self) -> pd.DataFrame:
        h = load(self.data_dir, "hltv/cs2_match_history", self.until)
        h = h[~showmatch_mask(h)].copy()
        h["time"] = naive(h.match_time.fillna(h.match_date))
        h["known"] = np.maximum(h.time, naive(h.match_date))
        self.series = self._series(h)
        self.hltv_known = h.groupby("match_id").known.max().to_dict()
        h = h[(h.team_a_score != h.team_b_score) & h.team_a_id.notna() & h.team_b_id.notna()]
        fa, oka = _fives(h.team_a_player_ids)
        fb, okb = _fives(h.team_b_player_ids)
        ok = oka & okb
        h = h[ok]
        fa = [f for f, k in zip(fa, ok) if k]
        fb = [f for f, k in zip(fb, ok) if k]
        ra, n = _regulation(h.team_a_score.to_numpy(int), h.team_b_score.to_numpy(int), np.zeros(len(h), bool))
        st = load(self.data_dir, "hltv/cs2_player_map_stats", self.until, ["match_id", "match_date", "map_name", "player_id", "side", "rating"])
        st = st[(st.side == "all") & st.rating.notna() & st.player_id.notna()]
        rating = dict(zip(zip(st.match_id.astype(np.int64), st.map_name, st.player_id.astype(np.int64)), st.rating.astype(float)))

        def ratings(fives):
            out = []
            for mid, mp, five in zip(h.match_id.astype(np.int64), h.map_name, fives):
                r = tuple(rating.get((mid, mp, p)) for p in five)
                out.append(None if any(x is None for x in r) else r)
            return out

        return pd.DataFrame({
            "match_id": h.match_id.astype(str).to_numpy(), "pos": h.map_position_in_series.to_numpy(),
            "time": h.time.to_numpy(), "known": h.known.to_numpy(),
            "team_a": h.team_a_id.astype(np.int64).to_numpy(), "team_b": h.team_b_id.astype(np.int64).to_numpy(),
            "five_a": fa, "five_b": fb, "rating_a": ratings(fa), "rating_b": ratings(fb), "rounds": n, "rounds_a": ra,
            "src": 0, "div": "hltv",
            "lan": h.is_lan.fillna(0).to_numpy(int), "tier": h.event_tier.fillna(0).to_numpy(int)})

    @staticmethod
    def _series(h: pd.DataFrame) -> pd.DataFrame:
        """HLTV series with an outcome and announced fives: walk-forward targets for development."""
        d = h[h.team_a_id.notna() & h.team_b_id.notna()]
        g = d.groupby("match_id")
        s = g.agg(time=("time", "min"), known=("known", "max"), team_a=("team_a_id", "first"), team_b=("team_b_id", "first"),
                  bo_type=("bo_type", "first"), is_lan=("is_lan", "first"), tier=("event_tier", "first"),
                  wins=("team_a_won", "sum"), maps=("team_a_won", "size"),
                  lineup_a=("team_a_lineup_ids", "first"), lineup_b=("team_b_lineup_ids", "first"),
                  rank_a=("team_a_rank", "first"), rank_b=("team_b_rank", "first"))
        s = s[(s.wins * 2 != s.maps)].reset_index()
        s["y"] = (s.wins * 2 > s.maps).astype(int)
        return s.astype({"team_a": np.int64, "team_b": np.int64})

    def _faceit(self) -> pd.DataFrame:
        d = self.data_dir
        link = load(d, "faceit/hltv_faceit_map")
        self.f2h = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))
        fm = load(d, "faceit/faceit_matches", self.until,
                  ["match_id", "finished_at", "has_stats", "competition_name", "faction1_hltv_team", "faction2_hltv_team"])
        fm["known"] = pd.to_datetime(fm.finished_at, unit="s")
        # The FACEIT copy of a series is dropped once its HLTV twin was known before the FACEIT copy finished, a rule
        # that reads the same at any cutoff because the link only exists once both copies are known.
        link = load(d, "faceit/faceit_hltv_series_link", self.until, ["faceit_match_id", "hltv_match_id"])
        twin = dict(zip(link.faceit_match_id, link.hltv_match_id))
        dup = [m in twin and self.hltv_known.get(twin[m], pd.Timestamp.max) < k for m, k in zip(fm.match_id, fm.known)]
        fm = fm[fm.has_stats & ~np.array(dup, dtype=bool)]
        ex = load(d, "faceit/faceit_exclusions", self.until)
        whole = set(ex.match_id[ex.map_name.isna()])
        bad = set(zip(ex.match_id[ex.map_name.notna()], ex.map_name[ex.map_name.notna()]))
        cols = ["match_id", "map_name", "map_position_in_series", "match_time", "match_date", "team_a_score", "team_b_score",
                "team_a_ot_score", "team_b_ot_score", "team_a_faceit_ids", "team_b_faceit_ids", "team_a_players",
                "team_b_players"]
        fh = load(d, "faceit/faceit_match_history", self.until, cols)
        fh = fh.merge(fm[["match_id", "known", "competition_name", "faction1_hltv_team", "faction2_hltv_team"]], on="match_id")
        fh = fh[~fh.match_id.isin(whole)]
        fh = fh[[(m, n) not in bad for m, n in zip(fh.match_id, fh.map_name)]]
        fh = fh[(fh.team_a_players == FIVE) & (fh.team_b_players == FIVE) & (fh.team_a_score != fh.team_b_score)]
        uid = self.uid
        fa, oka = _fives(fh.team_a_faceit_ids, uid)
        fb, okb = _fives(fh.team_b_faceit_ids, uid)
        ok = oka & okb
        fh = fh[ok]
        fa = [f for f, k in zip(fa, ok) if k]
        fb = [f for f, k in zip(fb, ok) if k]
        ot = (fh.team_a_ot_score.fillna(0) + fh.team_b_ot_score.fillna(0) > 0).to_numpy()
        ra, n = _regulation(fh.team_a_score.to_numpy(int), fh.team_b_score.to_numpy(int), ot)
        return pd.DataFrame({
            "match_id": fh.match_id.to_numpy(), "pos": fh.map_position_in_series.to_numpy(),
            "time": naive(fh.match_time.fillna(fh.match_date)).to_numpy(), "known": fh.known.to_numpy(),
            "team_a": fh.faction1_hltv_team.fillna(-1).astype(np.int64).to_numpy(),
            "team_b": fh.faction2_hltv_team.fillna(-1).astype(np.int64).to_numpy(),
            "five_a": fa, "five_b": fb, "rating_a": None, "rating_b": None, "rounds": n, "rounds_a": ra, "src": 1,
            "div": [division(c) for c in fh.competition_name], "lan": 0, "tier": -1})

    def uid(self, faceit_id: str) -> int:
        h = self.f2h.get(faceit_id)
        return int(h) if h is not None else faceit_uid(faceit_id)

    def rank(self, team: int, day: pd.Timestamp, max_age_days: int = 120) -> float:
        h = self.rankings.get(int(team))
        if h is None:
            return np.nan
        i = np.searchsorted(h[0], np.datetime64(day), side="left") - 1
        if i < 0 or (np.datetime64(day) - h[0][i]) > np.timedelta64(max_age_days, "D"):
            return np.nan
        return float(h[1][i])
