"""
Point-in-time inputs for g16: HLTV and FACEIT decisive maps on one player id space in the order they became known,
series with their announced fives and outcomes, and the ranking table the skill filter uses for newcomers.

Everything is read once through `bundle.load` at the latest cutoff of a run. The weekly pass in `skill.py` then only
consumes maps known before each cutoff, so a week's prices never depend on which other weeks are in the run. An HLTV
map is known when it is played; a FACEIT map when FACEIT finalised its series (finished_at), which the bundle cuts
by, and which can be weeks later for some ESEA maps.

A FACEIT player linked to HLTV takes the HLTV id; an unlinked one keeps a negative hash of its FACEIT id, so
FACEIT-only opponents still connect the skill graph.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

FIVE = 5
NO_TEAM = -1
_DIVISION_RE = re.compile(r"(entry|open10|open|intermediate|main|advanced|premier|challenger|pro league|qualifier)", re.I)


def naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def five(v, mapper=None) -> tuple[int, ...] | None:
    if v is None or isinstance(v, float) or len(v) != FIVE:
        return None
    ids = tuple(mapper(x) for x in v) if mapper else tuple(int(x) for x in v)
    return ids if len(set(ids)) == FIVE else None


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


def week_cutoff(times: pd.Series) -> pd.Series:
    """Monday 00:00 UTC of the week each time falls in."""
    t = naive(times)
    return t.dt.normalize() - pd.to_timedelta(t.dt.weekday, unit="D")


def regulation(score_a: np.ndarray, score_b: np.ndarray, overtime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An overtime map says the regulation rounds were level: it counts as 12-12."""
    ot = overtime | (score_a + score_b > 24)
    return np.where(ot, 12, score_a), np.where(ot, 12, score_b)


MAP_COLUMNS = ["match_id", "time", "day", "known", "team_a", "team_b", "five_a", "five_b", "rounds_a", "rounds_b", "pos", "source",
               "division", "rating_a", "rating_b", "won_a", "map_name", "level_a", "level_b"]


class Inputs:
    def __init__(self, data_dir: str | Path, until: pd.Timestamp, faceit: bool = True):
        self.data_dir = Path(data_dir)
        self.until = pd.Timestamp(until)
        self.faceit = faceit
        h = load(self.data_dir, "hltv/cs2_match_history", self.until)
        h = h[~showmatch_mask(h) & h.team_a_id.notna() & h.team_b_id.notna()].copy()
        h["time"] = naive(h.match_time).fillna(naive(h.match_date))
        h["day"] = naive(h.match_date)
        parts = [self._hltv_maps(h, self._ratings())]
        if faceit:
            parts.append(self._faceit_maps())
        maps = pd.concat(parts, ignore_index=True)
        self.maps = maps.sort_values(["known", "match_id", "pos"], kind="stable").reset_index(drop=True)
        self.series = self._series(h)
        rk = load(self.data_dir, "hltv/cs2_team_rankings", self.until, ["date", "team_id", "points"])
        self.rankings = rk.assign(date=naive(rk["date"]))
        self.levels = self._levels() if faceit else {}

    def _ratings(self) -> dict:
        st = load(self.data_dir, "hltv/cs2_player_map_stats", self.until, ["match_id", "match_date", "map_name", "player_id", "side", "rating"])
        st = st[(st.side == "all") & st.rating.notna() & st.player_id.notna()]
        return dict(zip(zip(st.match_id.astype(np.int64), st.map_name, st.player_id.astype(np.int64)), st.rating.astype(float)))

    @staticmethod
    def _hltv_maps(h: pd.DataFrame, ratings: dict) -> pd.DataFrame:
        d = h[h.team_a_score != h.team_b_score].copy()
        d["five_a"] = d.team_a_player_ids.map(five)
        d["five_b"] = d.team_b_player_ids.map(five)
        d = d[d.five_a.notna() & d.five_b.notna()]
        ra, rb = regulation(d.team_a_score.to_numpy(int), d.team_b_score.to_numpy(int), np.zeros(len(d), bool))

        def rated(fives, mids, names):
            out = []
            for f, m, n in zip(fives, mids, names):
                r = [ratings.get((int(m), n, p)) for p in f]
                out.append(None if any(v is None for v in r) else np.array(r, float))
            return out

        return pd.DataFrame({"match_id": d.match_id.astype(str).to_numpy(), "time": d.time.to_numpy(), "day": d.day.to_numpy(),
                             "known": d.time.to_numpy(),
                             "team_a": d.team_a_id.astype(np.int64).to_numpy(), "team_b": d.team_b_id.astype(np.int64).to_numpy(),
                             "five_a": d.five_a.to_numpy(), "five_b": d.five_b.to_numpy(), "rounds_a": ra, "rounds_b": rb,
                             "pos": d.map_position_in_series.fillna(0).to_numpy(int), "source": "hltv", "division": "hltv",
                             "rating_a": rated(d.five_a, d.match_id, d.map_name), "rating_b": rated(d.five_b, d.match_id, d.map_name),
                             "won_a": (d.team_a_score > d.team_b_score).to_numpy(float), "map_name": d.map_name.to_numpy(),
                             "level_a": None, "level_b": None})[MAP_COLUMNS]

    def _uid(self):
        link = load(self.data_dir, "faceit/hltv_faceit_map")
        f2h = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))

        def uid(faceit_id: str) -> int:
            h = f2h.get(faceit_id)
            return int(h) if h is not None else faceit_uid(faceit_id)

        return uid

    def _levels(self) -> dict:
        """Per unified player id: FACEIT skill levels in the order they became known (finished_at of the match)."""
        d = self.data_dir
        fm = load(d, "faceit/faceit_matches", self.until, ["match_id", "finished_at"])
        known = dict(zip(fm.match_id, pd.to_datetime(fm.finished_at, unit="s")))
        ro = load(d, "faceit/faceit_rosters", self.until, ["match_id", "faceit_id", "played", "skill_level"])
        ro = ro[ro.played & ro.skill_level.notna()]
        uid = self._uid()
        lv = pd.DataFrame({"uid": [uid(f) for f in ro.faceit_id], "known": ro.match_id.map(known).to_numpy(),
                           "level": ro.skill_level.to_numpy(float)}).dropna().sort_values(["known", "uid"], kind="stable")
        return {int(u): (g.known.to_numpy(), g.level.to_numpy()) for u, g in lv.groupby("uid", sort=False)}

    def _faceit_maps(self) -> pd.DataFrame:
        d = self.data_dir
        uid = self._uid()

        fm = load(d, "faceit/faceit_matches", self.until,
                  ["match_id", "finished_at", "has_stats", "competition_name", "faction1_hltv_team", "faction2_hltv_team"])
        fm = fm[fm.has_stats].copy()
        fm["known"] = pd.to_datetime(fm.finished_at, unit="s")
        ex = load(d, "faceit/faceit_exclusions", self.until)
        whole = set(ex.match_id[ex.map_name.isna()])
        bad_maps = set(zip(ex.match_id[ex.map_name.notna()], ex.map_name[ex.map_name.notna()]))
        cols = ["match_id", "map_name", "match_time", "match_date", "map_position_in_series", "team_a_score", "team_b_score",
                "team_a_ot_score", "team_b_ot_score", "team_a_faceit_ids", "team_b_faceit_ids", "team_a_players", "team_b_players"]
        fh = load(d, "faceit/faceit_match_history", self.until, cols)
        fh = fh.merge(fm[["match_id", "known", "competition_name", "faction1_hltv_team", "faction2_hltv_team"]], on="match_id")
        fh = fh[~fh.match_id.isin(whole)]
        fh = fh[[(m, n) not in bad_maps for m, n in zip(fh.match_id, fh.map_name)]]
        fh = fh[(fh.team_a_players == FIVE) & (fh.team_b_players == FIVE) & (fh.team_a_score != fh.team_b_score)].copy()
        fh["five_a"] = fh.team_a_faceit_ids.map(lambda v: five(v, uid))
        fh["five_b"] = fh.team_b_faceit_ids.map(lambda v: five(v, uid))
        fh = fh[fh.five_a.notna() & fh.five_b.notna()]
        ro = load(d, "faceit/faceit_rosters", self.until, ["match_id", "faceit_id", "played", "skill_level"])
        ro = ro[ro.played & ro.skill_level.notna()]
        level_of = dict(zip(zip(ro.match_id, ro.faceit_id), ro.skill_level.astype(float)))

        def levels(mids, ids):
            return [np.array([level_of.get((m, f), np.nan) for f in v], float) for m, v in zip(mids, ids)]

        ot = (fh.team_a_ot_score.fillna(0) + fh.team_b_ot_score.fillna(0) > 0).to_numpy()
        ra, rb = regulation(fh.team_a_score.to_numpy(int), fh.team_b_score.to_numpy(int), ot)
        time = naive(fh.match_time).fillna(naive(fh.match_date))
        return pd.DataFrame({"match_id": fh.match_id.to_numpy(), "time": time.to_numpy(), "day": naive(fh.match_date).to_numpy(),
                             "known": np.maximum(fh.known.to_numpy(), time.to_numpy()),
                             "team_a": fh.faction1_hltv_team.fillna(NO_TEAM).astype(np.int64).to_numpy(),
                             "team_b": fh.faction2_hltv_team.fillna(NO_TEAM).astype(np.int64).to_numpy(),
                             "five_a": fh.five_a.to_numpy(), "five_b": fh.five_b.to_numpy(), "rounds_a": ra, "rounds_b": rb,
                             "pos": fh.map_position_in_series.fillna(0).to_numpy(int), "source": "faceit",
                             "division": [division(c) for c in fh.competition_name], "rating_a": None, "rating_b": None,
                             "won_a": (fh.team_a_score > fh.team_b_score).to_numpy(float), "map_name": None,
                             "level_a": levels(fh.match_id, fh.team_a_faceit_ids), "level_b": levels(fh.match_id, fh.team_b_faceit_ids)})[MAP_COLUMNS]

    def _series(self, h: pd.DataFrame) -> pd.DataFrame:
        g = h.groupby("match_id")
        s = g.agg(time=("time", "min"), team_a=("team_a_id", "first"), team_b=("team_b_id", "first"), bo_type=("bo_type", "first"),
                  is_lan=("is_lan", "first"), tier=("event_tier", "first"), wins=("team_a_won", "sum"), maps=("team_a_won", "size"),
                  lineup_a=("team_a_lineup_ids", "first"), lineup_b=("team_b_lineup_ids", "first")).reset_index()
        s = s[s.wins * 2 != s.maps].copy()
        s["y"] = (s.wins * 2 > s.maps).astype(int)
        s["lineup_a"] = s.lineup_a.map(five)
        s["lineup_b"] = s.lineup_b.map(five)
        lu = load(self.data_dir, "hltv/cs2_match_lineups", self.until, ["match_id", "match_date", "team_id", "is_standin"])
        st = lu.dropna(subset=["team_id"]).groupby(["match_id", "team_id"]).is_standin.sum()
        st = {(int(m), int(t)): int(v) for (m, t), v in st.items()}
        s["standins_a"] = [st.get((int(m), int(t)), 0) for m, t in zip(s.match_id, s.team_a)]
        s["standins_b"] = [st.get((int(m), int(t)), 0) for m, t in zip(s.match_id, s.team_b)]
        s["week_cutoff"] = week_cutoff(s.time)
        s = s.astype({"team_a": np.int64, "team_b": np.int64, "match_id": np.int64})
        return s.sort_values(["time", "match_id"]).reset_index(drop=True)


def target_rows(targets: pd.DataFrame) -> pd.DataFrame:
    """Targets in the shape of `Inputs.series` (no outcome)."""
    t = targets.rename(columns={"team_a_id": "team_a", "team_b_id": "team_b"})
    out = pd.DataFrame({"match_id": t.match_id.astype(np.int64).to_numpy(), "time": naive(t.match_time).to_numpy(),
                        "team_a": t.team_a.astype(np.int64).to_numpy(), "team_b": t.team_b.astype(np.int64).to_numpy(),
                        "bo_type": t.bo_type.to_numpy(), "is_lan": t.is_lan.to_numpy(), "tier": t.event_tier.to_numpy(),
                        "lineup_a": t.team_a_lineup_ids.map(five).to_numpy(), "lineup_b": t.team_b_lineup_ids.map(five).to_numpy(),
                        "standins_a": [0 if v is None else len(v) for v in t.team_a_standin_ids],
                        "standins_b": [0 if v is None else len(v) for v in t.team_b_standin_ids],
                        "week_cutoff": naive(t.week_cutoff).to_numpy()})
    return out
