"""Point-in-time inputs: decisive HLTV (and optionally FACEIT) maps on one player id space, past HLTV series with
outcomes, weekly HLTV ranks and past series' market paths.

Everything is read once through `bundle.load` at the latest cutoff of a run; each weekly fit then keeps what was known
before its own cutoff, so a fit never depends on which other weeks are in the run."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g08.market import path_features
from gnomepy_research.sessions.cs2_comp.g08.ratings import FIVE, regulation

UNRANKED = 300.0


def naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def week_start(ts: pd.Series) -> pd.Series:
    d = naive(ts).dt.normalize()
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def faceit_uid(faceit_id: str) -> int:
    return -int(hashlib.blake2b(faceit_id.encode(), digest_size=7).hexdigest(), 16)


def _five(v, mapper=None):
    if v is None or len(v) != FIVE:
        return None
    ids = tuple(mapper(x) for x in v) if mapper else tuple(int(x) for x in v)
    return ids if len(set(ids)) == FIVE else None


class Inputs:
    def __init__(self, data_dir: str | Path, until: pd.Timestamp, faceit: bool = False, market: bool = False):
        self.data_dir = Path(data_dir)
        self.until = pd.Timestamp(until)
        h = load(self.data_dir, "hltv/cs2_match_history", self.until)
        h = h[~showmatch_mask(h) & h.team_a_id.notna() & h.team_b_id.notna()].copy()
        h["time"] = naive(h.match_time.fillna(h.match_date))
        h["known"] = np.maximum(h.time, naive(h.match_date))
        self.series = self._series(h)
        parts = [self._hltv_maps(h)]
        if faceit:
            parts.append(self._faceit_maps())
        if market:
            parts.append(self._market_rows())
        self.maps = pd.concat(parts, ignore_index=True)
        rk = load(self.data_dir, "hltv/cs2_team_rankings", self.until, ["date", "team_id", "rank"])
        rk = rk.assign(date=naive(rk["date"])).sort_values("date")
        self.rankings = {int(t): (g.date.to_numpy(), g["rank"].to_numpy(float)) for t, g in rk.groupby("team_id")}

    @staticmethod
    def _series(h: pd.DataFrame) -> pd.DataFrame:
        g = h.groupby("match_id")
        s = g.agg(time=("time", "min"), known=("known", "max"), team_a=("team_a_id", "first"), team_b=("team_b_id", "first"),
                  bo_type=("bo_type", "first"), is_lan=("is_lan", "first"), event_tier=("event_tier", "first"),
                  wins=("team_a_won", "sum"), maps=("team_a_won", "size"),
                  five_a=("team_a_lineup_ids", "first"), five_b=("team_b_lineup_ids", "first"),
                  played_a=("team_a_player_ids", "first"), played_b=("team_b_player_ids", "first"))
        s = s[s.wins * 2 != s.maps].reset_index()
        s["y"] = (s.wins * 2 > s.maps).astype(int)
        for side in ("a", "b"):
            s[f"five_{side}"] = [_five(x) or _five(y) for x, y in zip(s[f"five_{side}"], s[f"played_{side}"])]
        s["cutoff"] = week_start(s.time)
        return s.drop(columns=["played_a", "played_b"]).astype({"team_a": np.int64, "team_b": np.int64})

    @staticmethod
    def _hltv_maps(h: pd.DataFrame) -> pd.DataFrame:
        h = h[h.team_a_score != h.team_b_score]
        fa = [_five(x) for x in h.team_a_player_ids]
        fb = [_five(x) for x in h.team_b_player_ids]
        ok = np.array([a is not None and b is not None for a, b in zip(fa, fb)])
        h = h[ok]
        ra, n = regulation(h.team_a_score.to_numpy(int), h.team_b_score.to_numpy(int), np.zeros(len(h), bool))
        return pd.DataFrame({"match_id": h.match_id.astype(str).to_numpy(), "known": h.known.to_numpy(),
                             "five_a": [x for x, k in zip(fa, ok) if k], "five_b": [x for x, k in zip(fb, ok) if k],
                             "rounds_a": ra, "rounds": n, "source": "hltv", "map_name": h.map_name.to_numpy(),
                             "team_a": h.team_a_id.astype(np.int64).to_numpy(), "team_b": h.team_b_id.astype(np.int64).to_numpy(),
                             "is_lan": h.is_lan.fillna(0).astype(int).to_numpy()})

    def _faceit_maps(self) -> pd.DataFrame:
        d = self.data_dir
        link = load(d, "faceit/hltv_faceit_map")
        f2h = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))

        def uid(f):
            return int(f2h[f]) if f in f2h else faceit_uid(f)

        fm = load(d, "faceit/faceit_matches", self.until, ["match_id", "finished_at", "has_stats", "faction1_hltv_team", "faction2_hltv_team"])
        fm["known"] = pd.to_datetime(fm.finished_at, unit="s")
        dup = set(load(d, "faceit/faceit_hltv_series_link", self.until).faceit_match_id)
        ex = load(d, "faceit/faceit_exclusions", self.until)
        whole = set(ex.match_id[ex.map_name.isna()])
        bad = set(zip(ex.match_id[ex.map_name.notna()], ex.map_name[ex.map_name.notna()]))
        fh = load(d, "faceit/faceit_match_history", self.until,
                  ["match_id", "map_name", "team_a_score", "team_b_score", "team_a_ot_score", "team_b_ot_score",
                   "team_a_faceit_ids", "team_b_faceit_ids", "team_a_players", "team_b_players"])
        fh = fh.merge(fm[fm.has_stats][["match_id", "known", "faction1_hltv_team", "faction2_hltv_team"]], on="match_id")
        fh = fh[~fh.match_id.isin(whole | dup)]
        fh = fh[[(m, n) not in bad for m, n in zip(fh.match_id, fh.map_name)]]
        fh = fh[(fh.team_a_players == FIVE) & (fh.team_b_players == FIVE) & (fh.team_a_score != fh.team_b_score)]
        fa = [_five(x, uid) for x in fh.team_a_faceit_ids]
        fb = [_five(x, uid) for x in fh.team_b_faceit_ids]
        ok = np.array([a is not None and b is not None for a, b in zip(fa, fb)])
        fh = fh[ok]
        ot = (fh.team_a_ot_score.fillna(0) + fh.team_b_ot_score.fillna(0) > 0).to_numpy()
        ra, n = regulation(fh.team_a_score.to_numpy(int), fh.team_b_score.to_numpy(int), ot)
        return pd.DataFrame({"match_id": fh.match_id.astype(str).to_numpy(), "known": fh.known.to_numpy(),
                             "five_a": [x for x, k in zip(fa, ok) if k], "five_b": [x for x, k in zip(fb, ok) if k],
                             "rounds_a": ra, "rounds": n, "source": "faceit", "map_name": fh.map_name.to_numpy(),
                             "team_a": fh.faction1_hltv_team.fillna(-1).astype(np.int64).to_numpy(),
                             "team_b": fh.faction2_hltv_team.fillna(-1).astype(np.int64).to_numpy(), "is_lan": 0})

    def _market_rows(self) -> pd.DataFrame:
        """Past series' kickoff prices as gap measurements on the fives the series announced. A row counts once the
        series is known on HLTV (after kickoff), the same gate as its maps."""
        paths = load(self.data_dir, "market/series_paths", self.until)
        f = path_features(paths)
        f = f[f.m0.notna()]
        s = self.series.set_index("match_id")
        f = f[f.match_id.isin(s.index)]
        s = s.loc[f.match_id]
        ok = np.array([a is not None and b is not None for a, b in zip(s.five_a, s.five_b)])
        s, f = s[ok], f[ok]
        return pd.DataFrame({"match_id": s.index.astype(str).to_numpy(), "known": s.known.to_numpy(),
                             "five_a": list(s.five_a), "five_b": list(s.five_b), "rounds_a": np.nan, "rounds": np.nan,
                             "source": "market", "team_a": s.team_a.to_numpy(np.int64), "team_b": s.team_b.to_numpy(np.int64),
                             "is_lan": s.is_lan.fillna(0).astype(int).to_numpy(), "p_a": f.m0.to_numpy(float)})

    def rank(self, team: int, day: pd.Timestamp, max_age_days: int = 120) -> float:
        h = self.rankings.get(int(team))
        if h is None:
            return UNRANKED
        i = np.searchsorted(h[0], np.datetime64(day), side="left") - 1
        if i < 0 or (np.datetime64(day) - h[0][i]) > np.timedelta64(max_age_days, "D"):
            return UNRANKED
        return float(h[1][i])
