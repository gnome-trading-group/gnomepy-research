"""Point-in-time team features for the e02 series model.

Every feature is read as of a Monday 00:00 UTC cut. A target's week_cutoff is such a cut, and so is every training
series' week, so both are described by the same rule: the rows the bundle holds before that cut and nothing else.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

EPOCH = pd.Timestamp("1970-01-01")
ELO_START = 1500.0
ELO_K = 24.0
FORM_GAMES = 10
ACTIVITY_DAYS = 90
RATING_MAPS = 30
FACEIT_MAPS = 60
FACEIT_HISTORY_DAYS = 180
SHRINK_MAPS = 20.0
RATING_PRIOR = 1.0
KD_PRIOR = 1.0
ADR_PRIOR = 70.0

HISTORY_COLUMNS = [
    "match_id", "match_date", "match_time", "event_name", "format", "event_tier", "is_lan", "bo_type",
    "team_a_id", "team_b_id", "team_a_won", "map_position_in_series", "team_a_lineup_ids", "team_b_lineup_ids",
]


def monday(t: pd.Series) -> pd.Series:
    naive = t.dt.tz_convert("UTC").dt.tz_localize(None) if t.dt.tz is not None else t
    return naive.dt.normalize() - pd.to_timedelta(naive.dt.weekday, unit="D")


def _rows(frame: pd.DataFrame | None):
    return () if frame is None else frame.itertuples()


def _lineup(ids) -> tuple[int, ...]:
    if ids is None:
        return ()
    return tuple(int(x) for x in ids if x is not None and not pd.isna(x))


def series_table(history: pd.DataFrame) -> pd.DataFrame:
    h = history.loc[~showmatch_mask(history)]
    maps = h.groupby("match_id").size().rename("maps")
    a_maps = h.groupby("match_id").team_a_won.sum().rename("a_maps")
    head = h.sort_values("map_position_in_series").drop_duplicates("match_id").set_index("match_id")
    s = head[["match_time", "team_a_id", "team_b_id", "bo_type", "is_lan", "event_tier",
              "team_a_lineup_ids", "team_b_lineup_ids"]].join([maps, a_maps])
    s["b_maps"] = s.maps - s.a_maps
    winner_maps = np.maximum(s.a_maps, s.b_maps)
    s = s[(s.a_maps != s.b_maps) & (winner_maps == (s.bo_type + 1) // 2) & (s.maps <= s.bo_type)].copy()
    s["y"] = (s.a_maps > s.b_maps).astype(int)
    s["week"] = monday(s.match_time)
    s["lineup_a"] = s.team_a_lineup_ids.map(_lineup)
    s["lineup_b"] = s.team_b_lineup_ids.map(_lineup)
    return s.drop(columns=["team_a_lineup_ids", "team_b_lineup_ids"])


class PlayerLog:
    """Time-ordered per-player values with cumulative sums, so a window before a cut is one subtraction."""

    def __init__(self, frame: pd.DataFrame, key: str, time: str, values: list[str]):
        self._logs: dict = {}
        if frame.empty:
            return
        codes, uniques = pd.factorize(frame[key])
        t = frame[time].to_numpy()
        v = frame[values].to_numpy(dtype=float)
        order = np.lexsort((t, codes))
        codes, t, v = codes[order], t[order], v[order]
        cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
        bounds = np.flatnonzero(np.diff(codes)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(codes)]])
        for c, s, e in zip(codes[starts], starts, ends):
            self._logs[uniques[c]] = (t[s:e], cs[s:e + 1] - cs[s])

    def last_mean(self, key, cut, window: int):
        log = self._logs.get(key)
        if log is None:
            return None, 0
        t, cs = log
        hi = int(np.searchsorted(t, cut, side="left"))
        lo = max(hi - window, 0)
        n = hi - lo
        if n == 0:
            return None, 0
        return (cs[hi] - cs[lo]) / n, n

    def count_between(self, key, since, cut) -> int:
        log = self._logs.get(key)
        if log is None:
            return 0
        t = log[0]
        return int(np.searchsorted(t, cut, side="left") - np.searchsorted(t, since, side="left"))


class Ranks:
    def __init__(self, rankings: pd.DataFrame):
        self._days: dict = {}
        r = rankings.assign(day=(rankings.date - EPOCH).dt.days).sort_values(["team_id", "day"])
        for team, g in r.groupby("team_id", sort=False):
            self._days[team] = (g.day.to_numpy(), g.points.to_numpy(dtype=float))

    def points(self, team, cut_day: int) -> float:
        log = self._days.get(team)
        if log is None:
            return np.nan
        days, pts = log
        i = int(np.searchsorted(days, cut_day, side="left")) - 1
        return float(pts[i]) if i >= 0 else np.nan


def _shrink(mean: float | None, n: int, prior: float) -> float:
    if n == 0:
        return prior
    return (mean * n + SHRINK_MAPS * prior) / (n + SHRINK_MAPS)


class Ratings:
    """Running team state, advanced one week at a time; queries for a week see only earlier weeks."""

    def __init__(self, ranks: Ranks, hltv_rating: PlayerLog, faceit_log: PlayerLog | None, hltv_to_faceit: dict):
        self.elo = defaultdict(lambda: ELO_START)
        self.form = defaultdict(lambda: deque(maxlen=FORM_GAMES))
        self.recent = defaultdict(deque)
        self.lineup: dict[int, tuple[int, ...]] = {}
        self.ranks = ranks
        self.hltv_rating = hltv_rating
        self.faceit_log = faceit_log
        self.hltv_to_faceit = hltv_to_faceit

    def features(self, team, week: pd.Timestamp) -> dict:
        cut_day = (week - EPOCH).days
        cut_sec = int((week - EPOCH).total_seconds())
        rec = self.recent[team]
        while rec and rec[0][0] < cut_day - ACTIVITY_DAYS:
            rec.popleft()
        form = self.form[team]
        pts = self.ranks.points(team, cut_day)
        out = {
            "elo": self.elo[team],
            "pts": np.log1p(pts) if not np.isnan(pts) else np.nan,
            "form": float(np.mean(form)) if form else np.nan,
            "act": float(np.log1p(sum(m for _, m in rec))),
        }
        out.update(self._lineup_features(self.lineup.get(team, ()), cut_day, cut_sec))
        return out

    def _lineup_features(self, lineup, cut_day: int, cut_sec: int) -> dict:
        if not lineup:
            return {"hrat": np.nan, "hcov": 0, "kd": np.nan, "adr": np.nan, "fmaps": np.nan, "fcov": 0}
        ratings, hcov = [], 0
        kd, adr, fmaps, fcov = [], [], 0, 0
        for pid in lineup:
            mean, n = self.hltv_rating.last_mean(pid, cut_day, RATING_MAPS)
            hcov += n > 0
            ratings.append(_shrink(mean[0] if n else None, n, RATING_PRIOR))
            if self.faceit_log is None:
                continue
            fid = self.hltv_to_faceit.get(pid)
            if fid is None:
                kd.append(KD_PRIOR)
                adr.append(ADR_PRIOR)
                continue
            mean, n = self.faceit_log.last_mean(fid, cut_sec, FACEIT_MAPS)
            kd.append(_shrink(mean[0] if n else None, n, KD_PRIOR))
            adr.append(_shrink(mean[1] if n else None, n, ADR_PRIOR))
            seen = self.faceit_log.count_between(fid, cut_sec - FACEIT_HISTORY_DAYS * 86400, cut_sec)
            fmaps += seen
            fcov += seen > 0
        out = {"hrat": float(np.mean(ratings)), "hcov": hcov}
        if self.faceit_log is not None:
            out.update({"kd": float(np.mean(kd)), "adr": float(np.mean(adr)),
                        "fmaps": float(np.log1p(fmaps)), "fcov": fcov})
        return out

    def update(self, row) -> None:
        a, b, y = row.team_a_id, row.team_b_id, row.y
        expected_a = 1 / (1 + 10 ** ((self.elo[b] - self.elo[a]) / 400))
        delta = ELO_K * (y - expected_a)
        self.elo[a] += delta
        self.elo[b] -= delta
        self.form[a].append(y)
        self.form[b].append(1 - y)
        day = (row.week - EPOCH).days
        games = row.a_maps + row.b_maps
        self.recent[a].append((day, games))
        self.recent[b].append((day, games))
        if row.lineup_a:
            self.lineup[a] = row.lineup_a
        if row.lineup_b:
            self.lineup[b] = row.lineup_b


def _pair_row(fa: dict, fb: dict, is_lan, tier, faceit: bool) -> dict:
    row = {
        "d_elo": fa["elo"] - fb["elo"],
        "d_pts": fa["pts"] - fb["pts"],
        "d_form": fa["form"] - fb["form"],
        "d_act": fa["act"] - fb["act"],
        "d_hrat": fa["hrat"] - fb["hrat"],
        "a_hcov": fa["hcov"],
        "b_hcov": fb["hcov"],
        "is_lan": is_lan,
        "tier": tier,
    }
    if faceit:
        row.update({
            "d_kd": fa["kd"] - fb["kd"],
            "d_adr": fa["adr"] - fb["adr"],
            "d_fmaps": fa["fmaps"] - fb["fmaps"],
            "a_fcov": fa["fcov"],
            "b_fcov": fb["fcov"],
        })
    return row


@dataclass(frozen=True)
class Queries:
    meta: pd.DataFrame
    ab: pd.DataFrame
    ba: pd.DataFrame


def feature_columns(faceit: bool) -> list[str]:
    cols = ["d_elo", "d_pts", "d_form", "d_act", "d_hrat", "a_hcov", "b_hcov", "is_lan", "tier"]
    if faceit:
        cols += ["d_kd", "d_adr", "d_fmaps", "a_fcov", "b_fcov"]
    return cols


def _load_faceit(data_dir: str, cut: pd.Timestamp) -> tuple[PlayerLog, dict]:
    fm = load(data_dir, "faceit/faceit_matches", cut, ["match_id", "finished_at"])
    fpm = load(data_dir, "faceit/faceit_player_map_stats", cut, ["match_id", "faceit_id", "k_d_ratio", "adr"])
    fpm = fpm.merge(fm, on="match_id").dropna(subset=["k_d_ratio", "adr"])
    log = PlayerLog(fpm, "faceit_id", "finished_at", ["k_d_ratio", "adr"])
    hm = load(data_dir, "faceit/hltv_faceit_map", None)
    return log, dict(zip(hm.hltv_id.astype(int), hm.faceit_id))


def build_queries(data_dir: str, cut: pd.Timestamp, targets: pd.DataFrame | None, faceit: bool) -> Queries:
    history = load(data_dir, "hltv/cs2_match_history", cut, HISTORY_COLUMNS)
    series = series_table(history)
    ranks = Ranks(load(data_dir, "hltv/cs2_team_rankings", cut, ["date", "team_id", "points"]))
    pms = load(data_dir, "hltv/cs2_player_map_stats", cut, ["match_date", "player_id", "side", "rating"])
    pms = pms[(pms.side == "all") & pms.rating.notna()].assign(day=lambda d: (d.match_date - EPOCH).dt.days)
    hltv_rating = PlayerLog(pms, "player_id", "day", ["rating"])
    faceit_log, hltv_to_faceit = _load_faceit(data_dir, cut) if faceit else (None, {})
    state = Ratings(ranks, hltv_rating, faceit_log, hltv_to_faceit)

    target_weeks = {} if targets is None else {w: g for w, g in targets.groupby("week_cutoff")}
    series_weeks = {w: g.sort_values("match_time") for w, g in series.groupby("week")}
    meta_rows, ab_rows, ba_rows = [], [], []

    for week in sorted(set(series_weeks) | set(target_weeks)):
        for t in _rows(target_weeks.get(week)):
            fa = state.features(t.team_a_id, week)
            fb = state.features(t.team_b_id, week)
            meta_rows.append({"match_id": t.match_id, "week": week, "bo": t.bo_type, "y": np.nan, "is_target": True})
            ab_rows.append(_pair_row(fa, fb, t.is_lan, t.event_tier, faceit))
            ba_rows.append(_pair_row(fb, fa, t.is_lan, t.event_tier, faceit))
        games = series_weeks.get(week)
        if games is None:
            continue
        for s in games.itertuples():
            fa = state.features(s.team_a_id, week)
            fb = state.features(s.team_b_id, week)
            meta_rows.append({"match_id": s.Index, "week": week, "bo": s.bo_type, "y": s.y, "is_target": False})
            ab_rows.append(_pair_row(fa, fb, s.is_lan, s.event_tier, faceit))
            ba_rows.append(_pair_row(fb, fa, s.is_lan, s.event_tier, faceit))
        for s in games.itertuples():
            state.update(s)

    meta = pd.DataFrame(meta_rows)
    cols = feature_columns(faceit)
    return Queries(meta=meta, ab=pd.DataFrame(ab_rows)[cols], ba=pd.DataFrame(ba_rows)[cols])
