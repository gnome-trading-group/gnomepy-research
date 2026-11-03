"""
Point-in-time inputs for g14: every decisive map (HLTV, optionally FACEIT) on one player id space, ordered by the
moment it became known; HLTV bo3 series with their announced fives and outcomes; team rankings and roster moves;
Polymarket kickoff prices and price-path features.

Everything is read once through `bundle.load` at the run's latest cutoff. Rows carry `known` (HLTV: match time;
FACEIT: when FACEIT finalised the series) so a state after the rows known before a cutoff is the same whatever else
the bundle holds.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths

FIVE = 5
UNRANKED = 300.0
LOG_UNRANKED = float(np.log(UNRANKED))
DIVISIONS = ("entry", "open", "intermediate", "main", "advanced", "pro", "qualifier", "other")
_DIVISION_RE = re.compile(r"(entry|open10|open|intermediate|main|advanced|premier|challenger|pro league|qualifier)", re.I)
SOURCE_HLTV, SOURCE_FACEIT = 0, 1


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


def naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _fives(col: pd.Series, mapper=None) -> tuple[np.ndarray, np.ndarray]:
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


def _regulation(score_a: np.ndarray, score_b: np.ndarray, overtime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An overtime map counts as 12-12 of 24 regulation rounds."""
    ot = overtime | (score_a + score_b > 24)
    return np.where(ot, 12, score_a), np.where(ot, 24, score_a + score_b)


@dataclass
class Inputs:
    maps: pd.DataFrame          # one decisive map per row, sorted by (known, time, match_id, position)
    five_a: np.ndarray          # (N, 5) unified player ids, aligned with maps
    five_b: np.ndarray
    rating_a: np.ndarray        # (N, 5) HLTV map ratings of the five, NaN where unknown
    rating_b: np.ndarray
    series: pd.DataFrame        # HLTV bo3 series with an outcome, announced fives and stand-in counts
    rankings: dict[int, tuple[np.ndarray, np.ndarray]]
    moves: pd.DataFrame         # roster moves: event_date, team_id
    market: pd.DataFrame        # per series with a market: kickoff, m0 and path features (known at kickoff)
    h2f: dict[int, str]

    def rank(self, team: int, day: pd.Timestamp, max_age_days: int = 120) -> float:
        h = self.rankings.get(int(team))
        if h is None:
            return UNRANKED
        i = np.searchsorted(h[0], np.datetime64(day), side="left") - 1
        if i < 0 or (np.datetime64(day) - h[0][i]) > np.timedelta64(max_age_days, "D"):
            return UNRANKED
        return float(h[1][i])


def _ratings(data_dir: Path, until: pd.Timestamp, h: pd.DataFrame, five_a: np.ndarray, five_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    st = load(data_dir, "hltv/cs2_player_map_stats", until, ["match_id", "match_date", "map_name", "player_id", "side", "rating"])
    st = st[(st.side == "all") & st.rating.notna() & st.player_id.notna()]
    rating = dict(zip(zip(st.match_id.astype(str), st.map_name, st.player_id.astype(np.int64)), st.rating.astype(float)))
    mids, names = h.match_id.astype(str).to_numpy(), h.map_name.to_numpy()
    out = []
    for five in (five_a, five_b):
        r = np.full(five.shape, np.nan)
        for i in range(len(five)):
            for j in range(FIVE):
                r[i, j] = rating.get((mids[i], names[i], int(five[i, j])), np.nan)
        out.append(r)
    return out[0], out[1]


def _hltv_maps(data_dir: Path, until: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    h = load(data_dir, "hltv/cs2_match_history", until)
    h = h[~showmatch_mask(h)].copy()
    h["time"] = naive(h.match_time.fillna(h.match_date))
    h["known"] = np.maximum(h.time, naive(h.match_date))
    series = _series(h)
    h = h[(h.team_a_score != h.team_b_score) & h.team_a_id.notna() & h.team_b_id.notna()]
    five_a, ok_a = _fives(h.team_a_player_ids)
    five_b, ok_b = _fives(h.team_b_player_ids)
    ok = ok_a & ok_b
    h, five_a, five_b = h[ok], five_a[ok], five_b[ok]
    ra, n = _regulation(h.team_a_score.to_numpy(int), h.team_b_score.to_numpy(int), np.zeros(len(h), bool))
    rating_a, rating_b = _ratings(data_dir, until, h, five_a, five_b)
    frame = pd.DataFrame({
        "match_id": h.match_id.astype(str).to_numpy(), "time": h.time.to_numpy(), "known": h.known.to_numpy(),
        "position": h.map_position_in_series.fillna(0).to_numpy(int),
        "team_a": h.team_a_id.astype(np.int64).to_numpy(), "team_b": h.team_b_id.astype(np.int64).to_numpy(),
        "rank_a": h.team_a_rank.fillna(UNRANKED).to_numpy(float), "rank_b": h.team_b_rank.fillna(UNRANKED).to_numpy(float),
        "rounds": n, "rounds_a": ra, "source": SOURCE_HLTV, "division": "hltv", "map_name": h.map_name.fillna("").to_numpy(),
        "is_lan": h.is_lan.fillna(0).to_numpy(int), "tier": h.event_tier.fillna(0).to_numpy(int),
        "bo_type": h.bo_type.fillna(0).to_numpy(int)})
    return frame, five_a, five_b, rating_a, rating_b, series


def _series(h: pd.DataFrame) -> pd.DataFrame:
    d = h[h.team_a_id.notna() & h.team_b_id.notna()]
    g = d.groupby("match_id")
    s = g.agg(time=("time", "min"), known=("known", "max"), team_a=("team_a_id", "first"), team_b=("team_b_id", "first"),
              bo_type=("bo_type", "first"), is_lan=("is_lan", "first"), tier=("event_tier", "first"),
              wins=("team_a_won", "sum"), maps=("team_a_won", "size"),
              lineup_a=("team_a_lineup_ids", "first"), lineup_b=("team_b_lineup_ids", "first"),
              rank_a=("team_a_rank", "first"), rank_b=("team_b_rank", "first"))
    s = s[(s.wins * 2 != s.maps) & (s.bo_type == 3)].reset_index()
    s["y"] = (s.wins * 2 > s.maps).astype(int)
    s["rank_a"] = s.rank_a.fillna(UNRANKED)
    s["rank_b"] = s.rank_b.fillna(UNRANKED)
    return s.astype({"team_a": np.int64, "team_b": np.int64})


def _faceit_maps(data_dir: Path, until: pd.Timestamp, h2f: dict[int, str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    f2h = {f: h for h, f in h2f.items()}
    fm = load(data_dir, "faceit/faceit_matches", until,
              ["match_id", "finished_at", "has_stats", "competition_name", "faction1_hltv_team", "faction2_hltv_team",
               "faction1_id", "faction2_id"])
    fm["known"] = pd.to_datetime(fm.finished_at, unit="s")
    link = load(data_dir, "faceit/faceit_hltv_series_link", until, ["faceit_match_id", "hltv_match_id"])
    fm = fm[fm.has_stats & ~fm.match_id.isin(set(link.faceit_match_id))]
    ex = load(data_dir, "faceit/faceit_exclusions", until)
    whole = set(ex.match_id[ex.map_name.isna()])
    bad_maps = set(zip(ex.match_id[ex.map_name.notna()], ex.map_name[ex.map_name.notna()]))
    cols = ["match_id", "map_name", "match_time", "match_date", "map_position_in_series", "team_a_score", "team_b_score",
            "team_a_ot_score", "team_b_ot_score", "team_a_faceit_ids", "team_b_faceit_ids", "team_a_players", "team_b_players"]
    fh = load(data_dir, "faceit/faceit_match_history", until, cols)
    fh = fh.merge(fm[["match_id", "known", "competition_name", "faction1_hltv_team", "faction2_hltv_team", "faction1_id", "faction2_id"]],
                  on="match_id")
    fh = fh[~fh.match_id.isin(whole)]
    fh = fh[[(m, n) not in bad_maps for m, n in zip(fh.match_id, fh.map_name)]]
    fh = fh[(fh.team_a_players == FIVE) & (fh.team_b_players == FIVE) & (fh.team_a_score != fh.team_b_score)]

    def uid(fid: str) -> int:
        h = f2h.get(fid)
        return int(h) if h is not None else faceit_uid(fid)

    five_a, ok_a = _fives(fh.team_a_faceit_ids, uid)
    five_b, ok_b = _fives(fh.team_b_faceit_ids, uid)
    ok = ok_a & ok_b
    fh, five_a, five_b = fh[ok], five_a[ok], five_b[ok]
    ot = (fh.team_a_ot_score.fillna(0) + fh.team_b_ot_score.fillna(0) > 0).to_numpy()
    ra, n = _regulation(fh.team_a_score.to_numpy(int), fh.team_b_score.to_numpy(int), ot)
    team_a = np.where(fh.faction1_hltv_team.notna(), fh.faction1_hltv_team.fillna(0).astype(np.int64),
                      [faceit_uid(str(x)) for x in fh.faction1_id])
    team_b = np.where(fh.faction2_hltv_team.notna(), fh.faction2_hltv_team.fillna(0).astype(np.int64),
                      [faceit_uid(str(x)) for x in fh.faction2_id])
    frame = pd.DataFrame({
        "match_id": fh.match_id.to_numpy(), "time": naive(fh.match_time.fillna(fh.match_date)).to_numpy(),
        "known": fh.known.to_numpy(), "position": fh.map_position_in_series.fillna(0).to_numpy(int),
        "team_a": team_a.astype(np.int64), "team_b": team_b.astype(np.int64),
        "rank_a": UNRANKED, "rank_b": UNRANKED, "rounds": n, "rounds_a": ra, "source": SOURCE_FACEIT,
        "division": [division(c) for c in fh.competition_name], "map_name": fh.map_name.fillna("").to_numpy(),
        "is_lan": 0, "tier": 0, "bo_type": 0})
    return frame, five_a, five_b


def _standins(data_dir: Path, until: pd.Timestamp, series: pd.DataFrame) -> pd.DataFrame:
    lu = load(data_dir, "hltv/cs2_match_lineups", until, ["match_id", "match_date", "team_id", "is_standin"]).dropna(subset=["team_id"])
    st = lu.groupby(["match_id", "team_id"]).is_standin.sum()
    subs = {(int(m), int(t)): int(v) for (m, t), v in st.items()}
    series["sub_a"] = [subs.get((int(m), int(t)), 0) for m, t in zip(series.match_id, series.team_a)]
    series["sub_b"] = [subs.get((int(m), int(t)), 0) for m, t in zip(series.match_id, series.team_b)]
    return series


def path_features(paths: pd.DataFrame) -> pd.DataFrame:
    """Per series: the kickoff price m0 (last print within 30 minutes of kickoff) and how the path behaved before it.
    Prints are read only up to each series' kickoff."""
    p = paths.copy()
    p["t"] = naive(p.t)
    p["kickoff"] = naive(p.kickoff)
    p = p[p.t < p.kickoff].sort_values(["match_id", "t"], kind="stable")
    p["lead_h"] = (p.kickoff - p.t).dt.total_seconds() / 3600.0
    p["lp"] = logit(p.p_a)
    rows = []
    for mid, g in p.groupby("match_id", sort=False):
        lead = g.lead_h.to_numpy()
        lp = g.lp.to_numpy()
        pa = g.p_a.to_numpy()
        last = len(g) - 1
        m0 = pa[last] if lead[last] <= 0.5 else np.nan
        m_any = pa[last]

        def at(hours: float) -> float:
            i = np.searchsorted(-lead, -hours, side="right") - 1
            return lp[i] if i >= 0 else lp[0]

        recent = lead <= 6.0
        dl = np.diff(lp[recent]) if recent.sum() > 2 else np.zeros(1)
        rows.append({"match_id": mid, "kickoff": g.kickoff.iloc[0], "m0": m0, "m_any": m_any,
                     "lead_last_h": lead[last], "n_prints": len(g), "span_h": lead[0] - lead[last],
                     "drift24": lp[last] - at(24.0), "drift6": lp[last] - at(6.0), "drift1": lp[last] - at(1.0),
                     "flat_half": float((pa == 0.5).mean()), "vol6": float(np.sqrt(np.mean(dl * dl))),
                     "prints_1h": int((lead <= 1.0).sum())})
    return pd.DataFrame(rows)


def _market(data_dir: Path, until: pd.Timestamp) -> pd.DataFrame:
    sp = load(data_dir, "market/series_paths", until)
    if sp.empty:
        return pd.DataFrame(columns=["match_id", "kickoff", "m0"])
    mk = path_features(sp)
    mk["known"] = mk.kickoff
    return mk


def load_inputs(data_dir: str | Path, until: pd.Timestamp, faceit: bool) -> Inputs:
    data_dir = Path(data_dir)
    until = pd.Timestamp(until)
    link = load(data_dir, "faceit/hltv_faceit_map")
    h2f = dict(zip(link.hltv_id.astype(np.int64), link.faceit_id))
    frame, five_a, five_b, rating_a, rating_b, series = _hltv_maps(data_dir, until)
    parts, fa, fb, ra, rb = [frame], [five_a], [five_b], [rating_a], [rating_b]
    if faceit:
        f_frame, f_a, f_b = _faceit_maps(data_dir, until, h2f)
        parts.append(f_frame)
        fa.append(f_a)
        fb.append(f_b)
        ra.append(np.full(f_a.shape, np.nan))
        rb.append(np.full(f_b.shape, np.nan))
    maps = pd.concat(parts, ignore_index=True)
    five_a, five_b = np.concatenate(fa), np.concatenate(fb)
    rating_a, rating_b = np.concatenate(ra), np.concatenate(rb)
    order = np.lexsort((maps.position.to_numpy(), maps.match_id.astype(str).to_numpy(), maps.time.to_numpy(), maps.known.to_numpy()))
    maps = maps.iloc[order].reset_index(drop=True)
    five_a, five_b = five_a[order], five_b[order]
    rating_a, rating_b = rating_a[order], rating_b[order]
    series = _standins(data_dir, until, series)
    series = series.sort_values(["time", "match_id"], kind="stable").reset_index(drop=True)
    rk = load(data_dir, "hltv/cs2_team_rankings", until, ["date", "team_id", "rank"])
    rk = rk.assign(date=naive(rk["date"])).sort_values("date")
    rankings = {int(t): (g.date.to_numpy(), g["rank"].to_numpy(float)) for t, g in rk.groupby("team_id")}
    mv = load(data_dir, "hltv/cs2_roster_moves", until, ["event_date", "kind", "from_team_id", "to_team_id", "person_role"])
    mv = mv[mv.person_role.eq("player") & mv.kind.isin(["join", "leave", "transfer", "benched", "activated", "loan", "loan_return"])]
    mv = pd.concat([pd.DataFrame({"event_date": naive(mv.event_date), "team_id": mv.to_team_id}),
                    pd.DataFrame({"event_date": naive(mv.event_date), "team_id": mv.from_team_id})])
    mv = mv[mv.team_id.fillna(0) > 0].astype({"team_id": np.int64}).sort_values("event_date").reset_index(drop=True)
    market = _market(data_dir, until)
    return Inputs(maps, five_a, five_b, rating_a, rating_b, series, rankings, mv, market, h2f)


def target_market(data_dir: str | Path, match_ids) -> pd.DataFrame:
    return path_features(target_paths(data_dir, match_ids))
