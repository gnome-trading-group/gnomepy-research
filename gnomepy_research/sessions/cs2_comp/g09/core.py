"""
g09 building blocks: market path summaries, an online player power rating from HLTV results, and an online team
rating implied by past market closes. Every pass is chronological and answers each request with the state as of the
request's snapshot time, so one pass yields point-in-time features for past series (snapshotted at their own week
start) and for the targets (snapshotted at the cutoff).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths
from gnomepy_research.sessions.cs2_comp.g09.team import team_features

UNRANKED = 300.0
LOW_DIVISION = re.compile(r"entry|open|intermediate", re.I)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def week_start(ts: pd.Series) -> pd.Series:
    d = naive(ts).dt.normalize()
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def bo3(p: np.ndarray) -> np.ndarray:
    return p * p * (3 - 2 * p)


def path_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Per series: the scored kickoff price m0 (last print within 30 minutes of kickoff), time-weighted mean logits
    over the last hours, and how active the book was. Minute prints mean a plain mean over prints is time-weighted."""
    if paths.empty:
        return pd.DataFrame(columns=["m0", "tw1", "tw3", "tw6", "nchg3", "nprint", "zlast"]).rename_axis("match_id")
    d = paths[["match_id", "kickoff", "t", "p_a"]].copy()
    d["age"] = (naive(d.kickoff) - naive(d.t)).dt.total_seconds() / 3600.0
    d = d[d.age >= 0].sort_values(["match_id", "t"])
    d["z"] = logit(np.clip(d.p_a.to_numpy(float), 0.005, 0.995))
    d["chg"] = d.groupby("match_id").p_a.diff().fillna(0).ne(0)
    g = d.groupby("match_id")
    out = pd.DataFrame(index=g.size().index)
    last = d[d.age <= 0.5].groupby("match_id").p_a.last()
    out["m0"] = last
    for h in (1, 3, 6):
        out[f"tw{h}"] = d[d.age <= h].groupby("match_id").z.mean()
    out["nchg3"] = d[d.age <= 3].groupby("match_id").chg.sum()
    out["nprint"] = g.size()
    out["zlast"] = g.z.last()
    return out


def series_outcomes(maps: pd.DataFrame) -> pd.DataFrame:
    g = maps.groupby("match_id").agg(a=("team_a_won", "sum"), n=("team_a_won", "size"), team_a=("team_a_id", "first"),
                                     team_b=("team_b_id", "first"), time=("time", "first"), bo=("bo_type", "first"))
    g["y"] = np.where(2 * g.a > g.n, 1.0, np.where(2 * g.a < g.n, 0.0, np.nan))
    return g


def load_maps(data_dir: str | Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", cutoff)
    h = h[~showmatch_mask(h)].copy()
    h["time"] = naive(h.match_time.fillna(h.match_date))
    h = h[h.time < cutoff]
    return h.sort_values(["time", "match_id", "map_position_in_series"]).reset_index(drop=True)


def _ids(v) -> tuple:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ()
    return tuple(x if isinstance(x, str) else int(x) for x in v)


def load_faceit_maps(data_dir: str | Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    """FACEIT 5v5 maps known before cutoff, timed by when FACEIT finalised the match, without series that HLTV also
    has and without excluded rows. Players are keyed by HLTV id where linked, else by FACEIT id."""
    m = load(data_dir, "faceit/faceit_matches", cutoff, ["match_id", "finished_at", "competition_name"])
    h = load(data_dir, "faceit/faceit_match_history", cutoff,
             ["match_id", "map_name", "map_position_in_series", "team_a_faceit_ids", "team_b_faceit_ids", "team_a_score",
              "team_b_score", "team_a_won"])
    link = load(data_dir, "faceit/faceit_hltv_series_link", cutoff)
    ex = load(data_dir, "faceit/faceit_exclusions", cutoff)
    h = h[~h.match_id.isin(link.faceit_match_id)]
    h = h[~h.match_id.isin(ex[ex.map_name.isna()].match_id)]
    exm = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
    h = h[[(a, b) not in exm for a, b in zip(h.match_id, h.map_name)]]
    ok = np.array([v is not None and len(v) == 5 for v in h.team_a_faceit_ids]) & np.array([v is not None and len(v) == 5 for v in h.team_b_faceit_ids])
    h = h[ok]
    idmap = dict(zip(*[load(data_dir, "faceit/hltv_faceit_map")[c] for c in ("faceit_id", "hltv_id")]))
    for side in ("a", "b"):
        h[f"team_{side}_player_ids"] = [tuple(int(idmap[f]) if f in idmap else f for f in v) for v in h[f"team_{side}_faceit_ids"]]
    h = h.merge(m, on="match_id")
    h["time"] = pd.to_datetime(h.finished_at, unit="s")
    h = h[h.time < cutoff]
    return pd.DataFrame({"time": h.time.to_numpy(), "team_a_player_ids": h.team_a_player_ids.to_numpy(),
                         "team_b_player_ids": h.team_b_player_ids.to_numpy(), "team_a_score": h.team_a_score.to_numpy(float),
                         "team_b_score": h.team_b_score.to_numpy(float), "team_a_won": h.team_a_won.to_numpy(float),
                         "team_a_rank": np.nan, "team_b_rank": np.nan, "src": "faceit",
                         "low": [bool(LOW_DIVISION.search(c or "")) for c in h.competition_name],
                         "mapstatsid": [f"{m}#{p}" for m, p in zip(h.match_id, h.map_position_in_series)]})


def faceit_map_stats(data_dir: str | Path, cutoff: pd.Timestamp) -> dict:
    """FACEIT map key -> {player: (performance on an HLTV-rating-like scale, 0)}, from kills per round and ADR, so
    FACEIT maps also say who carries a five. Players keyed as in load_faceit_maps."""
    s = load(data_dir, "faceit/faceit_player_map_stats", cutoff, ["match_id", "map_position_in_series", "hltv_id", "faceit_id", "adr", "k_r_ratio"])
    s = s[s.adr.notna() & s.k_r_ratio.notna()]
    perf = 0.5 * s.k_r_ratio.astype(float) / 0.7 + 0.5 * s.adr.astype(float) / 75.0
    key = [int(h) if h == h and h is not None else f for h, f in zip(s.hltv_id, s.faceit_id)]
    mk = s.match_id.astype(str) + "#" + s.map_position_in_series.astype(int).astype(str)
    out: dict = {}
    for k, p, v in zip(mk, key, perf):
        out.setdefault(k, {})[p] = (float(v), 0.0)
    return out


def map_ratings(data_dir: str | Path, cutoff: pd.Timestamp) -> dict:
    """mapstatsid -> {player_id: (HLTV rating, round swing / 10)} for maps before cutoff."""
    s = load(data_dir, "hltv/cs2_player_map_stats", cutoff, ["match_date", "mapstatsid", "player_id", "side", "rating", "round_swing_pct"])
    s = s[(s.side == "all") & s.rating.notna() & s.mapstatsid.notna()]
    sw = s.round_swing_pct.fillna(0.0).astype(float) / 10.0
    return {k: dict(zip(g.player_id.astype(int), zip(g.rating.astype(float), sw[g.index]))) for k, g in s.groupby("mapstatsid")}


def player_stats(data_dir: str | Path, cutoff: pd.Timestamp, cfg: EloConfig) -> dict | None:
    if cfg.ind_weight == 0 and cfg.swing_weight == 0:
        return None
    st = map_ratings(data_dir, cutoff)
    if cfg.k_faceit > 0 and cfg.faceit_stats:
        st.update(faceit_map_stats(data_dir, cutoff))
    return st


def combined_maps(maps: pd.DataFrame, faceit: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["time", "team_a_player_ids", "team_b_player_ids", "team_a_score", "team_b_score", "team_a_won", "team_a_rank", "team_b_rank",
            "mapstatsid", "is_lan"]
    h = maps[cols].assign(src="hltv", hidx=np.arange(len(maps)))
    if faceit is None or faceit.empty:
        return h
    return pd.concat([h, faceit.assign(hidx=-1)], ignore_index=True).sort_values("time", kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class EloConfig:
    k: float = 0.10
    k_new: float = 0.82
    n_settle: float = 30.0
    new_gap: float = 0.71
    rank_scale: float = 2.15
    margin: float = 1.0
    decay_days: float = 11680.0
    # FACEIT maps are another, noisier measurement of the same players; a five never seen anywhere enters below an
    # unranked HLTV team.
    k_faceit: float = 0.0
    faceit_offset: float = -0.5
    # Who carries a five: a recency-weighted HLTV rating deviation from the five's mean, added to the player's
    # strength, so a five that changes players is priced by who leaves and who joins.
    ind_weight: float = 13.0
    ind_alpha: float = 0.0044
    swing_weight: float = 0.0
    faceit_stats: bool = False
    # FACEIT's entry, open and intermediate divisions, relative to its other competitions.
    k_faceit_low: float = 1.0
    # LAN maps relative to online HLTV maps.
    k_lan: float = 1.0
    # Each HLTV map nudges a five toward the strength its HLTV rank implies, as a bookmaker blends in the rankings.
    rank_pull: float = 0.0
    # A fast component on top of the slow rating: reacts to the last few maps, then fades.
    k_fast: float = 0.0
    fast_decay: float = 0.1


def elo_pass(maps: pd.DataFrame, requests: pd.DataFrame, cfg: EloConfig = EloConfig(),
             stats: dict | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Player-level map Elo in logit units. A team is the mean of its five. A player never seen enters at the mean
    of the teammates already rated minus new_gap, or from the team's HLTV rank when the whole five is new. The update
    mixes map win and round share (margin) and is larger while a player has few maps.

    requests: match_id, snap (time), lineup_a, lineup_b, rank_a, rank_b. Returns per request the map-level rating gap
    and how many of each five the rating already knew, plus each map's pre-map residual (team A's result target minus
    its expected score; NaN for maps without two fives)."""
    rating: dict[int, float] = {}
    nmaps: dict[int, int] = {}
    last_seen: dict[int, pd.Timestamp] = {}
    dev: dict[int, float] = {}
    req = requests.sort_values("snap").reset_index(drop=True)
    snaps = req.snap.to_numpy()
    out = np.full((len(req), 5), np.nan)

    dev2: dict[int, float] = {}
    fast: dict[int, float] = {}

    def strength(p):
        return rating[p] + cfg.ind_weight * dev.get(p, 0.0) + cfg.swing_weight * dev2.get(p, 0.0) + fast.get(p, 0.0)
    ri = 0

    def team_rating(five, rank, now):
        known = [strength(p) for p in five if p in rating]
        base = (np.mean(known) - cfg.new_gap) if known else cfg.rank_scale * (np.log(UNRANKED) - np.log(rank if rank and rank == rank else UNRANKED)) - 1.0
        vals = []
        for p in five:
            if p in rating:
                age = (now - last_seen[p]).days if now is not None else 0
                shrink = 0.5 ** (max(age - 60, 0) / cfg.decay_days)
                vals.append(strength(p) * shrink + base * (1 - shrink) if age > 60 else strength(p))
            else:
                vals.append(base)
        return float(np.mean(vals)) if vals else base, len(known)

    req_rows = list(req.itertuples(index=False))

    def answer(upto):
        nonlocal ri
        while ri < len(req) and snaps[ri] <= upto:
            r = req_rows[ri]
            now = pd.Timestamp(r.snap)
            a, ka = team_rating(r.lineup_a, r.rank_a, now)
            b, kb = team_rating(r.lineup_b, r.rank_b, now)
            out[ri] = [a - b, ka, kb, min(nmaps.get(p, 0) for p in r.lineup_a) if r.lineup_a else 0,
                       min(nmaps.get(p, 0) for p in r.lineup_b) if r.lineup_b else 0]
            ri += 1

    times = maps.time.to_numpy()
    fa = [_ids(v) for v in maps.team_a_player_ids]
    fb = [_ids(v) for v in maps.team_b_player_ids]
    sa, sb = maps.team_a_score.to_numpy(float), maps.team_b_score.to_numpy(float)
    won = maps.team_a_won.to_numpy(float)
    rka, rkb = maps.team_a_rank.to_numpy(float), maps.team_b_rank.to_numpy(float)
    faceit = (maps.src == "faceit").to_numpy() if "src" in maps else np.zeros(len(maps), bool)
    msid = maps.mapstatsid.to_numpy() if "mapstatsid" in maps else np.full(len(maps), None)
    low = maps.low.fillna(False).to_numpy(bool) if "low" in maps else np.zeros(len(maps), bool)
    lanm = maps.is_lan.fillna(0).to_numpy(float) > 0 if "is_lan" in maps else np.zeros(len(maps), bool)
    resid = np.full(len(maps), np.nan)
    for i in range(len(maps)):
        answer(times[i])
        A, B = fa[i], fb[i]
        if len(A) < 4 or len(B) < 4 or (faceit[i] and cfg.k_faceit <= 0):
            continue
        kmult = (cfg.k_faceit * (cfg.k_faceit_low if low[i] else 1.0)) if faceit[i] else (cfg.k_lan if lanm[i] else 1.0)
        now = pd.Timestamp(times[i])
        for five, rank in ((A, rka[i]), (B, rkb[i])):
            new = [p for p in five if p not in rating]
            if new:
                base, _ = team_rating(tuple(p for p in five if p in rating), rank, now)
                if not any(p in rating for p in five):
                    base = cfg.rank_scale * (np.log(UNRANKED) - np.log(rank if rank == rank else UNRANKED)) - 1.0
                    if faceit[i]:
                        base += cfg.faceit_offset
                else:
                    base -= cfg.new_gap
                for p in new:
                    rating[p] = base
                    nmaps[p] = 0
        ra = np.mean([strength(p) for p in A])
        rb = np.mean([strength(p) for p in B])
        e = 1 / (1 + np.exp(-(ra - rb)))
        tot = sa[i] + sb[i]
        share = sa[i] / tot if tot > 0 else won[i]
        target = (1 - cfg.margin) * won[i] + cfg.margin * np.clip(0.5 + 2.5 * (share - 0.5), 0, 1)
        delta = target - e
        resid[i] = delta
        for five, sgn in ((A, 1.0), (B, -1.0)):
            for p in five:
                k = cfg.k + (cfg.k_new - cfg.k) * np.exp(-nmaps[p] / cfg.n_settle)
                rating[p] += sgn * kmult * k * delta
                if cfg.k_fast > 0:
                    fast[p] = (1 - cfg.fast_decay) * fast.get(p, 0.0) + sgn * kmult * cfg.k_fast * delta
                nmaps[p] += 1
                last_seen[p] = now
        if cfg.rank_pull > 0:
            for five, rank in ((A, rka[i]), (B, rkb[i])):
                if rank == rank:
                    gap = cfg.rank_scale * (np.log(UNRANKED) - np.log(rank)) - 1.0 - np.mean([rating[p] for p in five])
                    for p in five:
                        rating[p] += cfg.rank_pull * gap
        st = stats.get(msid[i]) if stats is not None else None
        if st:
            for five in (A, B):
                rs = [st[p] for p in five if p in st]
                if len(rs) >= 4:
                    m = np.mean(rs, axis=0)
                    for p in five:
                        if p in st:
                            dev[p] = (1 - cfg.ind_alpha) * dev.get(p, 0.0) + cfg.ind_alpha * (st[p][0] - m[0])
                            dev2[p] = (1 - cfg.ind_alpha) * dev2.get(p, 0.0) + cfg.ind_alpha * (st[p][1] - m[1])
    answer(np.datetime64("2262-01-01"))
    res = pd.DataFrame(out, columns=["elo_gap", "known_a", "known_b", "minmaps_a", "minmaps_b"])
    res["match_id"] = req.match_id.to_numpy()
    return res, resid


@dataclass(frozen=True)
class MarketRatingConfig:
    obs_var: float = 0.15
    init_var: float = 1.0
    drift_per_day: float = 0.004
    roster_var: float = 0.3


def market_rating_pass(closes: pd.DataFrame, requests: pd.DataFrame, cfg: MarketRatingConfig = MarketRatingConfig()) -> pd.DataFrame:
    """Team ratings implied by past series closes: a Kalman filter on close_logit = r_A - r_B + noise, with ratings
    drifting over time. A team whose announced five departs from the five that set its rating gets extra variance.

    closes: match_id, kickoff, team_a, team_b, z (closing logit), five_a, five_b.
    requests: match_id, snap, team_a, team_b, lineup_a, lineup_b."""
    mu: dict[int, float] = {}
    var: dict[int, float] = {}
    seen: dict[int, pd.Timestamp] = {}
    fives: dict[int, frozenset] = {}
    req = requests.sort_values("snap").reset_index(drop=True)
    snaps = req.snap.to_numpy()
    out = np.full((len(req), 3), np.nan)
    ri = 0

    def state(team, now, five):
        if team not in mu:
            return 0.0, cfg.init_var, 0.0
        v = var[team] + cfg.drift_per_day * max((now - seen[team]).total_seconds() / 86400.0, 0.0)
        overlap = len(fives[team] & frozenset(five)) if five else 5
        v += cfg.roster_var * max(4 - overlap, 0)
        return mu[team], v, 1.0

    req_rows = list(req.itertuples(index=False))

    def answer(upto):
        nonlocal ri
        while ri < len(req) and snaps[ri] <= upto:
            r = req_rows[ri]
            now = pd.Timestamp(r.snap)
            ma, va, ha = state(r.team_a, now, r.lineup_a)
            mb, vb, hb = state(r.team_b, now, r.lineup_b)
            out[ri] = [ma - mb, va + vb, ha * hb]
            ri += 1

    c = closes.sort_values("kickoff").reset_index(drop=True)
    for r in c.itertuples(index=False):
        answer(np.datetime64(r.kickoff))
        now = pd.Timestamp(r.kickoff)
        ma, va, _ = state(r.team_a, now, r.five_a)
        mb, vb, _ = state(r.team_b, now, r.five_b)
        s = va + vb + cfg.obs_var
        e = r.z - (ma - mb)
        mu[r.team_a] = ma + va / s * e
        mu[r.team_b] = mb - vb / s * e
        var[r.team_a] = va - va * va / s
        var[r.team_b] = vb - vb * vb / s
        seen[r.team_a] = seen[r.team_b] = now
        if r.five_a:
            fives[r.team_a] = frozenset(r.five_a)
        if r.five_b:
            fives[r.team_b] = frozenset(r.five_b)
    answer(np.datetime64("2262-01-01"))
    res = pd.DataFrame(out, columns=["mrat", "mrat_var", "mrat_has"])
    res["match_id"] = req.match_id.to_numpy()
    return res


def build_rows(data_dir: str | Path, cutoff: pd.Timestamp, targets: pd.DataFrame,
               elo_cfg: EloConfig = EloConfig(), mr_cfg: MarketRatingConfig = MarketRatingConfig(),
               train_gap_hours: float = 4.0, train_days: float = 400.0,
               extra_elo: dict[str, EloConfig] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature rows for the past bo3 series (training, with outcome y; market columns where the series had a
    market) and for the targets, from data before cutoff plus the targets' own lineups and paths."""
    maps = load_maps(data_dir, cutoff)
    outcomes = series_outcomes(maps)
    past_paths = load(data_dir, "market/series_paths", cutoff)
    past_paths = past_paths[naive(past_paths.kickoff) < cutoff - pd.Timedelta(hours=train_gap_hours)]
    ps = path_summary(past_paths)
    first = maps.drop_duplicates("match_id").set_index("match_id")
    tr = outcomes[(outcomes.bo == 3) & outcomes.y.notna() & (outcomes.time < cutoff - pd.Timedelta(hours=train_gap_hours))
                  & (outcomes.time >= cutoff - pd.Timedelta(days=train_days))][["y", "time"]]
    tr = tr.rename(columns={"time": "kickoff"}).join(ps, how="left")
    tr = tr.join(first[["team_a_id", "team_b_id", "team_a_lineup_ids", "team_b_lineup_ids", "team_a_player_ids",
                        "team_b_player_ids", "team_a_rank", "team_b_rank", "is_lan", "event_tier"]])
    tr = tr.rename_axis("match_id").reset_index()
    tr["snap"] = week_start(tr.kickoff)
    tr["lineup_a"] = [_ids(l) or _ids(p) for l, p in zip(tr.team_a_lineup_ids, tr.team_a_player_ids)]
    tr["lineup_b"] = [_ids(l) or _ids(p) for l, p in zip(tr.team_b_lineup_ids, tr.team_b_player_ids)]
    lu = load(data_dir, "hltv/cs2_match_lineups", cutoff, ["match_date", "match_id", "team_id", "is_standin"])
    subs = lu[lu.is_standin.fillna(False).astype(bool)].groupby(["match_id", "team_id"]).size()
    tr["sub_a"] = [float(subs.get((m, t), 0)) for m, t in zip(tr.match_id, tr.team_a_id)]
    tr["sub_b"] = [float(subs.get((m, t), 0)) for m, t in zip(tr.match_id, tr.team_b_id)]

    tg = targets.copy()
    tp = path_summary(target_paths(data_dir, tg.match_id))
    tg = tg.merge(tp, left_on="match_id", right_index=True, how="left")
    tg["snap"] = cutoff
    tg["lineup_a"] = [_ids(v) for v in tg.team_a_lineup_ids]
    tg["lineup_b"] = [_ids(v) for v in tg.team_b_lineup_ids]
    tg["sub_a"] = [len(_ids(v)) for v in tg.team_a_standin_ids]
    tg["sub_b"] = [len(_ids(v)) for v in tg.team_b_standin_ids]
    rank = latest_ranks(data_dir, cutoff)
    tg["team_a_rank"] = tg.team_a_id.map(rank)
    tg["team_b_rank"] = tg.team_b_id.map(rank)
    tg["kickoff"] = naive(tg.match_time)

    cols = ["match_id", "snap", "kickoff", "team_a_id", "team_b_id", "lineup_a", "lineup_b", "team_a_rank", "team_b_rank",
            "is_lan", "event_tier", "sub_a", "sub_b", "m0", "tw1", "tw3", "tw6", "nchg3", "nprint", "zlast"]
    rows = pd.concat([tr[cols + ["y"]].assign(train=True), tg[cols].assign(train=False, y=np.nan)], ignore_index=True)
    rows["key"] = np.arange(len(rows))
    req = rows[["key", "snap", "lineup_a", "lineup_b", "team_a_rank", "team_b_rank"]].rename(
        columns={"key": "match_id", "team_a_rank": "rank_a", "team_b_rank": "rank_b"})
    fm = load_faceit_maps(data_dir, cutoff) if any(c.k_faceit > 0 for c in [elo_cfg, *(extra_elo or {}).values()]) else None
    cm = combined_maps(maps, fm if elo_cfg.k_faceit > 0 else None)
    e, resid_all = elo_pass(cm, req, elo_cfg, player_stats(data_dir, cutoff, elo_cfg))
    e = e.rename(columns={"match_id": "key"})
    for name, cfg in (extra_elo or {}).items():
        x, _ = elo_pass(combined_maps(maps, fm if cfg.k_faceit > 0 else None), req, cfg, player_stats(data_dir, cutoff, cfg))
        e = e.merge(x[["match_id", "elo_gap"]].rename(columns={"match_id": "key", "elo_gap": f"elo_gap_{name}"}), on="key")
    resid = np.full(len(maps), np.nan)
    hidx = cm.hidx.to_numpy()
    resid[hidx[hidx >= 0]] = resid_all[hidx >= 0]
    tf = team_features(maps, resid, rows)

    closes = tr[["match_id", "kickoff", "team_a_id", "team_b_id", "tw3", "lineup_a", "lineup_b"]].assign(kickoff=lambda d: d.kickoff.astype("datetime64[ns]")).rename(
        columns={"team_a_id": "team_a", "team_b_id": "team_b", "tw3": "z", "lineup_a": "five_a", "lineup_b": "five_b"}).dropna(subset=["z"])
    mreq = rows[["key", "snap", "team_a_id", "team_b_id", "lineup_a", "lineup_b"]].rename(
        columns={"key": "match_id", "team_a_id": "team_a", "team_b_id": "team_b"})
    m = market_rating_pass(closes, mreq, mr_cfg).rename(columns={"match_id": "key"})
    rows = rows.merge(e, on="key").merge(m, on="key").merge(tf, on="key").sort_values("key").reset_index(drop=True)
    rk = load(data_dir, "hltv/cs2_team_rankings", cutoff, ["date", "team_id", "rank"])
    rk["date"] = naive(rk.date)
    rk = rk.sort_values("date")
    for side in ("a", "b"):
        q = rows[["key", "snap", f"team_{side}_id"]].rename(columns={f"team_{side}_id": "team_id"}).sort_values("snap")
        q["snap"] = q.snap.astype(rk.date.dtype)
        got = pd.merge_asof(q, rk, left_on="snap", right_on="date", by="team_id", allow_exact_matches=False)
        got = got[(got.snap - got.date) <= pd.Timedelta(days=60)]
        rows[f"hrank_{side}"] = rows.key.map(got.set_index("key")["rank"]).astype(float)
    return rows[rows.train].reset_index(drop=True), rows[~rows.train].reset_index(drop=True)


def latest_ranks(data_dir: str | Path, cutoff: pd.Timestamp) -> pd.Series:
    r = load(data_dir, "hltv/cs2_team_rankings", cutoff)
    if r.empty:
        return pd.Series(dtype=float)
    r = r.sort_values("date")
    return r.groupby("team_id")["rank"].last().astype(float)
