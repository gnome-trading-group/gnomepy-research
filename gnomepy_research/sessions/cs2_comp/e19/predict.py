"""
Competitor e19: prices every target series from player-skill filters run over HLTV and FACEIT maps.

    python -m gnomepy_research.sessions.cs2_comp.e19.predict --data <bundle> --targets <parquet> --out <parquet>
        [--no-faceit] [--weeks 2026-04-06,2026-04-13] [--dev-out <parquet> --dev-from 2025-12-01] [--faceit-weight W] ...

Every map before a target's week cutoff is one binomial observation of the round-logit gap between the ten players
who played it; a filter (filter.py) updates the players' Gaussian skills map by map in time order. A target is
priced from the state at its cutoff with each team's most recent five, because the announced lineup is not in
the data yet on the Monday. The state at a cutoff depends only on maps dated before it, so one chronological pass
with a snapshot at every cutoff is the same as a separate fit per cutoff.

Three channels, each its own filter and its own measurement of the same matchup:
  hltv    HLTV maps only (the --no-faceit model);
  both    HLTV and FACEIT maps in one stream, FACEIT maps with their own overdispersion and weight, linked players
          under their HLTV id and the rest under their own FACEIT-derived id so FACEIT-only teams connect the graph;
  faceit  FACEIT maps only, zeroed for a series where either team has no FACEIT five (coverage indicator).
FACEIT maps join after the bundle's cleaning rules (exclusions, 5v5, decided maps), and a FACEIT-only debutant
starts from the prior of the division it debuted in.

Four more channels are switchable in FilterConfig and off by default, because none moved the stacked dev log loss:
  ridge   a trailing-window ridge refit of player and team strengths every Monday (ridge.py);
  tridge  the same with no players, a decayed team Bradley-Terry;
  rank    the gap in log(1 + HLTV ranking points);
  rating  per-player map-rating residuals against the filter's pre-map expectation.

Stacking and calibration: every Monday the week's HLTV bo3 series are priced one step ahead by every channel; a
target's cutoff fits a no-intercept logistic regression of the outcome on the channels' logits over the previous
26 weeks of those prices, all dated before the cutoff. No intercept keeps the price symmetric in A and B. With
--dev-out the weekly channel prices are written for diagnostics on data before the tune period.
"""
from __future__ import annotations

import argparse
import collections
import math
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e19.filter import FACEIT, HLTV, FilterConfig, RosterMoves, SeriesPricer, \
    SkillFilter, division, placeholder_five, regulars_back
from gnomepy_research.sessions.cs2_comp.e19.ridge import RidgeRater

CALIBRATION_WEEKS = 26
CALIBRATION_MIN = 300
WEEK = pd.Timedelta(days=7)
CHANNELS = ("hltv", "both", "faceit", "ridge", "tridge", "rank", "rating")
FILTER_CHANNELS = {"hltv": (HLTV,), "both": (HLTV, FACEIT), "faceit": (FACEIT,)}
FACEIT_ONLY_CHANNELS = ("both", "faceit")


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def faceit_player_id(faceit_id: str) -> int:
    """A negative id derived from the FACEIT uuid, so it never collides with an HLTV id and needs no table."""
    return -(int(faceit_id.replace("-", "")[:15], 16) + 1)


def standins_by_team(data_dir: Path, before: pd.Timestamp) -> dict[tuple[str, int], tuple[int, ...]]:
    lu = load(data_dir, "hltv/cs2_match_lineups", before, ["match_id", "match_date", "team_id", "player_id", "is_standin"])
    lu = lu[lu.is_standin.fillna(False).astype(bool)]
    out: dict[tuple[str, int], tuple[int, ...]] = {}
    for m, t, p in zip(lu.match_id.astype(str), lu.team_id.astype(int), lu.player_id.astype(int)):
        out[(m, t)] = out.get((m, t), ()) + (p,)
    return out


def hltv_stream(data_dir: Path, before: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", before)
    h = h[~showmatch_mask(h) & (h.team_a_score != h.team_b_score)].copy()
    h["eff"] = np.maximum(_naive(h.match_date), _naive(h.match_time).fillna(_naive(h.match_date)))
    ids = lambda col: [tuple(int(p) for p in v) if v is not None and len(v) == 5 else () for v in h[col]]
    standins = standins_by_team(data_dir, before)
    mids = h.match_id.astype(str)
    return pd.DataFrame({
        "standins_a": [standins.get((m, int(t)), ()) for m, t in zip(mids, h.team_a_id)],
        "standins_b": [standins.get((m, int(t)), ()) for m, t in zip(mids, h.team_b_id)],
        "eff": h.eff.to_numpy(), "day": _naive(h.match_date).dt.normalize().to_numpy(), "source": HLTV,
        "match_id": h.match_id.astype(str).to_numpy(), "pos": h.map_position_in_series.fillna(0).astype(int).to_numpy(),
        "map_name": h.map_name.astype(str).to_numpy(),
        "key_a": h.team_a_id.astype(int).to_numpy(), "key_b": h.team_b_id.astype(int).to_numpy(),
        "hltv_a": h.team_a_id.astype(int).to_numpy(), "hltv_b": h.team_b_id.astype(int).to_numpy(),
        "players_a": ids("team_a_player_ids"), "players_b": ids("team_b_player_ids"),
        "rounds_a": h.team_a_score.astype(int).to_numpy(), "rounds_b": h.team_b_score.astype(int).to_numpy(),
        "won_a": h.team_a_won.astype(int).to_numpy(), "bo_type": h.bo_type.astype(int).to_numpy(), "div": "other",
    })


def faceit_stream(data_dir: Path, before: pd.Timestamp) -> pd.DataFrame:
    fm = load(data_dir, "faceit/faceit_match_history", before)
    m = load(data_dir, "faceit/faceit_matches", before,
             ["match_id", "finished_at", "competition_name", "has_stats", "faction1_hltv_team", "faction2_hltv_team"])
    ex = pd.read_parquet(data_dir / "faceit/faceit_exclusions.parquet")
    whole = set(ex[ex.map_name.isna()].match_id)
    per_map = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
    link = player_link(data_dir)
    fm = fm.merge(m[m.has_stats], on="match_id", how="inner")
    keep = ~fm.match_id.isin(whole) & ~pd.Series([(a, b) in per_map for a, b in zip(fm.match_id, fm.map_name)], index=fm.index) \
        & (fm.team_a_players == 5) & (fm.team_b_players == 5) & (fm.team_a_score != fm.team_b_score) \
        & fm.team_a_faceit_ids.map(lambda v: v is not None and len(v) == 5) & fm.team_b_faceit_ids.map(lambda v: v is not None and len(v) == 5)
    fm = fm[keep].copy()
    finished = pd.to_datetime(fm.finished_at, unit="s")
    fm["eff"] = np.maximum(_naive(fm.match_date), finished)

    def unified(col):
        return [tuple(link.get(f) if f in link else faceit_player_id(f) for f in v) for v in fm[col]]

    def keys(hltv_col, faceit_col):
        return [int(h) if pd.notna(h) else faceit_player_id(f) for h, f in zip(fm[hltv_col], fm[faceit_col])]

    def hltv(col):
        return [int(h) if pd.notna(h) else None for h in fm[col]]

    return pd.DataFrame({
        "standins_a": [()] * len(fm), "standins_b": [()] * len(fm),
        "eff": fm.eff.to_numpy(), "day": _naive(fm.match_date).dt.normalize().to_numpy(), "source": FACEIT,
        "match_id": fm.match_id.astype(str).to_numpy(), "pos": fm.map_position_in_series.astype(int).to_numpy(),
        "map_name": fm.map_name.astype(str).to_numpy(),
        "key_a": keys("faction1_hltv_team", "team_a_faceit_id"), "key_b": keys("faction2_hltv_team", "team_b_faceit_id"),
        "hltv_a": hltv("faction1_hltv_team"), "hltv_b": hltv("faction2_hltv_team"),
        "players_a": unified("team_a_faceit_ids"), "players_b": unified("team_b_faceit_ids"),
        "rounds_a": fm.team_a_score.astype(int).to_numpy(), "rounds_b": fm.team_b_score.astype(int).to_numpy(),
        "won_a": fm.team_a_won.astype(int).to_numpy(), "bo_type": fm.bo_type.astype(int).to_numpy(),
        "div": fm.competition_name.map(division).to_numpy(),
    })


def player_link(data_dir: Path) -> dict[str, int]:
    return {r.faceit_id: int(r.hltv_id) for r in pd.read_parquet(data_dir / "faceit/hltv_faceit_map.parquet").itertuples()}


def map_stats(data_dir: Path, before: pd.Timestamp, stream: pd.DataFrame, faceit: bool) -> dict[tuple[str, str], dict[int, float]]:
    """Per (match, map): each player's rating. HLTV's map rating, and for FACEIT maps the HLTV 2.0 formula from
    kills, deaths, assists and ADR per round, with KAST (not in the FACEIT stats) held at a typical 70%."""
    h = load(data_dir, "hltv/cs2_player_map_stats", before, ["match_id", "match_date", "map_name", "player_id", "side", "rating"])
    h = h[(h.side == "all") & h.rating.notna() & h.match_id.astype(str).isin(set(stream.match_id))]
    frames = [pd.DataFrame({"match_id": h.match_id.astype(str).to_numpy(), "map_name": h.map_name.astype(str).to_numpy(),
                            "pid": h.player_id.astype(int).to_numpy(), "rating": h.rating.astype(float).to_numpy()})]
    if faceit:
        link = player_link(data_dir)
        f = load(data_dir, "faceit/faceit_player_map_stats", before,
                 ["match_id", "map_name", "faceit_id", "kills", "deaths", "assists", "adr"])
        rounds = stream[stream.source == FACEIT][["match_id", "map_name", "rounds_a", "rounds_b"]]
        f = f.merge(rounds.assign(rounds=rounds.rounds_a + rounds.rounds_b)[["match_id", "map_name", "rounds"]], on=["match_id", "map_name"])
        f = f[(f.rounds > 0) & f.kills.notna() & f.deaths.notna() & f.adr.notna()]
        kpr, dpr, apr = f.kills / f.rounds, f.deaths / f.rounds, f.assists.fillna(0) / f.rounds
        impact = 2.13 * kpr + 0.42 * apr - 0.41
        rating = 0.0073 * 70 + 0.3591 * kpr - 0.5329 * dpr + 0.2372 * impact + 0.0032 * f.adr + 0.1587
        frames.append(pd.DataFrame({"match_id": f.match_id.astype(str).to_numpy(), "map_name": f.map_name.astype(str).to_numpy(),
                                    "pid": [link.get(x) if x in link else faceit_player_id(x) for x in f.faceit_id],
                                    "rating": rating.astype(float).to_numpy()}))
    allf = pd.concat(frames, ignore_index=True)
    out: dict[tuple[str, str], dict[int, float]] = {}
    for m, n, p, r in zip(allf.match_id, allf.map_name, allf.pid, allf.rating):
        out.setdefault((m, n), {})[p] = r
    return out


def build_stream(data_dir: Path, before: pd.Timestamp, faceit: bool) -> pd.DataFrame:
    parts = [hltv_stream(data_dir, before)]
    if faceit:
        parts.append(faceit_stream(data_dir, before))
    s = pd.concat(parts, ignore_index=True)
    s = s[s.eff < before]
    return s.sort_values(["eff", "match_id", "pos"], kind="mergesort").reset_index(drop=True)


def hltv_series(stream: pd.DataFrame) -> pd.DataFrame:
    """One row per HLTV bo3 series: kickoff, teams and outcome, for one-step-ahead pricing."""
    h = stream[(stream.source == HLTV) & (stream.bo_type == 3)]
    g = h.groupby("match_id", sort=False)
    s = pd.DataFrame({"eff": g.eff.min(), "key_a": g.key_a.first(), "key_b": g.key_b.first(),
                      "maps": g.won_a.size(), "won": g.won_a.sum()})
    s["y"] = (s.won * 2 > s.maps).astype(int)
    return s[s.maps >= 2].reset_index().sort_values("eff").reset_index(drop=True)


def monday(t: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t).normalize()
    return t - pd.Timedelta(days=t.weekday())


def expected_five(f: SkillFilter, key: int, at: pd.Timestamp, moves: RosterMoves | None) -> tuple[int, ...] | None:
    five = f.last_five.get(key)
    if five is None:
        return None
    if f.cfg.standin_fix:
        five = regulars_back(f, key, five)
    if moves is not None and key > 0:
        five = moves.adjust(key, five, f.last_five_day[key], at)
    return five


def price_matchup(f: SkillFilter, pricer: SeriesPricer, key_a: int, key_b: int, at: pd.Timestamp,
                  moves: RosterMoves | None = None) -> tuple[float, bool]:
    """The channel's series price and whether both teams had a five in its stream."""
    five_a, five_b = expected_five(f, key_a, at, moves), expected_five(f, key_b, at, moves)
    covered = five_a is not None and five_b is not None
    a = f.lineup(five_a or placeholder_five(key_a), key_a, key_a if key_a > 0 else None, at)
    b = f.lineup(five_b or placeholder_five(key_b), key_b, key_b if key_b > 0 else None, at)
    d, s2, _, _ = f.gap(a, b)
    return pricer.series(d, s2), covered


def price_ridge(rater: RidgeRater, f: SkillFilter, pricer: SeriesPricer, key_a: int, key_b: int, at: pd.Timestamp,
                moves: RosterMoves | None = None) -> tuple[float, bool]:
    """The ridge channel's series price from the main filter's expected fives and the latest refit."""
    five_a, five_b = expected_five(f, key_a, at, moves), expected_five(f, key_b, at, moves)
    covered = five_a is not None and five_b is not None
    sa = rater.strength(five_a or placeholder_five(key_a), key_a, key_a if key_a > 0 else None, at)
    sb = rater.strength(five_b or placeholder_five(key_b), key_b, key_b if key_b > 0 else None, at)
    return pricer.series(sa - sb, 0.0), covered


def price_rank(f: SkillFilter, key_a: int, key_b: int, at: pd.Timestamp) -> tuple[float, bool]:
    """A price whose logit is the gap in log(1 + ranking points) at the latest ranking before the cutoff; the stack
    scales it. A team without an HLTV id has no points."""
    pa = math.log1p(f.points(key_a, at)) if key_a > 0 else 0.0
    pb = math.log1p(f.points(key_b, at)) if key_b > 0 else 0.0
    return 1 / (1 + math.exp(-(pa - pb))), True


def update_residuals(resid: dict[int, float], cfg: FilterConfig, gap: float, team_stats: list) -> None:
    for sign, (five, found) in zip((1.0, -1.0), team_stats):
        expected = 1.0 + cfg.rating_slope * sign * gap
        for p, rating in found.items():
            resid[p] = (1 - cfg.rating_alpha) * resid.get(p, 0.0) + cfg.rating_alpha * (rating - expected)


def price_rating(resid: dict[int, float], cfg: FilterConfig, f: SkillFilter, key_a: int, key_b: int, at: pd.Timestamp,
                 moves: RosterMoves | None = None) -> tuple[float, bool]:
    """A price whose logit is rating_kappa times the gap in the fives' mean rating residuals; the stack scales it."""
    five_a, five_b = expected_five(f, key_a, at, moves), expected_five(f, key_b, at, moves)
    covered = five_a is not None and five_b is not None
    ra = sum(resid.get(p, 0.0) for p in five_a) / len(five_a) if five_a else 0.0
    rb = sum(resid.get(p, 0.0) for p in five_b) / len(five_b) if five_b else 0.0
    return 1 / (1 + math.exp(-cfg.rating_kappa * (ra - rb))), covered


def logit(q):
    q = np.clip(np.asarray(q, float), 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def features(rows: pd.DataFrame, channels: tuple[str, ...], interaction: bool = False) -> np.ndarray:
    cols = []
    for c in channels:
        x = logit(rows[f"q_{c}"])
        if c == "faceit":
            x = x * rows["cov_faceit"].to_numpy(float)
        cols.append(x)
    if interaction:
        thin = rows["thin"].to_numpy(float)
        cols += [x * thin for x in cols]
    return np.column_stack(cols)


def fit_stack(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """No-intercept logistic regression by Newton steps with a light ridge, starting from the first channel alone."""
    w = np.zeros(X.shape[1])
    w[0] = 1.0

    def loss(v):
        eta = X @ v
        return (y * np.logaddexp(0.0, -eta) + (1 - y) * np.logaddexp(0.0, eta)).sum() + 0.5e-3 * (v @ v)

    current = loss(w)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - 1e-3 * w
        h = X.T @ (X * (p * (1 - p))[:, None]) + 1e-3 * np.eye(X.shape[1])
        step = np.linalg.solve(h, g)
        # A full Newton step overshoots when a channel is wild in some week; halve until the loss drops.
        for _ in range(20):
            candidate = loss(w + step)
            if candidate <= current:
                break
            step = step / 2
        w += step
        current = candidate
        if np.abs(step).max() < 1e-10:
            break
    return w


def apply_stack(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(X @ w)))


def calibrate(weekly: pd.DataFrame, at: pd.Timestamp, channels: tuple[str, ...], weeks: int = CALIBRATION_WEEKS,
              interaction: bool = False) -> np.ndarray:
    """Stack weights from the weekly prices of the `weeks` before `at`, all dated before it."""
    cal = weekly[(weekly.week >= at - weeks * WEEK) & (weekly.week < at) & (weekly.eff < at)]
    if len(cal) < CALIBRATION_MIN:
        return np.eye(len(channels) * (2 if interaction else 1))[0]
    return fit_stack(features(cal, channels, interaction), cal.y.to_numpy(float))


def run(data_dir: Path, targets: pd.DataFrame, cfg: FilterConfig, faceit: bool, dev_from: pd.Timestamp | None = None,
        log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoffs = sorted(pd.Timestamp(c) for c in targets.week_cutoff.unique())
    last = cutoffs[-1]
    t0 = time.time()
    stream = build_stream(data_dir, last, faceit)
    series = hltv_series(stream)
    log(f"stream {len(stream)} maps ({int((stream.source == FACEIT).sum())} faceit), {len(series)} hltv bo3 series, "
        f"{time.time() - t0:.0f}s")
    rankings = load(data_dir, "hltv/cs2_team_rankings", last)
    enabled = {"ridge": cfg.ridge, "tridge": cfg.team_ridge, "rank": cfg.rank_channel, "rating": cfg.rating_channel}
    channels = tuple(c for c in CHANNELS if (faceit or c not in FACEIT_ONLY_CHANNELS) and enabled.get(c, True))
    takes = FILTER_CHANNELS
    filters = {c: SkillFilter(cfg, rankings) for c in channels if c in takes}
    main = "both" if faceit else "hltv"
    raters = {}
    if cfg.ridge:
        raters["ridge"] = RidgeRater(stream, cfg, takes[main], filters[main].prior)
    if cfg.team_ridge:
        team_cfg = replace(cfg, ridge_players=False, ridge_team=True, ridge_lam_team=cfg.team_ridge_lam)
        raters["tridge"] = RidgeRater(stream, team_cfg, takes[main], filters[main].prior)
    for name, r in raters.items():
        log(f"{name} {r.X.shape[0]} maps, {r.n_params} parameters, {time.time() - t0:.0f}s")
    pricer = SeriesPricer(cfg)
    moves = RosterMoves(load(data_dir, "hltv/cs2_roster_moves", last)) if cfg.roster_moves else None
    stats = map_stats(data_dir, last, stream, faceit) if cfg.rating_channel else {}
    if stats:
        log(f"{len(stats)} player-map ratings, {time.time() - t0:.0f}s")
    resid: dict[int, float] = {}

    first_week = monday(cutoffs[0] - CALIBRATION_WEEKS * WEEK)
    if dev_from is not None:
        first_week = min(first_week, monday(dev_from))
    mondays = list(pd.date_range(first_week, last, freq=WEEK))
    by_week = {m: series[(series.eff >= m) & (series.eff < m + WEEK)] for m in mondays}
    targets_by_cutoff = {c: targets[targets.week_cutoff == c] for c in cutoffs}
    weekly_rows: list[dict] = []
    out: list[dict] = []
    hltv_days: dict[int, collections.deque] = collections.defaultdict(collections.deque)

    def thin(key_a: int, key_b: int, at: pd.Timestamp) -> int:
        since = at - pd.Timedelta(days=cfg.thin_days)
        counts = [sum(1 for d in hltv_days.get(k, ()) if d >= since) for k in (key_a, key_b)]
        unranked = all(filters[main].points(k, at) <= 0 for k in (key_a, key_b))
        return int(min(counts) <= cfg.thin_maps or unranked)

    def priced(key_a: int, key_b: int, at: pd.Timestamp) -> dict:
        row = {"thin": thin(key_a, key_b, at)}
        for c in channels:
            if c in raters:
                row[f"q_{c}"], row[f"cov_{c}"] = price_ridge(raters[c], filters[main], pricer, key_a, key_b, at, moves)
            elif c == "rank":
                row["q_rank"], row["cov_rank"] = price_rank(filters[main], key_a, key_b, at)
            elif c == "rating":
                row["q_rating"], row["cov_rating"] = price_rating(resid, cfg, filters[main], key_a, key_b, at, moves)
            else:
                row[f"q_{c}"], row[f"cov_{c}"] = price_matchup(filters[c], pricer, key_a, key_b, at, moves)
        return row

    def snapshot(m: pd.Timestamp) -> None:
        for r in raters.values():
            r.fit(m)
        for r in by_week[m].itertuples():
            weekly_rows.append({"match_id": r.match_id, "week": m, "eff": r.eff, "y": r.y, **priced(r.key_a, r.key_b, m)})
        if m not in targets_by_cutoff:
            return
        weekly = pd.DataFrame(weekly_rows)
        w = calibrate(weekly, m, channels, interaction=cfg.thin_interaction)
        rows = pd.DataFrame([{"match_id": r.match_id, **priced(int(r.team_a_id), int(r.team_b_id), m)}
                             for r in targets_by_cutoff[m].itertuples()])
        rows["q"] = apply_stack(features(rows, channels, cfg.thin_interaction), w)
        rows["fit_cutoff"] = m
        out.append(rows)
        log(f"{m:%Y-%m-%d} targets {len(rows)} stack {np.round(w, 3).tolist()} {time.time() - t0:.0f}s")

    eff = stream.eff.to_numpy()
    rows = stream.itertuples(index=False)
    i = 0
    row = next(rows, None)
    for m in mondays:
        while row is not None and eff[i] < m:
            team_stats = []
            if stats:
                found = stats.get((row.match_id, row.map_name), {})
                for five in (row.players_a, row.players_b):
                    team_stats.append((five, {p: found[p] for p in five if p in found}))
            for c in filters:
                if row.source in takes[c]:
                    filters[c].observe_map(pd.Timestamp(row.day), row.key_a, row.key_b, row.hltv_a, row.hltv_b,
                                           row.players_a, row.players_b, row.rounds_a, row.rounds_b, row.source, row.div,
                                           row.standins_a, row.standins_b)
            if cfg.rating_channel and filters[main].last_gap is not None and row.source in takes[main]:
                update_residuals(resid, cfg, filters[main].last_gap, team_stats)
            if row.source == HLTV:
                day = pd.Timestamp(row.day)
                for k in (row.key_a, row.key_b):
                    q = hltv_days[k]
                    q.append(day)
                    while q and q[0] < day - pd.Timedelta(days=cfg.thin_days):
                        q.popleft()
            i += 1
            row = next(rows, None)
        snapshot(m)
    return pd.concat(out, ignore_index=True), pd.DataFrame(weekly_rows)


KNOBS = ("faceit_weight", "faceit_low_div_mult", "faceit_rho", "faceit_sigma0", "faceit_prior_shift", "sigma_map",
         "sigma_series", "tau", "rho", "team_sigma0", "team_tau", "ridge_window_weeks",
         "ridge_half_life_weeks", "ridge_lam_player", "ridge_lam_team", "ridge_faceit_weight", "team_ridge_lam",
         "rating_alpha", "rating_slope", "rating_kappa")
KNOB_TYPES = {f.name: f.type for f in fields(FilterConfig)}


def config_from_args(a: argparse.Namespace) -> FilterConfig:
    cfg = FilterConfig()
    for k in KNOBS:
        v = getattr(a, k, None)
        if v is not None:
            cfg = replace(cfg, **{k: int(v) if KNOB_TYPES[k] == "int" else v})
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-faceit", action="store_true")
    ap.add_argument("--weeks", help="comma-separated week cutoffs to restrict the targets to")
    ap.add_argument("--dev-out", type=Path, help="write the weekly one-step-ahead HLTV series channel prices")
    ap.add_argument("--dev-from", help="first week of the weekly prices to write")
    for k in KNOBS:
        ap.add_argument(f"--{k.replace('_', '-')}", type=float)
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = _naive(targets.week_cutoff)
    if a.weeks:
        targets = targets[targets.week_cutoff.isin([pd.Timestamp(w) for w in a.weeks.split(",")])]
    cfg = config_from_args(a)
    preds, weekly = run(a.data, targets, cfg, faceit=not a.no_faceit, dev_from=pd.Timestamp(a.dev_from) if a.dev_from else None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    preds[["match_id", "q", "fit_cutoff"]].to_parquet(a.out, index=False)
    if a.dev_out:
        weekly.to_parquet(a.dev_out, index=False)
    print(f"wrote {len(preds)} predictions to {a.out}", flush=True)


if __name__ == "__main__":
    main()
