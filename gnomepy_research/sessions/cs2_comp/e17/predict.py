"""Predict P(team A wins) for the target series, one fit per week cutoff, from data before that cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.e17.predict --data <bundle dir> --targets <targets parquet> --out <parquet>
        [--no-faceit] [--faceit-weight W] [--faceit-prior-offset A] [--faceit-sigma0 S] [--calibration-months M]
        [--calibration pooled|coverage] [--no-extra-features] [--filter NAME=VALUE ...]

The map stream (HLTV, plus FACEIT unless --no-faceit) is consumed chronologically by an HLTV-only filter and a unified
filter. At each Monday every series of the coming week is priced from the state at that Monday: a team's lineup is the
five it last fielded, on HLTV or FACEIT, before the cutoff. Past weeks' prices of bo3 HLTV series and their outcomes
train a logistic calibration on the HLTV-only price, the ranking-points gap, a team Elo gap, the lineups' recent HLTV
rating gap, their recent FACEIT kills-per-round gap and FACEIT's shift of the price by coverage bucket (the HLTV-only
features alone without FACEIT), refit for each target week on the trailing year.
Everything is deterministic, so a rerun on a bundle cut at one week's cutoff reproduces that week.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import PlayerSkillConfig
from gnomepy_research.sessions.cs2_comp.e17.data import (FACEIT, HLTV, faceit_performance, hltv_maps, hltv_series, logit, map_stream, monday,
                                                      player_ratings, rankings)
from gnomepy_research.sessions.cs2_comp.e17.skill import SkillFilter, SourceConfig

UNKNOWN_FIVE = tuple(-(1 << 62) - k for k in range(1, 6))
MIN_CALIBRATION_ROWS = 200
RIDGE = 1e-3
RATING_MAPS = 10
RATING_SHRINK_MAPS = 5
ELO_K = 20.0


class TeamElo:
    """A plain per-team Elo over HLTV maps: team-level strength beyond the five players, for the calibration."""

    def __init__(self):
        self._r: dict[int, float] = {}

    def gap(self, a: int, b: int) -> float:
        return (self._r.get(a, 1500.0) - self._r.get(b, 1500.0)) / 400.0

    def observe(self, a: int | None, b: int | None, rounds_a: int, rounds_b: int) -> None:
        if a is None or b is None or a == b:
            return
        ra, rb = self._r.get(a, 1500.0), self._r.get(b, 1500.0)
        won = 1.0 if rounds_a > rounds_b else 0.0 if rounds_a < rounds_b else 0.5
        step = ELO_K * (won - 1 / (1 + 10 ** ((rb - ra) / 400)))
        self._r[a], self._r[b] = ra + step, rb - step


def lineup_mean(history: dict, five: tuple[int, ...], cutoff: np.datetime64, prior: float) -> float:
    """Mean over the five of each player's recent per-map values before the cutoff, shrunk toward the prior over the
    first few maps; a player with no history sits at the prior."""
    total = 0.0
    for p in five:
        r = history.get(p)
        if r is None:
            total += prior
            continue
        i = int(np.searchsorted(r[0], cutoff, side="left"))
        last = r[1][max(0, i - RATING_MAPS):i]
        total += (last.sum() + RATING_SHRINK_MAPS * prior) / (len(last) + RATING_SHRINK_MAPS)
    return total / len(five)


class LastLineups:
    """The five each HLTV team last fielded, by source, so a prediction can read a lineup without seeing the series."""

    def __init__(self):
        self._last: dict[tuple[int, int], tuple[pd.Timestamp, tuple[int, ...]]] = {}

    def update(self, source: int, team: int | None, players: tuple[int, ...], t: pd.Timestamp) -> None:
        if team is not None and len(players) == 5:
            self._last[(source, team)] = (t, players)

    def get(self, team: int, sources: tuple[int, ...]) -> tuple[tuple[int, ...], int | None]:
        best = max((self._last[(s, team)] + (s,) for s in sources if (s, team) in self._last), default=None, key=lambda x: x[0])
        return (best[1], best[2]) if best else (UNKNOWN_FIVE, None)


def fit_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + RIDGE * np.eye(X.shape[1]), X.T @ (y - p) - RIDGE * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def _features(f: SkillFilter, lineups: LastLineups, sources: tuple[int, ...], ratings: dict, perf: dict, team_a: int, team_b: int,
              day: pd.Timestamp, cutoff: pd.Timestamp) -> dict:
    five_a, src_a = lineups.get(team_a, sources)
    five_b, src_b = lineups.get(team_b, sources)
    a = f.lineup(five_a, team_a, day, cutoff)
    b = f.lineup(five_b, team_b, day, cutoff)
    d, s2, q = f.series_price(a, b)
    rank = math.log1p(f.points(team_a, cutoff)) - math.log1p(f.points(team_b, cutoff))
    c64 = cutoff.to_datetime64()
    return {"d": d, "s2": s2, "q": q, "rank": rank, "known_a": src_a is not None, "known_b": src_b is not None,
            "cov_a": sum(1 for p in five_a if f.faceit_games[p] > 0), "cov_b": sum(1 for p in five_b if f.faceit_games[p] > 0),
            "rating": lineup_mean(ratings, five_a, c64, 1.0) - lineup_mean(ratings, five_b, c64, 1.0),
            "fperf": lineup_mean(perf, five_a, c64, 0.0) - lineup_mean(perf, five_b, c64, 0.0)}


def filter_config(overrides: list[str] | None) -> PlayerSkillConfig:
    """The fitted filter settings, with `name=value` overrides from the command line."""
    cfg = PlayerSkillConfig()
    for item in overrides or ():
        name, value = item.split("=")
        cfg = dataclasses.replace(cfg, **{name: type(getattr(cfg, name))(value)})
    return cfg


def snapshot(data_dir: str | Path, targets: pd.DataFrame, use_faceit: bool, src: SourceConfig = SourceConfig(),
             cfg: PlayerSkillConfig = PlayerSkillConfig(), log=print) -> pd.DataFrame:
    """Filter the map stream and price every training and target series at its week's Monday: one row each."""
    targets = targets.copy()
    targets["week_cutoff"] = pd.to_datetime(targets.week_cutoff)
    targets["day"] = pd.to_datetime(targets.match_date)
    before = targets.week_cutoff.max()
    t0 = time.time()
    stream = map_stream(data_dir, before, use_faceit)
    ranks = rankings(data_dir, before)
    ratings = player_ratings(data_dir, before)
    perf = faceit_performance(data_dir, before) if use_faceit else {}
    train = hltv_series(hltv_maps(data_dir, before))
    train = train[train.bo_type == 3].copy()
    train["cutoff"] = monday(train.time)
    train["day"] = train.time.dt.normalize()
    log(f"stream {len(stream)} maps ({int((stream.source == FACEIT).sum())} FACEIT), {len(train)} bo3 series, loaded in {time.time() - t0:.0f}s")

    filters = {"h": (SkillFilter(cfg, src, ranks, use_faceit=False), (HLTV,))}
    if use_faceit:
        filters["u"] = (SkillFilter(cfg, src, ranks, use_faceit=True), (HLTV, FACEIT))
    lineups = LastLineups()
    elo = TeamElo()
    weeks = sorted(set(monday(stream.time)) | set(train.cutoff) | set(targets.week_cutoff))
    train_by_week = {c: g for c, g in train.groupby("cutoff")}
    targets_by_week = {c: g for c, g in targets.groupby("week_cutoff")}
    times = stream.time.to_numpy()
    link_times = stream.hltv_time_of_link.to_numpy()
    rows = stream.itertuples(index=False)
    pos = 0
    snapshots: list[dict] = []
    for c in weeks:
        c64 = pd.Timestamp(c).to_datetime64()
        end = int(np.searchsorted(times, c64, side="left"))
        for r in rows if end > pos else ():
            pos += 1
            if r.source == FACEIT and not np.isnat(link_times[pos - 1]) and link_times[pos - 1] < c64:
                continue
            for f, _ in filters.values():
                f.observe_map(r.time.normalize(), r.source, r.team_a, r.team_b, r.players_a, r.players_b, r.rounds_a, r.rounds_b)
            lineups.update(r.source, r.team_a, r.players_a, r.time)
            lineups.update(r.source, r.team_b, r.players_b, r.time)
            if r.source == HLTV:
                elo.observe(r.team_a, r.team_b, r.rounds_a, r.rounds_b)
            if pos >= end:
                break
        for kind, frame in (("train", train_by_week.get(c)), ("target", targets_by_week.get(c))):
            if frame is None:
                continue
            for s in frame.itertuples(index=False):
                team_a, team_b = (int(s.team_a), int(s.team_b)) if kind == "train" else (int(s.team_a_id), int(s.team_b_id))
                row = {"kind": kind, "match_id": s.match_id, "cutoff": c, "y": s.y if kind == "train" else np.nan, "elo": elo.gap(team_a, team_b)}
                for name, (f, sources) in filters.items():
                    row.update({f"{k}_{name}": v for k, v in _features(f, lineups, sources, ratings, perf, team_a, team_b, s.day, c).items()})
                snapshots.append(row)
    snap = pd.DataFrame(snapshots)
    log(f"filtered {pos} maps and priced {len(snap)} series in {time.time() - t0:.0f}s")
    return snap


def run(data_dir: str | Path, targets: pd.DataFrame, use_faceit: bool, src: SourceConfig = SourceConfig(),
        calibration_months: int = 12, calibration: str = "coverage", extra: bool = True, cfg: PlayerSkillConfig = PlayerSkillConfig(),
        log=print) -> pd.DataFrame:
    return calibrate(snapshot(data_dir, targets, use_faceit, src, cfg, log), use_faceit, calibration_months, calibration, extra, log)


def _design(g: pd.DataFrame, use_faceit: bool, calibration: str, extra: bool) -> np.ndarray:
    cols = [logit(g.q_h), g.rank_h.to_numpy(float)]
    if extra:
        cols += [g.elo.to_numpy(float), g.rating_h.to_numpy(float)]
        if use_faceit:
            cols.append(g.fperf_u.to_numpy(float))
    if use_faceit:
        shift = logit(g.q_u) - logit(g.q_h)
        if calibration == "coverage":
            # FACEIT's shift of the price earns a coefficient per coverage bucket. Against a side that has never
            # fielded a lineup in either source the shift only says how far FACEIT play moved the known side off the
            # HLTV newcomer prior's scale, and where a lineup has no FACEIT history it is opponents' spillover, so
            # the calibration can discount it there and keep it where both fives have FACEIT history.
            unknown = ~(g.known_a_u & g.known_b_u).to_numpy(bool)
            cov = np.where(unknown, -1, np.minimum(g.cov_a_u, g.cov_b_u).to_numpy())
            cols += [shift * (cov == -1), shift * (cov == 0), shift * ((cov >= 1) & (cov <= 3)), shift * (cov >= 4)]
        else:
            cols.append(shift)
    return np.column_stack(cols + [np.ones(len(g))])


def calibrate(snap: pd.DataFrame, use_faceit: bool, calibration_months: int, calibration: str = "coverage", extra: bool = True,
              log=print) -> pd.DataFrame:
    train = snap[snap.kind == "train"]
    out = []
    for c, g in snap[snap.kind == "target"].groupby("cutoff"):
        tr = train[(train.cutoff < c) & (train.cutoff >= c - pd.DateOffset(months=calibration_months))]
        if len(tr) >= MIN_CALIBRATION_ROWS:
            w = fit_logistic(_design(tr, use_faceit, calibration, extra), tr.y.to_numpy(float))
            q = 1 / (1 + np.exp(-_design(g, use_faceit, calibration, extra) @ w))
        else:
            w = None
            q = (g.q_u if use_faceit else g.q_h).to_numpy(float)
        log(f"{c:%Y-%m-%d}: {len(g)} targets, calibration on {len(tr)} series, w={None if w is None else np.round(w, 3).tolist()}")
        res = g[["match_id"]].copy()
        res["q"] = np.clip(q, 1e-3, 1 - 1e-3)
        res["fit_cutoff"] = c
        for k in ("q_h", "cov_a_u", "cov_b_u", "known_a_u", "known_b_u", "q_u", "known_a_h", "known_b_h", "elo", "rating_h", "fperf_u"):
            if k in g:
                res[k] = g[k].to_numpy()
        out.append(res)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True, help="output parquet; $OUTPUT_DIR is expanded so a Batch task can write there")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--faceit-weight", type=float, default=SourceConfig.faceit_weight)
    ap.add_argument("--faceit-prior-offset", type=float, default=SourceConfig.faceit_prior_offset)
    ap.add_argument("--faceit-sigma0", type=float, default=SourceConfig.faceit_sigma0)
    ap.add_argument("--calibration-months", type=int, default=12)
    ap.add_argument("--calibration", choices=("pooled", "coverage"), default="coverage")
    ap.add_argument("--no-extra-features", dest="extra", action="store_false", help="drop the team Elo, HLTV rating channel and FACEIT performance channel from the calibration")
    ap.add_argument("--filter", action="append", metavar="NAME=VALUE", help="override a PlayerSkillConfig field, e.g. tau=0.02")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    src = SourceConfig(a.faceit_weight, a.faceit_prior_offset, a.faceit_sigma0)
    pred = run(a.data, targets, a.faceit, src, a.calibration_months, a.calibration, a.extra, filter_config(a.filter))
    pred = targets[["match_id"]].merge(pred, on="match_id", how="left")
    pred["fit_cutoff"] = pd.to_datetime(pred.fit_cutoff)
    out = Path(os.path.expandvars(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out, index=False)
    print(f"wrote {len(pred)} predictions ({int(pred.q.isna().sum())} missing) to {out}")


if __name__ == "__main__":
    main()
