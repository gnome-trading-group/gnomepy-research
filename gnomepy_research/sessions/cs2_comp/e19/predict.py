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

Stacking and calibration: every Monday the week's HLTV bo3 series are priced one step ahead by every channel; a
target's cutoff fits a no-intercept logistic regression of the outcome on the channels' logits over the previous
26 weeks of those prices, all dated before the cutoff. No intercept keeps the price symmetric in A and B. With
--dev-out the weekly channel prices are written for diagnostics on data before the tune period.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e19.filter import FACEIT, HLTV, FilterConfig, RosterMoves, SeriesPricer, \
    SkillFilter, division, placeholder_five

CALIBRATION_WEEKS = 26
CALIBRATION_MIN = 300
WEEK = pd.Timedelta(days=7)
CHANNELS = ("hltv", "both", "faceit")


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def faceit_player_id(faceit_id: str) -> int:
    """A negative id derived from the FACEIT uuid, so it never collides with an HLTV id and needs no table."""
    return -(int(faceit_id.replace("-", "")[:15], 16) + 1)


def hltv_stream(data_dir: Path, before: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", before)
    h = h[~showmatch_mask(h) & (h.team_a_score != h.team_b_score)].copy()
    h["eff"] = np.maximum(_naive(h.match_date), _naive(h.match_time).fillna(_naive(h.match_date)))
    ids = lambda col: [tuple(int(p) for p in v) if v is not None and len(v) == 5 else () for v in h[col]]
    return pd.DataFrame({
        "eff": h.eff.to_numpy(), "day": _naive(h.match_date).dt.normalize().to_numpy(), "source": HLTV,
        "match_id": h.match_id.astype(str).to_numpy(), "pos": h.map_position_in_series.fillna(0).astype(int).to_numpy(),
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
    link = {r.faceit_id: int(r.hltv_id) for r in pd.read_parquet(data_dir / "faceit/hltv_faceit_map.parquet").itertuples()}
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
        "eff": fm.eff.to_numpy(), "day": _naive(fm.match_date).dt.normalize().to_numpy(), "source": FACEIT,
        "match_id": fm.match_id.astype(str).to_numpy(), "pos": fm.map_position_in_series.astype(int).to_numpy(),
        "key_a": keys("faction1_hltv_team", "team_a_faceit_id"), "key_b": keys("faction2_hltv_team", "team_b_faceit_id"),
        "hltv_a": hltv("faction1_hltv_team"), "hltv_b": hltv("faction2_hltv_team"),
        "players_a": unified("team_a_faceit_ids"), "players_b": unified("team_b_faceit_ids"),
        "rounds_a": fm.team_a_score.astype(int).to_numpy(), "rounds_b": fm.team_b_score.astype(int).to_numpy(),
        "won_a": fm.team_a_won.astype(int).to_numpy(), "bo_type": fm.bo_type.astype(int).to_numpy(),
        "div": fm.competition_name.map(division).to_numpy(),
    })


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
    if moves is not None and key > 0:
        five = moves.adjust(key, five, f.last_five_day[key], at)
    return five


def price_matchup(f: SkillFilter, pricer: SeriesPricer, key_a: int, key_b: int, at: pd.Timestamp,
                  moves: RosterMoves | None = None) -> tuple[float, bool]:
    """The channel's series price and whether both teams had a five in its stream."""
    five_a, five_b = expected_five(f, key_a, at, moves), expected_five(f, key_b, at, moves)
    covered = five_a is not None and five_b is not None
    a = f.lineup(five_a or placeholder_five(key_a), key_a if key_a > 0 else None, at)
    b = f.lineup(five_b or placeholder_five(key_b), key_b if key_b > 0 else None, at)
    d, s2, _, _ = f.gap(a, b)
    return pricer.series(d, s2), covered


def logit(q):
    q = np.clip(np.asarray(q, float), 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def features(rows: pd.DataFrame, channels: tuple[str, ...]) -> np.ndarray:
    cols = []
    for c in channels:
        x = logit(rows[f"q_{c}"])
        if c == "faceit":
            x = x * rows["cov_faceit"].to_numpy(float)
        cols.append(x)
    return np.column_stack(cols)


def fit_stack(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """No-intercept logistic regression by Newton steps with a light ridge, starting from the first channel alone."""
    w = np.zeros(X.shape[1])
    w[0] = 1.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - 1e-3 * w
        h = X.T @ (X * (p * (1 - p))[:, None]) + 1e-3 * np.eye(X.shape[1])
        step = np.linalg.solve(h, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def apply_stack(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(X @ w)))


def calibrate(weekly: pd.DataFrame, at: pd.Timestamp, channels: tuple[str, ...]) -> np.ndarray:
    """Stack weights from the weekly prices of the CALIBRATION_WEEKS before `at`, all dated before it."""
    cal = weekly[(weekly.week >= at - CALIBRATION_WEEKS * WEEK) & (weekly.week < at) & (weekly.eff < at)]
    if len(cal) < CALIBRATION_MIN:
        return np.eye(len(channels))[0]
    return fit_stack(features(cal, channels), cal.y.to_numpy(float))


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
    channels = CHANNELS if faceit else CHANNELS[:1]
    takes = {"hltv": (HLTV,), "both": (HLTV, FACEIT), "faceit": (FACEIT,)}
    filters = {c: SkillFilter(cfg, rankings) for c in channels}
    pricer = SeriesPricer(cfg)
    moves = RosterMoves(load(data_dir, "hltv/cs2_roster_moves", last)) if cfg.roster_moves else None

    first_week = monday(cutoffs[0] - CALIBRATION_WEEKS * WEEK)
    if dev_from is not None:
        first_week = min(first_week, monday(dev_from))
    mondays = list(pd.date_range(first_week, last, freq=WEEK))
    by_week = {m: series[(series.eff >= m) & (series.eff < m + WEEK)] for m in mondays}
    targets_by_cutoff = {c: targets[targets.week_cutoff == c] for c in cutoffs}
    weekly_rows: list[dict] = []
    out: list[dict] = []

    def priced(key_a: int, key_b: int, at: pd.Timestamp) -> dict:
        row = {}
        for c in channels:
            row[f"q_{c}"], row[f"cov_{c}"] = price_matchup(filters[c], pricer, key_a, key_b, at, moves)
        return row

    def snapshot(m: pd.Timestamp) -> None:
        for r in by_week[m].itertuples():
            weekly_rows.append({"match_id": r.match_id, "week": m, "eff": r.eff, "y": r.y, **priced(r.key_a, r.key_b, m)})
        if m not in targets_by_cutoff:
            return
        weekly = pd.DataFrame(weekly_rows)
        w = calibrate(weekly, m, channels)
        rows = pd.DataFrame([{"match_id": r.match_id, **priced(int(r.team_a_id), int(r.team_b_id), m)}
                             for r in targets_by_cutoff[m].itertuples()])
        rows["q"] = apply_stack(features(rows, channels), w)
        rows["fit_cutoff"] = m
        out.append(rows)
        log(f"{m:%Y-%m-%d} targets {len(rows)} stack {np.round(w, 3).tolist()} {time.time() - t0:.0f}s")

    eff = stream.eff.to_numpy()
    rows = stream.itertuples(index=False)
    i = 0
    row = next(rows, None)
    for m in mondays:
        while row is not None and eff[i] < m:
            for c in channels:
                if row.source in takes[c]:
                    filters[c].observe_map(pd.Timestamp(row.day), row.key_a, row.key_b, row.hltv_a, row.hltv_b,
                                           row.players_a, row.players_b, row.rounds_a, row.rounds_b, row.source, row.div)
            i += 1
            row = next(rows, None)
        snapshot(m)
    return pd.concat(out, ignore_index=True), pd.DataFrame(weekly_rows)


KNOBS = ("faceit_weight", "faceit_low_div_mult", "faceit_rho", "faceit_sigma0", "faceit_prior_shift", "sigma_map",
         "sigma_series", "tau", "rho")


def config_from_args(a: argparse.Namespace) -> FilterConfig:
    cfg = FilterConfig()
    for k in KNOBS:
        v = getattr(a, k, None)
        if v is not None:
            cfg = replace(cfg, **{k: v})
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
