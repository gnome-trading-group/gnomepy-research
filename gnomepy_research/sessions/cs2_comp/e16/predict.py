"""
Pre-match bo3 series prices for the competition targets, one fit per distinct week_cutoff on data before it.

    python -m gnomepy_research.sessions.cs2_comp.e16.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--no-faceit] [--weeks 2026-04-06,2026-04-13] [--set name=value ...]

Each target is priced on its announced fives (team_x_lineup_ids). A target without one falls back to each team's
latest five, with HLTV roster moves dated after that sighting applied.

Each week fits the HLTV-only model. Without FACEIT that fit prices every target. With FACEIT a second, joint fit
over HLTV and FACEIT maps (FACEIT impact rows, FACEIT skill level and ESEA division priors) runs as well, and:
  - a player of the five whom the HLTV-only fit never saw takes his skill from the joint fit (shifted onto the
    HLTV-only scale) instead of the newcomer prior;
  - the latest-five fallback looks across HLTV and FACEIT maps (a FACEIT qualifier is often the freshest sighting
    of an online team);
  - cfg.faceit_weight > 0 also mixes in the joint fit's own gaps (0 by default: pooling FACEIT results moved
    established HLTV players too far and scored worse on the tune period).
--set overrides a Config field, for tuning on the dev period only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e16.data import LINEUP, Bundle, apply_moves, read_bundle, team_lineups
from gnomepy_research.sessions.cs2_comp.e16.model import Config, Fit, fit, rank_feature


def window(df: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
    return df[(df.cut < cutoff) & (df.time >= cutoff - pd.Timedelta(days=days))]


def _five(fives: dict, team: str, moves: pd.DataFrame, cfg: Config) -> list[str] | None:
    five, seen = fives.get(team, (None, None))
    if not cfg.roster_moves:
        return five
    return apply_moves(five, seen, team, moves)


def _announced(ids) -> list[str] | None:
    if ids is None or len(ids) != LINEUP or any(pd.isna(x) for x in ids):
        return None
    return [f"h{int(x)}" for x in ids]


def _gaps(maps: pd.DataFrame, ratings: pd.DataFrame, levels: pd.DataFrame | None, b: Bundle, targets: pd.DataFrame,
          cutoff: pd.Timestamp, cfg: Config, lineup_maps: pd.DataFrame,
          prefit: Fit | None = None) -> tuple[np.ndarray, np.ndarray, Fit]:
    """Fit on the given maps and return each target's pre-veto gap and its variance, each team priced with its
    latest five in lineup_maps."""
    rank_of = rank_feature(b.rankings, cutoff)
    f = fit(maps, ratings, levels, rank_of, cutoff, cfg) if prefit is None else prefit
    teams = {f"h{int(t)}" for t in pd.concat([targets.team_a_id, targets.team_b_id])}
    fives = team_lineups(lineup_maps, teams, bool(cfg.revert_standins))
    moves = b.moves[b.moves.event_date < cutoff]
    announced = bool(cfg.announced_lineups) and "team_a_lineup_ids" in targets.columns
    gaps = []
    for r in targets.itertuples():
        ka, kb = f"h{int(r.team_a_id)}", f"h{int(r.team_b_id)}"
        five_a, five_b = (_five(fives, k, moves, cfg) for k in (ka, kb))
        if announced:
            five_a = _announced(r.team_a_lineup_ids) or five_a
            five_b = _announced(r.team_b_lineup_ids) or five_b
        gaps.append(f.gap(ka, five_a, rank_of.get(ka, 0.0), kb, five_b, rank_of.get(kb, 0.0)))
    g = np.array(gaps, dtype=float).reshape(-1, 2)
    return g[:, 0], g[:, 1], f


def predict_week(b: Bundle, targets: pd.DataFrame, cutoff: pd.Timestamp, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """HLTV-only gaps; with FACEIT, unseen players from the joint fit, and the joint fit's gaps mixed in by
    cfg.faceit_weight."""
    maps = window(b.hltv_maps, cutoff, cfg.window_days)
    ratings = window(b.hltv_ratings, cutoff, cfg.window_days).assign(source="hltv")
    gap, var, f = _gaps(maps, ratings, None, b, targets, cutoff, cfg, maps)
    info = {"hltv": f.info}
    if b.faceit_maps is not None:
        fmaps = window(b.faceit_maps, cutoff, cfg.faceit_window_days)
        fmaps = fmaps[fmaps.n_linked >= cfg.faceit_min_linked]
        frat = window(b.faceit_ratings, cutoff, cfg.faceit_window_days)
        frat = frat[frat.mkey.isin(set(fmaps.mkey))]
        jmaps = pd.concat([maps, fmaps.drop(columns=["n_linked", "mkey"])], ignore_index=True)
        jratings = pd.concat([ratings, frat.drop(columns=["mkey"]).assign(source="faceit")], ignore_index=True)
        levels = b.faceit_levels[b.faceit_levels.cut < cutoff]
        gap_j, var_j, fj = _gaps(jmaps, jratings, levels, b, targets, cutoff, cfg, jmaps if cfg.faceit_lineups else maps)
        if cfg.faceit_fallback:
            gap, var, _ = _gaps(maps, ratings, None, b, targets, cutoff, cfg, jmaps if cfg.faceit_lineups else maps,
                                f.with_fallback(fj))
        w = cfg.faceit_weight
        gap, var = (1 - w) * gap + w * gap_j, (1 - w) * var + w * var_j
        info["joint"] = fj.info
    q = [f.price(m, v) for m, v in zip(gap, var)]
    out = pd.DataFrame({"match_id": targets.match_id.to_numpy(), "q": q, "gap": gap, "gap_var": var, "fit_cutoff": cutoff})
    return out, info


def run(data: Path, targets: pd.DataFrame, faceit: bool, cfg: Config, verbose: bool = True) -> pd.DataFrame:
    cutoffs = sorted(pd.to_datetime(targets.week_cutoff).unique())
    since = pd.Timestamp(min(cutoffs)) - pd.Timedelta(days=max(cfg.window_days, cfg.faceit_window_days))
    b = read_bundle(data, max(cutoffs), faceit, since)
    parts = []
    for c in cutoffs:
        t0 = time.time()
        c = pd.Timestamp(c)
        week = targets[pd.to_datetime(targets.week_cutoff) == c]
        p, info = predict_week(b, week, c, cfg)
        parts.append(p)
        if verbose:
            print(f"{c:%Y-%m-%d} {len(p)} series {time.time() - t0:.1f}s {json.dumps(info)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def parse_overrides(items: list[str]) -> dict:
    fields = {f.name: f.type for f in dataclasses.fields(Config)}
    over = {}
    for kv in items:
        k, v = kv.split("=", 1)
        over[k] = int(v) if fields[k] in (int, "int") else float(v)
    return over


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="a relative path lands under $OUTPUT_DIR when set (Batch)")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--set", action="append", default=[], help="Config override name=value (dev tuning)")
    args = ap.parse_args()
    cfg = dataclasses.replace(Config(), **parse_overrides(args.set))
    targets = pd.read_parquet(args.targets)
    if args.weeks:
        keep = pd.to_datetime(args.weeks.split(","))
        targets = targets[pd.to_datetime(targets.week_cutoff).isin(keep)]
    pred = run(args.data, targets, args.faceit, cfg)
    pred["q"] = np.clip(pred.q, 1e-4, 1 - 1e-4)
    if not args.out.is_absolute() and "OUTPUT_DIR" in os.environ:
        args.out = Path(os.environ["OUTPUT_DIR"]) / args.out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(args.out, index=False)
    print(f"wrote {len(pred)} predictions to {args.out}")


if __name__ == "__main__":
    main()
