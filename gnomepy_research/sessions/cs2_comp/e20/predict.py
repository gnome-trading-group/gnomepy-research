"""
Competition predict module for e20.

    python -m gnomepy_research.sessions.cs2_comp.e20.predict --data <bundle> --targets <parquet> --out <parquet> [--no-faceit] [--weeks ...]

One MAP fit per distinct week_cutoff among the targets, on maps dated before it; every target gets a price from
its teams' latest fives, or the rank-group prior when a team has no map at all.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.bayes_series import RankStrength
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e20.data import hltv_rank_group, latest_lineups, load_maps, load_ratings
from gnomepy_research.sessions.cs2_comp.e20.model import Config, Fit, fit

NO_LINEUP = [0] * 5


def fit_week(data_dir: str, cutoff: pd.Timestamp, cfg: Config, use_faceit: bool) -> tuple[Fit, dict, RankStrength]:
    ranks = RankStrength(load(data_dir, "hltv/cs2_team_rankings", cutoff))
    maps = load_maps(data_dir, cutoff, use_faceit, ranks)
    ratings = load_ratings(data_dir, cutoff, maps, use_faceit) if cfg.ratings else None
    model = fit(maps, cutoff, cfg, ratings)
    return model, latest_lineups(maps), ranks


def price_targets(model: Fit, lineups: dict, ranks: RankStrength, targets: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(targets))
    for i, r in enumerate(targets.itertuples()):
        ga, gb = hltv_rank_group(ranks, int(r.team_a_id), model.cutoff), hltv_rank_group(ranks, int(r.team_b_id), model.cutoff)
        la = lineups[int(r.team_a_id)][0] if int(r.team_a_id) in lineups else NO_LINEUP
        lb = lineups[int(r.team_b_id)][0] if int(r.team_b_id) in lineups else NO_LINEUP
        out[i] = model.price(la, ga, lb, gb)
    return out


def run(data_dir: str, targets: pd.DataFrame, cfg: Config, use_faceit: bool, weeks: list[str] | None = None) -> pd.DataFrame:
    t = targets.assign(week_cutoff=pd.to_datetime(targets.week_cutoff))
    cutoffs = sorted(t.week_cutoff.unique())
    if weeks:
        wanted = {pd.Timestamp(w) for w in weeks}
        cutoffs = [c for c in cutoffs if pd.Timestamp(c) in wanted]
    frames = []
    for c in cutoffs:
        t0 = time.time()
        c = pd.Timestamp(c)
        model, lineups, ranks = fit_week(data_dir, c, cfg, use_faceit)
        wk = t[t.week_cutoff == c]
        q = price_targets(model, lineups, ranks, wk)
        frames.append(pd.DataFrame({"match_id": wk.match_id.to_numpy(), "q": q, "fit_cutoff": c}))
        print(f"{c:%Y-%m-%d} {time.time() - t0:.0f}s players {len(model.player_ids)} series {len(wk)} "
              f"iter {model.iterations} gamma {np.round(model.gamma, 3).tolist()} groups {np.round(model.group_mu, 3).tolist()}", flush=True)
    return pd.concat(frames, ignore_index=True)


def _flag(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def add_config_args(ap: argparse.ArgumentParser) -> None:
    for f in dataclasses.fields(Config):
        kind = _flag if isinstance(f.default, bool) else type(f.default)
        ap.add_argument(f"--{f.name}", type=kind, default=f.default)


def config_from(args: argparse.Namespace) -> Config:
    return Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--weeks", nargs="*")
    add_config_args(ap)
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    pred = run(a.data, targets, config_from(a), a.faceit, a.weeks)
    pred.to_parquet(a.out, index=False)
    print(json.dumps({"n": len(pred), "faceit": a.faceit, "out": a.out}))


if __name__ == "__main__":
    sys.exit(main())
