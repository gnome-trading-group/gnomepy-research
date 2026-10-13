"""Team-week posterior snapshots for every weekly cutoff, with rolling-origin hyperparameters.

    poetry run python -m gnomepy_research.models.cfb.efficiency.posterior_snapshots 2020 2026

gnome-pipelines publishes the outputs (gnome_pipelines.cfb.posterior).

Season S uses hyperparameters fitted on 2019..S-1 only, so nothing about S (or later) shapes its
priors. 2019 is the first season with prior-season ratings, so it only ever trains.

Outputs:
- dataset cfb_team_week_posterior: one row per (cutoff, team) with posterior means/sds, O-D
  correlation, data counts and versions (the spec's team-week contract)
- artifact posterior_draws/cfb_team_week_draws_<season>: npz with thinned joint draws of
  (alpha, O, D) per cutoff, for the outcome model and the future live engine
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.efficiency.data import UNITS_ALL, UNITS_SPLIT, season_periods
from gnomepy_research.models.cfb.efficiency.gaussian_solver import fit_cutoff_exact
from gnomepy_research.models.cfb.features.rush_share import league_rush_share, rush_share
from gnomepy_research.models.cfb.efficiency.inference import (
    EFFICIENCY,
    HYPER_DIR,
    PRESEASON_FEATURES,
    Inputs,
    fit_cutoff,
    fit_hyperparameters,
    load_hyperparameters,
    save_hyperparameters,
)

SPLIT_SUFFIX = "_split"
FIRST_TRAIN_SEASON = 2019
N_DRAWS_KEPT = 400
DRAWS_DIR = Path(__file__).resolve().parents[3] / "sessions" / "cfb_win_probability" / "results" / "posterior_draws"


def _suffix(units: tuple[str, ...]) -> str:
    return "" if units == UNITS_ALL else SPLIT_SUFFIX


def model_version(units: tuple[str, ...]) -> str:
    return "dyn-eff-1" if units == UNITS_ALL else "dyn-eff-split-1"


MODEL_VERSION = model_version(UNITS_ALL)


def hyper_name(likelihood: str, season: int, units: tuple[str, ...] = UNITS_ALL) -> str:
    return f"{likelihood}{_suffix(units)}_{FIRST_TRAIN_SEASON}_{season - 1}"


def draws_name(season: int, units: tuple[str, ...] = UNITS_ALL) -> str:
    return f"cfb_team_week_draws{_suffix(units)}_{season}"


def ensure_hyperparameters(inputs: Inputs, likelihood: str, season: int, units: tuple[str, ...] = UNITS_ALL) -> str:
    name = hyper_name(likelihood, season, units)
    if not (HYPER_DIR / f"{name}.json").exists():
        h, diag = fit_hyperparameters(inputs, list(range(FIRST_TRAIN_SEASON, season)), likelihood, units=units)
        save_hyperparameters(h, diag, name)
        print(f"fitted {name}: {diag['hyper']}", flush=True)
    return name


def snapshot_season(inputs: Inputs, season: int, likelihood: str, seed: int = 0,
                    out_dir: Path = DRAWS_DIR, units: tuple[str, ...] = UNITS_ALL) -> tuple[pd.DataFrame, Path]:
    name = ensure_hyperparameters(inputs, likelihood, season, units)
    h = load_hyperparameters(name)
    rng = np.random.default_rng(seed)
    summaries, draws = [], {}
    e = inputs.efficiency
    for p in season_periods(inputs.games, [season]).itertuples():
        # The Gaussian model has a closed-form posterior (validated against NUTS); Student-t needs MCMC.
        fit = fit_cutoff_exact if likelihood == "normal" else fit_cutoff
        post, _ = fit(inputs, season, p.cutoff, h, seed=seed)
        keep = rng.choice(post.off.shape[0], size=min(N_DRAWS_KEPT, post.off.shape[0]), replace=False)
        key = f"{p.season_type}_{p.week}"
        draws[f"{key}__off"] = post.off[keep].astype(np.float32)
        draws[f"{key}__def"] = post.de[keep].astype(np.float32)
        draws[f"{key}__alpha"] = post.alpha[keep].astype(np.float32)
        draws[f"{key}__team_ids"] = post.team_ids
        seen = e[(e["available_at"] < p.cutoff) & (e["season"] == season) & e["usable"]]
        shares = rush_share(seen, season)
        lg = league_rush_share(seen, season)
        draws[f"{key}__units"] = post.units.astype("U8")
        draws[f"{key}__rush_share"] = shares.reindex(post.team_ids).fillna(lg).to_numpy(np.float32)
        counts = seen.groupby("team_id").agg(games_observed=("game_id", "nunique"), plays_observed=("plays", "sum"))
        s = post.summary().merge(counts, left_on="team_id", right_index=True, how="left")
        s[["games_observed", "plays_observed"]] = s[["games_observed", "plays_observed"]].fillna(0)
        summaries.append(s.assign(season_type=p.season_type, week=p.week, period=p.period,
                                  max_rhat=post.diagnostics["max_rhat"],
                                  divergences=post.diagnostics["divergences"]))
        print(f"{season} {key}: rhat {post.diagnostics['max_rhat']:.3f} div {post.diagnostics['divergences']}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{draws_name(season, units)}.npz"
    np.savez_compressed(path, **draws)
    out = pd.concat(summaries, ignore_index=True).assign(
        model_version=model_version(units), hyperparameters=name, likelihood=likelihood,
        training_data_version=f"{PRESEASON_FEATURES},{EFFICIENCY}@{inputs.as_of.isoformat()}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return out, path


def snapshot_seasons(inputs: Inputs, first: int, last: int, likelihood: str,
                     out_dir: Path = DRAWS_DIR, units: tuple[str, ...] = UNITS_ALL) -> tuple[pd.DataFrame, list[tuple[int, Path]]]:
    """The team-week table for seasons first..last, and each season's draws file under out_dir."""
    if first <= FIRST_TRAIN_SEASON:
        raise ValueError(f"first snapshot season must be after {FIRST_TRAIN_SEASON}")
    frames, paths = [], []
    for season in range(first, last + 1):
        df, path = snapshot_season(inputs, season, likelihood, out_dir=out_dir, units=units)
        frames.append(df)
        paths.append((season, path))
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(out_dir / f"cfb_team_week_posterior_{first}_{last}.parquet", index=False)
    return table, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--likelihood", default="studentt")
    ap.add_argument("--units", choices=["all", "split"], default="all")
    args = ap.parse_args()
    units = UNITS_ALL if args.units == "all" else UNITS_SPLIT
    table, _ = snapshot_seasons(Inputs.load(), args.first, args.last, args.likelihood, units=units)
    print(f"{len(table):,} team-week rows")


if __name__ == "__main__":
    main()
