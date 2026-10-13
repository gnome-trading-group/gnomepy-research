"""Team-week posterior snapshots for every weekly cutoff, with rolling-origin hyperparameters.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.models.posterior_snapshots 2020 2026 --publish

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

from gnomepy_research.artifacts import ArtifactStore, DatasetStore
from gnomepy_research.sessions.cfb_win_probability.models.data import season_periods
from gnomepy_research.sessions.cfb_win_probability.models.gaussian_solver import fit_cutoff_exact
from gnomepy_research.sessions.cfb_win_probability.models.inference import (
    EFFICIENCY,
    HYPER_DIR,
    PRESEASON_FEATURES,
    Inputs,
    fit_cutoff,
    fit_hyperparameters,
    load_hyperparameters,
    save_hyperparameters,
)

MODEL_VERSION = "dyn-eff-1"
FIRST_TRAIN_SEASON = 2019
N_DRAWS_KEPT = 400
SESSION = "cfb_win_probability"
DRAWS_DIR = Path(__file__).resolve().parent.parent / "results" / "posterior_draws"


def hyper_name(likelihood: str, season: int) -> str:
    return f"{likelihood}_{FIRST_TRAIN_SEASON}_{season - 1}"


def ensure_hyperparameters(inputs: Inputs, likelihood: str, season: int) -> str:
    name = hyper_name(likelihood, season)
    if not (HYPER_DIR / f"{name}.json").exists():
        h, diag = fit_hyperparameters(inputs, list(range(FIRST_TRAIN_SEASON, season)), likelihood)
        save_hyperparameters(h, diag, name)
        print(f"fitted {name}: {diag['hyper']}", flush=True)
    return name


def snapshot_season(inputs: Inputs, season: int, likelihood: str, seed: int = 0) -> tuple[pd.DataFrame, Path]:
    name = ensure_hyperparameters(inputs, likelihood, season)
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
        counts = seen.groupby("team_id").agg(games_observed=("game_id", "nunique"), plays_observed=("plays", "sum"))
        s = post.summary().merge(counts, left_on="team_id", right_index=True, how="left")
        s[["games_observed", "plays_observed"]] = s[["games_observed", "plays_observed"]].fillna(0)
        summaries.append(s.assign(season_type=p.season_type, week=p.week, period=p.period,
                                  alpha_mean=float(post.alpha.mean()), max_rhat=post.diagnostics["max_rhat"],
                                  divergences=post.diagnostics["divergences"]))
        print(f"{season} {key}: rhat {post.diagnostics['max_rhat']:.3f} div {post.diagnostics['divergences']}", flush=True)
    DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    path = DRAWS_DIR / f"cfb_team_week_draws_{season}.npz"
    np.savez_compressed(path, **draws)
    out = pd.concat(summaries, ignore_index=True).assign(
        model_version=MODEL_VERSION, hyperparameters=name, likelihood=likelihood,
        training_data_version=f"{PRESEASON_FEATURES},{EFFICIENCY}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return out, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--likelihood", default="studentt")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    if args.first <= FIRST_TRAIN_SEASON:
        raise SystemExit(f"first snapshot season must be after {FIRST_TRAIN_SEASON}")
    inputs = Inputs.load()
    frames, paths = [], []
    for season in range(args.first, args.last + 1):
        df, path = snapshot_season(inputs, season, args.likelihood)
        frames.append(df)
        paths.append((season, path))
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(DRAWS_DIR / f"cfb_team_week_posterior_{args.first}_{args.last}.parquet", index=False)
    print(f"{len(table):,} team-week rows")
    if args.publish:
        print(DatasetStore().publish(
            table, "cfb_team_week_posterior",
            description=f"{MODEL_VERSION} {args.likelihood} team-week posteriors (rolling-origin hyperparameters "
                        f"{FIRST_TRAIN_SEASON}..S-1), seasons {args.first}-{args.last}",
            producing_session=SESSION))
        store = ArtifactStore()
        for season, path in paths:
            print(store.publish(str(path), artifact_type="posterior_draws", name=f"cfb_team_week_draws_{season}",
                                session_name=SESSION,
                                description=f"{MODEL_VERSION} joint draws (alpha, O, D) per weekly cutoff, {season}",
                                params={"likelihood": args.likelihood, "hyperparameters": hyper_name(args.likelihood, season),
                                        "draws": N_DRAWS_KEPT}))


if __name__ == "__main__":
    main()
