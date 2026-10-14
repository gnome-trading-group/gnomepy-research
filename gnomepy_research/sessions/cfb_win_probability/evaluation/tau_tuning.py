"""Tune the weekly rating drift (tau) on one-step-ahead predictive accuracy.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.tau_tuning 2023 2024

The MCMC fit picks tau to explain past PPA, and with game-level PPA this noisy it picks a drift so
small that the latest game gets ~6% weight. Here tau is instead chosen for what we care about:
predicting next week. For each tuning season (rolling-origin hyperparameters 2019..S-1) and
each weekly cutoff, refit the states with tau scaled by each multiplier (exact Gaussian solver)
and score that week's team-game PPA by log predictive density and squared error.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.efficiency.data import season_periods
from gnomepy_research.models.cfb.efficiency.gaussian_solver import fit_cutoff_exact
from gnomepy_research.models.cfb.efficiency.inference import (
    Inputs,
    load_hyperparameters,
    predictive_logpdf,
    week_observations,
)
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import hyper_name

MULTIPLIERS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
RESULTS = Path(__file__).resolve().parent.parent / "results" / "tau_tuning"


def run(seasons: list[int], multipliers: tuple[float, ...] = MULTIPLIERS) -> pd.DataFrame:
    inputs = Inputs.load()
    rows = []
    for season in seasons:
        base = load_hyperparameters(hyper_name("normal", season))
        for p in season_periods(inputs.games, [season]).itertuples():
            for m in multipliers:
                h = replace(base, tau_off=base.tau_off * m, tau_def=base.tau_def * m)
                post, data = fit_cutoff_exact(inputs, season, p.cutoff, h, n_draws=1000)
                obs = week_observations(inputs, post, data)
                if obs.empty:
                    continue
                mu = (post.alpha[:, obs["alpha_idx"].to_numpy()] + post.off[:, obs["off_idx"].to_numpy()]
                      - post.de[:, obs["def_idx"].to_numpy()] + h.hfa * obs["venue"].to_numpy()[None, :]).mean(0)
                rows.append(pd.DataFrame({
                    "season": season, "season_type": p.season_type, "week": p.week, "multiplier": m,
                    "tau_off": h.tau_off, "game_id": obs["game_id"].to_numpy(), "team_id": obs["team_id"].to_numpy(),
                    "logpdf": predictive_logpdf(post, obs), "se": (obs["off_ppa"].to_numpy() - mu) ** 2,
                }))
            print(f"{season} {p.season_type} wk{p.week}", flush=True)
    return pd.concat(rows, ignore_index=True)


def report(r: pd.DataFrame) -> str:
    r = r.assign(phase=np.where(r["season_type"] == "postseason", "post",
                                pd.cut(r["week"], [0, 5, 9, 20], labels=["wk1-5", "wk6-9", "wk10+"]).astype(str)))
    overall = r.groupby("multiplier").agg(n=("logpdf", "size"), mean_logpdf=("logpdf", "mean"),
                                          rmse=("se", lambda s: np.sqrt(s.mean())), tau_off=("tau_off", "mean"))
    by_phase = r.pivot_table(index="phase", columns="multiplier", values="logpdf", aggfunc="mean")
    by_season = r.pivot_table(index="season", columns="multiplier", values="logpdf", aggfunc="mean")
    # Paired by observation against the MCMC-chosen tau (multiplier 1), clustered by game.
    w = r.pivot_table(index=["game_id", "team_id"], columns="multiplier", values="logpdf")
    paired = {}
    for m in w.columns:
        d = (w[m] - w[1.0]).groupby(level=0).sum()
        paired[m] = {"per_game_gain": d.mean(), "se": d.std() / np.sqrt(len(d))}
    return "\n\n".join([
        "overall (higher logpdf = better):\n" + overall.round(5).to_string(),
        "mean log predictive density by phase:\n" + by_phase.round(5).to_string(),
        "by season:\n" + by_season.round(5).to_string(),
        "paired gain vs MCMC tau (per game, game-clustered se):\n" + pd.DataFrame(paired).T.round(5).to_string(),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    args = ap.parse_args()
    r = run(list(range(args.first, args.last + 1)))
    RESULTS.mkdir(parents=True, exist_ok=True)
    r.to_parquet(RESULTS / f"tau_{args.first}_{args.last}.parquet", index=False)
    text = report(r)
    (RESULTS / f"tau_{args.first}_{args.last}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
