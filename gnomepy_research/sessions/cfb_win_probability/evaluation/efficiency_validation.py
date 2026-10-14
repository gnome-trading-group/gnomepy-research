"""One-step-ahead validation of the efficiency model on the tuning seasons.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.efficiency_validation 2023 2024

At every weekly cutoff, fit the states on data before the cutoff and score that week's team-game
PPA observations:
- log predictive density (mixture over posterior draws)
- squared error of the predictive mean
- PIT value (posterior-predictive CDF at the observation), for calibration

Variants: Student-t, Gaussian, Student-t with no in-season drift (tau = 0), and a ridge baseline
(season-to-date ridge ratings at the same cutoff, league mean before any games).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gnomepy_research.models.cfb.features.ratings import FCS_TEAM, ridge_ratings
from gnomepy_research.models.cfb.efficiency.data import season_periods
from gnomepy_research.models.cfb.efficiency.inference import (
    Inputs,
    fit_cutoff,
    load_hyperparameters,
    predictive_logpdf,
    week_observations,
)

RESULTS = Path(__file__).resolve().parent.parent / "results" / "efficiency_validation"


def pit(post, obs: pd.DataFrame) -> np.ndarray:
    h = post.hypers
    mu = (post.alpha[:, obs["alpha_idx"].to_numpy()] + post.off[:, obs["off_idx"].to_numpy()] - post.de[:, obs["def_idx"].to_numpy()]
          + h.hfa * obs["venue"].to_numpy()[None, :])
    sigma = np.sqrt(h.sigma_game ** 2 + h.sigma_play ** 2 / obs["plays"].to_numpy())[None, :]
    y = obs["off_ppa"].to_numpy()[None, :]
    cdf = stats.t.cdf(y, df=h.nu, loc=mu, scale=sigma) if h.likelihood == "studentt" else stats.norm.cdf(y, mu, sigma)
    return cdf.mean(0), mu.mean(0)


def ridge_prediction(inputs: Inputs, season: int, cutoff: pd.Timestamp, obs: pd.DataFrame) -> np.ndarray:
    eff = inputs.visible(cutoff)
    eff = eff[eff["season"] == season]
    if eff.empty:
        prior = inputs.visible(cutoff)
        return np.full(len(obs), prior.loc[prior["usable"], "off_ppa"].mean())
    r, glob = ridge_ratings(eff)
    r = r.set_index("team_id")
    off_t = np.where(obs["team_classification"] == "fbs", obs["team_id"], FCS_TEAM)
    def_t = np.where(obs["opponent_classification"] == "fbs", obs["opponent_id"], FCS_TEAM)
    o = r["off_rating"].reindex(off_t).fillna(0.0).to_numpy()
    d = r["def_rating"].reindex(def_t).fillna(0.0).to_numpy()
    return glob["mu"] + o - d + glob["hfa"] * obs["venue"].to_numpy()


def run(seasons: list[int], hyper_names: dict[str, str]) -> pd.DataFrame:
    inputs = Inputs.load()
    variants = {
        "studentt": load_hyperparameters(hyper_names["studentt"]),
        "normal": load_hyperparameters(hyper_names["normal"]),
    }
    variants["studentt_static"] = replace(variants["studentt"], tau_off=1e-6, tau_def=1e-6)
    rows = []
    for season in seasons:
        for p in season_periods(inputs.games, [season]).itertuples():
            ridge_done = False
            for name, h in variants.items():
                post, data = fit_cutoff(inputs, season, p.cutoff, h)
                obs = week_observations(inputs, post, data)
                if obs.empty:
                    continue
                lp = predictive_logpdf(post, obs)
                u, mean = pit(post, obs)
                rows.append(pd.DataFrame({
                    "variant": name, "season": season, "season_type": p.season_type, "week": p.week,
                    "period": p.period, "game_id": obs["game_id"].to_numpy(), "team_id": obs["team_id"].to_numpy(),
                    "y": obs["off_ppa"].to_numpy(), "plays": obs["plays"].to_numpy(), "pred": mean,
                    "logpdf": lp, "pit": u, "max_rhat": post.diagnostics["max_rhat"],
                    "divergences": post.diagnostics["divergences"],
                }))
                if not ridge_done:
                    rows.append(pd.DataFrame({
                        "variant": "ridge", "season": season, "season_type": p.season_type, "week": p.week,
                        "period": p.period, "game_id": obs["game_id"].to_numpy(), "team_id": obs["team_id"].to_numpy(),
                        "y": obs["off_ppa"].to_numpy(), "plays": obs["plays"].to_numpy(),
                        "pred": ridge_prediction(inputs, season, p.cutoff, obs),
                        "logpdf": np.nan, "pit": np.nan, "max_rhat": np.nan, "divergences": np.nan,
                    }))
                    ridge_done = True
            print(f"{season} {p.season_type} wk{p.week}: done", flush=True)
    return pd.concat(rows, ignore_index=True)


def report(r: pd.DataFrame) -> str:
    r = r.assign(se=(r["y"] - r["pred"]) ** 2,
                 phase=np.where(r["season_type"] == "postseason", "post",
                                pd.cut(r["week"], [0, 2, 5, 9, 20], labels=["wk1-2", "wk3-5", "wk6-9", "wk10+"]).astype(str)))
    lines = []
    overall = r.groupby("variant").agg(n=("y", "size"), rmse=("se", lambda s: np.sqrt(s.mean())),
                                       mean_logpdf=("logpdf", "mean"), max_rhat=("max_rhat", "max"),
                                       divergences=("divergences", "sum"))
    lines.append("overall:\n" + overall.round(4).to_string())
    by = r.pivot_table(index="phase", columns="variant", values="se", aggfunc=lambda s: np.sqrt(s.mean()))
    lines.append("RMSE by phase:\n" + by.round(4).to_string())
    lp = r.pivot_table(index="phase", columns="variant", values="logpdf", aggfunc="mean")
    lines.append("mean log predictive density by phase:\n" + lp.round(4).to_string())
    cal = r.dropna(subset=["pit"]).groupby("variant")["pit"].agg(
        cov50=lambda u: ((u > 0.25) & (u < 0.75)).mean(), cov90=lambda u: ((u > 0.05) & (u < 0.95)).mean(),
        pit_mean="mean", pit_sd="std")
    lines.append("calibration (ideal: cov50 0.50, cov90 0.90, pit mean 0.5, pit sd 0.289):\n" + cal.round(3).to_string())
    # Paired difference: same observations, t vs normal, clustered by game.
    w = r.pivot_table(index=["game_id", "team_id"], columns="variant", values="logpdf").dropna()
    if {"studentt", "normal"} <= set(w.columns):
        d = (w["studentt"] - w["normal"]).groupby(level=0).sum()
        lines.append(f"studentt - normal log score: {d.sum():.2f} total over {len(d)} games "
                     f"(per-game mean {d.mean():.4f}, se {d.std() / np.sqrt(len(d)):.4f})")
    if {"studentt", "studentt_static"} <= set(w.columns):
        d = (w["studentt"] - w["studentt_static"]).groupby(level=0).sum()
        lines.append(f"dynamic - static log score: {d.sum():.2f} total (per-game mean {d.mean():.4f}, "
                     f"se {d.std() / np.sqrt(len(d)):.4f})")
    return "\n\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--hyper-train", default="2019_2022")
    args = ap.parse_args()
    seasons = list(range(args.first, args.last + 1))
    r = run(seasons, {"studentt": f"studentt_{args.hyper_train}", "normal": f"normal_{args.hyper_train}"})
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"one_step_{args.first}_{args.last}.parquet"
    r.to_parquet(out, index=False)
    text = report(r)
    (RESULTS / f"one_step_{args.first}_{args.last}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
