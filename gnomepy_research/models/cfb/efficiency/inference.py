"""Hyperparameter fits and per-cutoff state fits, plus one-step-ahead predictive scoring.

    poetry run python -m gnomepy_research.models.cfb.efficiency.inference hyper 2019 2022 --likelihood studentt
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gnomepy.data import DatasetStore
from gnomepy_research.models.cfb.efficiency.data import ModelData, build_model_data
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import (
    Hyperparameters,
    build_model,
    diagnostics,
    hyperparameters_from,
    sample,
)

# Fitted in research, read by the posterior pipeline, so they ship (and are versioned) with the code.
HYPER_DIR = Path(__file__).resolve().parent.parent / "hyperparameters"
HYPER_VARS = ["alpha", "sigma_alpha", "hfa", "beta_off", "beta_def", "gamma", "s_off", "s_def", "tau_off",
              "tau_def", "sigma_game", "sigma_play"]
PRESEASON_FEATURES = "cfb_team_preseason_features"
EFFICIENCY = "cfb_team_game_efficiency"
GAMES = "cfb_games"


@dataclass
class Inputs:
    preseason: pd.DataFrame
    efficiency: pd.DataFrame
    games: pd.DataFrame
    as_of: datetime

    @classmethod
    def load(cls, ds: DatasetStore | None = None, as_of: datetime | None = None) -> Inputs:
        """The model inputs as known at as_of (default now); ds records the loads for lineage."""
        ds = ds or DatasetStore()
        as_of = as_of or datetime.now(timezone.utc)
        return cls(ds.load(PRESEASON_FEATURES, as_of=as_of), ds.load(EFFICIENCY, as_of=as_of),
                   ds.load(GAMES, as_of=as_of), as_of)

    def visible(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        return self.efficiency[self.efficiency["available_at"] < cutoff]


@dataclass
class CutoffPosterior:
    season: int
    cutoff: pd.Timestamp
    period: int                    # state index a prediction at this cutoff uses
    team_ids: np.ndarray           # (T,) FBS team ids, then FCS_TEAM
    off: np.ndarray                # (S, T) draws of O at `period`
    de: np.ndarray                 # (S, T) draws of D at `period`
    alpha: np.ndarray              # (S,)
    hypers: Hyperparameters
    diagnostics: dict

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "season": self.season, "cutoff": self.cutoff, "team_id": self.team_ids,
            "offense_mean": self.off.mean(0), "offense_sd": self.off.std(0),
            "defense_mean": self.de.mean(0), "defense_sd": self.de.std(0),
            "od_corr": [np.corrcoef(self.off[:, i], self.de[:, i])[0, 1] for i in range(len(self.team_ids))],
        })


def hyper_cutoff(inputs: Inputs, last_train_season: int) -> pd.Timestamp:
    """Hyperparameters may only use games public before the next season's first cutoff."""
    g = inputs.games
    nxt = g[g["season"] == last_train_season + 1]
    return nxt["kickoff"].min() if len(nxt) else g["kickoff"].max() + pd.Timedelta(days=2)


def fit_hyperparameters(inputs: Inputs, seasons: list[int], likelihood: str, draws: int = 1000,
                        tune: int = 1500, seed: int = 0) -> tuple[Hyperparameters, dict]:
    cutoff = hyper_cutoff(inputs, max(seasons))
    data = build_model_data(inputs.preseason, inputs.visible(cutoff), inputs.games, seasons)
    idata = sample(build_model(data, likelihood=likelihood, store_states=False), draws=draws, tune=tune, seed=seed)
    hv = HYPER_VARS + (["nu"] if likelihood == "studentt" else [])
    diag = {"hyper": diagnostics(idata, hv), "states": diagnostics(idata, ["z0_off", "z0_def"]),
            "seasons": seasons, "cutoff": str(cutoff), "n_obs": int(len(data.obs_y))}
    return hyperparameters_from(idata, data, likelihood), diag


def save_hyperparameters(h: Hyperparameters, diag: dict, name: str) -> Path:
    HYPER_DIR.mkdir(parents=True, exist_ok=True)
    path = HYPER_DIR / f"{name}.json"
    path.write_text(json.dumps({"hyperparameters": h.to_dict(), "diagnostics": diag}, indent=2, default=str))
    return path


def load_hyperparameters(name: str) -> Hyperparameters:
    return Hyperparameters.from_dict(json.loads((HYPER_DIR / f"{name}.json").read_text())["hyperparameters"])


def fit_cutoff(inputs: Inputs, season: int, cutoff: pd.Timestamp, hypers: Hyperparameters,
               draws: int = 500, tune: int = 500, seed: int = 0) -> tuple[CutoffPosterior, ModelData]:
    data = build_model_data(inputs.preseason, inputs.visible(cutoff), inputs.games, [season])
    idata = sample(build_model(data, hypers=hypers, likelihood=hypers.likelihood), draws=draws, tune=tune, seed=seed)
    period = int(data.periods.loc[data.periods["cutoff"] == cutoff, "period"].iloc[0])
    post = idata.posterior.to_dataset().stack(sample=("chain", "draw"))
    off = post["O"].values[:, period, :].T
    de = post["D"].values[:, period, :].T
    alpha = post["alpha_season"].values[0]
    return CutoffPosterior(
        season=season, cutoff=cutoff, period=period, team_ids=data.ts_keys["team_id"].to_numpy(),
        off=off, de=de, alpha=alpha, hypers=hypers,
        diagnostics=diagnostics(idata, ["z0_off", "z0_def", "eps_off", "eps_def", "z_alpha"]),
    ), data


def predictive_logpdf(post: CutoffPosterior, obs: pd.DataFrame) -> np.ndarray:
    """Log posterior-predictive density of each observed team-game PPA (mixture over draws).
    `obs` rows need off_idx/def_idx (columns of post.off) plus venue, plays, off_ppa."""
    h = post.hypers
    mu = (post.alpha[:, None] + post.off[:, obs["off_idx"].to_numpy()] - post.de[:, obs["def_idx"].to_numpy()]
          + h.hfa * obs["venue"].to_numpy()[None, :])
    sigma = np.sqrt(h.sigma_game ** 2 + h.sigma_play ** 2 / obs["plays"].to_numpy())[None, :]
    y = obs["off_ppa"].to_numpy()[None, :]
    if h.likelihood == "studentt":
        lp = stats.t.logpdf(y, df=h.nu, loc=mu, scale=sigma)
    else:
        lp = stats.norm.logpdf(y, loc=mu, scale=sigma)
    return np.logaddexp.reduce(lp, axis=0) - np.log(lp.shape[0])


def week_observations(inputs: Inputs, post: CutoffPosterior, data: ModelData) -> pd.DataFrame:
    """The team-game rows played in the cutoff's own week, mapped onto the posterior's columns."""
    p = data.periods.loc[data.periods["period"] == post.period].iloc[0]
    e = inputs.efficiency
    e = e[(e["season"] == p["season"]) & (e["season_type"] == p["season_type"]) & (e["week"] == p["week"]) & e["usable"]]
    col = {t: i for i, t in enumerate(post.team_ids)}
    fcs = col[post.team_ids[-1]]
    e = e.assign(
        off_idx=[col.get(t, fcs) if c == "fbs" else fcs for t, c in zip(e["team_id"], e["team_classification"])],
        def_idx=[col.get(t, fcs) if c == "fbs" else fcs for t, c in zip(e["opponent_id"], e["opponent_classification"])],
    )
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hyper"])
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--likelihood", default="studentt")
    args = ap.parse_args()
    inputs = Inputs.load()
    seasons = list(range(args.first, args.last + 1))
    h, diag = fit_hyperparameters(inputs, seasons, args.likelihood)
    path = save_hyperparameters(h, diag, f"{args.likelihood}_{args.first}_{args.last}")
    print(json.dumps(diag, indent=2, default=str))
    print("saved", path)


if __name__ == "__main__":
    main()
