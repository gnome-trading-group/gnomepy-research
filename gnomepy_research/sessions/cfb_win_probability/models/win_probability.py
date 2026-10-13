"""Pregame win-probability model on top of the efficiency posteriors.

    P(home wins) = logistic(z)
    z = g_off * (O_home - O_away) + g_def * (D_home - D_away) + g_venue * venue
        + g_bowl * postseason * [(O_home - O_away) + (D_home - D_away)]

No intercept: swapping the teams in a neutral-site game gives exactly 1 - p. Rating uncertainty is
integrated out inside the likelihood (each game's probability averages logistic(z) over posterior
rating draws), so the coefficients aren't attenuated by noisy ratings and uncertainty isn't
double counted at prediction time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from gnomepy_research.models.cfb.features.ratings import FCS_TEAM
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import DRAWS_DIR

TERMS = ["off_diff", "def_diff", "venue", "bowl_net_diff"]
N_FIT_DRAWS = 50


@dataclass
class GameDraws:
    games: pd.DataFrame        # one row per game: game_id, season, week, season_type, home/away ids, venue, label
    x: np.ndarray              # (G, S, len(TERMS)) design per rating draw


def load_season_draws(season: int, directory: Path = DRAWS_DIR) -> dict:
    with np.load(directory / f"cfb_team_week_draws_{season}.npz") as z:
        return {k: z[k] for k in z.files}


def game_draws(games: pd.DataFrame, seasons: list[int], n_draws: int | None = None, seed: int = 0,
               directory: Path = DRAWS_DIR) -> GameDraws:
    """Design arrays for completed or upcoming FBS-vs-FBS games, using each game's own weekly
    cutoff posterior (built only from data before that cutoff)."""
    rng = np.random.default_rng(seed)
    g = games[games["season"].isin(seasons) & (games["home_classification"] == "fbs")
              & (games["away_classification"] == "fbs")].copy()
    rows, xs = [], []
    for season in seasons:
        draws = load_season_draws(season, directory)
        for (st, wk), gg in g[g["season"] == season].groupby(["season_type", "week"]):
            key = f"{st}_{wk}"
            if f"{key}__off" not in draws:
                continue
            off, de, ids = draws[f"{key}__off"], draws[f"{key}__def"], draws[f"{key}__team_ids"]
            col = {t: i for i, t in enumerate(ids)}
            s = off.shape[0]
            pick = np.arange(s) if n_draws is None else rng.choice(s, size=min(n_draws, s), replace=False)
            fcs = col[FCS_TEAM]
            hi = np.array([col.get(t, fcs) for t in gg["home_id"]])
            ai = np.array([col.get(t, fcs) for t in gg["away_id"]])
            od = (off[pick][:, hi] - off[pick][:, ai]).T
            dd = (de[pick][:, hi] - de[pick][:, ai]).T
            venue = np.where(gg["neutral_site"].to_numpy(bool), 0.0, 1.0)[:, None].repeat(len(pick), 1)
            post = float(st == "postseason")
            xs.append(np.stack([od, dd, venue, post * (od + dd)], axis=-1))
            rows.append(gg)
    out = pd.concat(rows, ignore_index=True)
    out["venue"] = np.where(out["neutral_site"], 0, 1)
    out["home_win"] = np.where(out["completed"], (out["home_points"] > out["away_points"]).astype(float), np.nan)
    return GameDraws(games=out, x=np.concatenate(xs, axis=0))


@dataclass
class OutcomeModel:
    gamma: np.ndarray          # (K, len(TERMS)) posterior draws of the coefficients

    def predict(self, x: np.ndarray, interval: float = 0.9, n_gamma: int = 200, chunk: int = 100,
                seed: int = 0) -> pd.DataFrame:
        """x: (G, S, T). Posterior predictive P(home) averages over rating draws and coefficient
        draws; the interval is over the conditional win probability, not the game's outcome.
        Games are processed in chunks against a subsample of coefficient draws: the full
        games x rating draws x coefficient draws array is ~10 GB for one season."""
        rng = np.random.default_rng(seed)
        k = min(n_gamma, len(self.gamma))
        gamma = self.gamma[rng.choice(len(self.gamma), size=k, replace=False)]
        lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
        parts = []
        for start in range(0, x.shape[0], chunk):
            z = np.einsum("gst,kt->gsk", x[start:start + chunk], gamma)
            p = (1 / (1 + np.exp(-z))).reshape(z.shape[0], -1)
            parts.append(np.column_stack([p.mean(1), np.quantile(p, lo, axis=1), np.quantile(p, hi, axis=1)]))
        out = np.vstack(parts) if parts else np.empty((0, 3))
        return pd.DataFrame(out, columns=["p_home", "p_lo", "p_hi"])


def fit(gd: GameDraws, draws: int = 1000, tune: int = 1000, seed: int = 0) -> OutcomeModel:
    keep = gd.games["home_win"].notna().to_numpy()
    x, y = gd.x[keep], gd.games.loc[keep, "home_win"].to_numpy()
    with pm.Model(coords={"term": TERMS}):
        # Ratings are in PPA/play; a 0.1 net edge is roughly a 10-point favourite.
        g = pm.Normal("gamma", mu=[8.0, 8.0, 0.3, 0.0], sigma=[5.0, 5.0, 0.5, 5.0], dims="term")
        z = pt.tensordot(x, g, axes=[[2], [0]])
        p = pt.mean(pm.math.sigmoid(z), axis=1)
        pm.Bernoulli("y", p=pt.clip(p, 1e-6, 1 - 1e-6), observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=4, random_seed=seed, nuts_sampler="nutpie",
                          progressbar=False)
    gamma = idata.posterior.to_dataset()["gamma"].stack(sample=("chain", "draw")).values.T
    return OutcomeModel(gamma=gamma)
