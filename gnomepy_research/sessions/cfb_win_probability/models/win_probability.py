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
from gnomepy_research.models.cfb.efficiency.data import UNITS_ALL
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import DRAWS_DIR, draws_name

TERMS_ALL = ["off_diff", "def_diff", "venue", "bowl_net_diff"]
TERMS_SPLIT = ["pass_edge", "rush_edge", "venue", "bowl_edge"]
TERMS = TERMS_ALL
STRETCH_TERM = "edge_abs_edge"
OPTION_TERM = "option_diff"
FORM_TERMS = ["early_success_diff", "late_down_luck_diff", "finishing_luck_diff", "field_position_diff", "pace_x_edge"]
# Ratings are in PPA/play; a 0.1 net edge is roughly a 10-point favourite.
PRIORS: dict[str, tuple[float, float]] = {
    "off_diff": (8.0, 5.0),
    "def_diff": (8.0, 5.0),
    "venue": (0.3, 0.5),
    "bowl_net_diff": (0.0, 5.0),
    "pass_edge": (8.0, 5.0),
    "rush_edge": (8.0, 5.0),
    "bowl_edge": (0.0, 5.0),
    "early_success_diff": (0.0, 0.5),
    "late_down_luck_diff": (0.0, 0.5),
    "finishing_luck_diff": (0.0, 0.5),
    "field_position_diff": (0.0, 0.5),
    "pace_x_edge": (0.0, 3.0),
    "qb_delta_diff": (4.0, 4.0),
    "edge_abs_edge": (0.0, 30.0),
    "option_diff": (0.0, 0.5),
}
N_FIT_DRAWS = 50


@dataclass
class GameDraws:
    games: pd.DataFrame        # one row per game: game_id, season, week, season_type, home/away ids, venue, label
    x: np.ndarray              # (G, S, len(terms)) design per rating draw
    terms: list[str]


def edge_terms(op, dp, orr, dr, share, hi, ai) -> tuple[np.ndarray, np.ndarray]:
    """Rush-share-weighted pass and rush matchup edges, each (G, S). Inputs are (S, T) rating draws."""
    sh, sa = share[hi][:, None], share[ai][:, None]
    rush = sh * (orr[:, hi] - dr[:, ai]).T - sa * (orr[:, ai] - dr[:, hi]).T
    pas = (1 - sh) * (op[:, hi] - dp[:, ai]).T - (1 - sa) * (op[:, ai] - dp[:, hi]).T
    return pas, rush


def form_terms(form_h: np.ndarray, form_a: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """(G, S, 5) form design: four home-minus-away z-score diffs, plus mean pace scaling the (G, S) edge."""
    diff = (form_h[:, :4] - form_a[:, :4])[:, None, :].repeat(edge.shape[1], 1)
    pace = ((form_h[:, 4] + form_a[:, 4]) / 2)[:, None] * edge
    return np.concatenate([diff, pace[..., None]], axis=-1)


def load_season_draws(season: int, directory: Path = DRAWS_DIR, units: tuple[str, ...] = UNITS_ALL,
                      conference: bool = False) -> dict:
    with np.load(directory / f"{draws_name(season, units, conference)}.npz") as z:
        return {k: z[k] for k in z.files}


def team_index(col: dict, gg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fcs = col[FCS_TEAM]
    return (np.array([col.get(t, fcs) for t in gg["home_id"]]),
            np.array([col.get(t, fcs) for t in gg["away_id"]]))


def split_columns(key: str, draws: dict, gg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pass and rush column indices of a split snapshot and each game's home/away index into the pass columns."""
    ids, u = draws[f"{key}__team_ids"], draws[f"{key}__units"]
    p_cols, r_cols = np.flatnonzero(u == "pass"), np.flatnonzero(u == "rush")
    if not np.array_equal(ids[p_cols], ids[r_cols]):
        raise ValueError(f"{key}: pass and rush team columns are not aligned")
    hi, ai = team_index({t: i for i, t in enumerate(ids[p_cols])}, gg)
    return p_cols, r_cols, hi, ai


def game_draws(games: pd.DataFrame, seasons: list[int], n_draws: int | None = None, seed: int = 0,
               directory: Path = DRAWS_DIR, units: tuple[str, ...] = UNITS_ALL, form: bool = False,
               qb: pd.DataFrame | None = None, conference: bool = False, stretch: bool = False,
               option: bool = False) -> GameDraws:
    """Design arrays for completed or upcoming FBS-vs-FBS games, using each game's own weekly
    cutoff posterior (built only from data before that cutoff)."""
    if form and units == UNITS_ALL:
        raise ValueError("form terms are defined on the split model only")
    if (stretch or option) and units == UNITS_ALL:
        raise ValueError("stretch and option terms are defined on the split model only")
    rng = np.random.default_rng(seed)
    g = games[games["season"].isin(seasons) & (games["home_classification"] == "fbs")
              & (games["away_classification"] == "fbs")].copy()
    rows, xs, tails = [], [], []
    for season in seasons:
        draws = load_season_draws(season, directory, units, conference)
        for (st, wk), gg in g[g["season"] == season].groupby(["season_type", "week"]):
            key = f"{st}_{wk}"
            if f"{key}__off" not in draws:
                continue
            off, de, ids = draws[f"{key}__off"], draws[f"{key}__def"], draws[f"{key}__team_ids"]
            s = off.shape[0]
            pick = np.arange(s) if n_draws is None else rng.choice(s, size=min(n_draws, s), replace=False)
            if units == UNITS_ALL:
                hi, ai = team_index({t: i for i, t in enumerate(ids)}, gg)
                main = [(off[pick][:, hi] - off[pick][:, ai]).T, (de[pick][:, hi] - de[pick][:, ai]).T]
            else:
                share_all = draws[f"{key}__rush_share"]
                p_cols, r_cols, hi, ai = split_columns(key, draws, gg)
                main = list(edge_terms(off[pick][:, p_cols], de[pick][:, p_cols], off[pick][:, r_cols],
                                       de[pick][:, r_cols], share_all[p_cols], hi, ai))
            extra, tail = [], []
            if stretch:
                net = main[0] + main[1]
                tail.append((net * np.abs(net))[..., None])
            if option:
                if f"{key}__option" not in draws:
                    raise ValueError(f"{key}: draws have no option flag; snapshot with --conference")
                o = draws[f"{key}__option"][p_cols]
                tail.append(np.broadcast_to((o[hi] - o[ai])[:, None, None], (len(gg), len(pick), 1)))
            if form:
                if f"{key}__form" not in draws:
                    raise ValueError(f"{key}: draws have no form; snapshot with --form")
                f = draws[f"{key}__form"][p_cols]
                extra = [form_terms(f[hi], f[ai], main[0] + main[1])]
            venue = np.where(gg["neutral_site"].to_numpy(bool), 0.0, 1.0)[:, None].repeat(len(pick), 1)
            post = float(st == "postseason")
            base = np.stack([*main, venue, post * (main[0] + main[1])], axis=-1)
            xs.append(np.concatenate([base, *extra], axis=-1))
            tails.append(np.concatenate(tail, axis=-1) if tail else np.empty((len(gg), len(pick), 0)))
            rows.append(gg)
    out = pd.concat(rows, ignore_index=True)
    out["venue"] = np.where(out["neutral_site"], 0, 1)
    out["home_win"] = np.where(out["completed"], (out["home_points"] > out["away_points"]).astype(float), np.nan)
    x = np.concatenate(xs, axis=0)
    terms = TERMS_ALL if units == UNITS_ALL else TERMS_SPLIT + (FORM_TERMS if form else [])
    if qb is not None:
        q = out[["game_id"]].merge(qb, on="game_id", how="left").fillna(0.0)
        qd = (q["home_qb_delta"] - q["away_qb_delta"]).to_numpy()[:, None, None]
        x = np.concatenate([x, np.broadcast_to(qd, (*x.shape[:2], 1))], axis=-1)
        terms = [*terms, "qb_delta_diff"]
    x = np.concatenate([x, np.concatenate(tails, axis=0)], axis=-1)
    terms = terms + [STRETCH_TERM] * stretch + [OPTION_TERM] * option
    return GameDraws(games=out, x=x, terms=terms)


@dataclass
class OutcomeModel:
    gamma: np.ndarray          # (K, len(terms)) posterior draws of the coefficients

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
    terms = gd.terms
    with pm.Model(coords={"term": terms}):
        g = pm.Normal("gamma", mu=[PRIORS[t][0] for t in terms], sigma=[PRIORS[t][1] for t in terms], dims="term")
        z = pt.tensordot(x, g, axes=[[2], [0]])
        p = pt.mean(pm.math.sigmoid(z), axis=1)
        pm.Bernoulli("y", p=pt.clip(p, 1e-6, 1 - 1e-6), observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=4, random_seed=seed, nuts_sampler="nutpie",
                          progressbar=False)
    gamma = idata.posterior.to_dataset()["gamma"].stack(sample=("chain", "draw")).values.T
    return OutcomeModel(gamma=gamma)
