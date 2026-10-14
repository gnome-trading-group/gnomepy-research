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
from scipy.optimize import minimize_scalar
from scipy.special import stdtr

from gnomepy_research.models.cfb.features.physical import zscore_by_season
from gnomepy_research.models.cfb.features.ratings import FCS_TEAM
from gnomepy_research.models.cfb.efficiency.data import UNITS_ALL
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import DRAWS_DIR, draws_name

TERMS_ALL = ["off_diff", "def_diff", "venue", "bowl_net_diff"]
TERMS_SPLIT = ["pass_edge", "rush_edge", "venue", "bowl_edge"]
TERMS = TERMS_ALL
STRETCH_TERM = "edge_abs_edge"
OPTION_TERM = "option_diff"
PHASE_TERMS = ["pass_edge_early", "rush_edge_early"]
EARLY_LAST_WEEK = 4
PHYSICAL_TERMS = ["talent_diff", "bluechip_diff", "trench_diff", "gap_x_edge"]
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
    "talent_diff": (0.0, 0.5),
    "bluechip_diff": (0.0, 0.5),
    "trench_diff": (0.0, 0.5),
    "gap_x_edge": (0.0, 10.0),
    "pass_edge_early": (0.0, 5.0),
    "rush_edge_early": (0.0, 5.0),
}
# Margin model priors in points: a 0.1 net PPA edge is about 12 points, so edge-type log-odds priors
# are scaled by 100/8; z-scored diffs (form, qb, option, physical) get N(0, 6); venue N(3, 3).
MARGIN_EDGE_SCALE = 12.5
MARGIN_EDGE_TERMS = {"off_diff", "def_diff", "bowl_net_diff", "pass_edge", "rush_edge", "bowl_edge", "pace_x_edge",
                     "edge_abs_edge", "gap_x_edge"}
MARGIN_PRIORS: dict[str, tuple[float, float]] = {
    t: ((PRIORS[t][0] * MARGIN_EDGE_SCALE, PRIORS[t][1] * MARGIN_EDGE_SCALE) if t in MARGIN_EDGE_TERMS
        else (3.0, 3.0) if t == "venue" else (0.0, 6.0))
    for t in PRIORS
}
RECAL_BOUNDS = (0.5, 2.0)
N_FIT_DRAWS = 50
PHYSICAL_MIN_MATCH = 0.95


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
                      conference: bool = False, projection: bool = False, drop_features: tuple[str, ...] = (),
                      drives: bool = False, independents_separate: bool = False) -> dict:
    name = draws_name(season, units, conference, projection, drop_features, drives, independents_separate)
    with np.load(directory / f"{name}.npz") as z:
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


def _team_ids(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s).astype("int64")


def physical_match_rate(games: pd.DataFrame, physical: pd.DataFrame) -> float:
    """Share of the games' home and away (season, team) pairs that have a row in the physical table."""
    have = pd.MultiIndex.from_arrays([physical["season"].astype("int64"), _team_ids(physical["team_id"])])
    sides = [pd.MultiIndex.from_arrays([games["season"].astype("int64"), _team_ids(games[c])]) for c in ("home_id", "away_id")]
    hits = sum(int(idx.isin(have).sum()) for idx in sides)
    return hits / (2 * len(games)) if len(games) else 1.0


def physical_diffs(games: pd.DataFrame, physical: pd.DataFrame, min_match: float = 0.0) -> np.ndarray:
    """(G, 3) talent, blue-chip and trench differences, home minus away; teams without a row score 0."""
    rate = physical_match_rate(games, physical)
    if rate < min_match:
        raise ValueError(f"physical table match rate {rate:.1%} < {min_match:.0%} of game team ids; check id dtypes/seasons")
    z = zscore_by_season(physical.assign(team_id=_team_ids(physical["team_id"]))).set_index(["season", "team_id"])

    def side(team_col: str) -> pd.DataFrame:
        idx = pd.MultiIndex.from_arrays([games["season"].astype("int64"), _team_ids(games[team_col])])
        return z.reindex(idx).fillna(0.0).reset_index(drop=True)

    h, a = side("home_id"), side("away_id")
    return np.column_stack([h["talent"] - a["talent"], h["blue_chip_ratio"] - a["blue_chip_ratio"],
                            (h["ol_weight"] - a["dl_weight"]) - (a["ol_weight"] - h["dl_weight"])])


def game_draws(games: pd.DataFrame, seasons: list[int], n_draws: int | None = None, seed: int = 0,
               directory: Path = DRAWS_DIR, units: tuple[str, ...] = UNITS_ALL, form: bool = False,
               qb: pd.DataFrame | None = None, conference: bool = False, stretch: bool = False,
               option: bool = False, physical: pd.DataFrame | None = None, projection: bool = False,
               keep_terms: list[str] | None = None, drop_features: tuple[str, ...] = (),
               drives: bool = False, independents_separate: bool = False, phase: bool = False) -> GameDraws:
    """Design arrays for completed or upcoming FBS-vs-FBS games, using each game's own weekly
    cutoff posterior (built only from data before that cutoff)."""
    if form and units == UNITS_ALL:
        raise ValueError("form terms are defined on the split model only")
    if (stretch or option or phase or physical is not None) and units == UNITS_ALL:
        raise ValueError("stretch, option, phase and physical terms are defined on the split model only")
    rng = np.random.default_rng(seed)
    g = games[games["season"].isin(seasons) & (games["home_classification"] == "fbs")
              & (games["away_classification"] == "fbs")].copy()
    rows, xs, tails, phases = [], [], [], []
    for season in seasons:
        draws = load_season_draws(season, directory, units, conference, projection, drop_features, drives,
                                  independents_separate)
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
            early = float(st == "regular" and wk <= EARLY_LAST_WEEK)
            base = np.stack([*main, venue, post * (main[0] + main[1])], axis=-1)
            xs.append(np.concatenate([base, *extra], axis=-1))
            tails.append(np.concatenate(tail, axis=-1) if tail else np.empty((len(gg), len(pick), 0)))
            phases.append(early * np.stack(main, axis=-1))
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
    if physical is not None:
        d = physical_diffs(out, physical, min_match=PHYSICAL_MIN_MATCH)[:, None, :]
        net = x[..., 0] + x[..., 1]
        x = np.concatenate([x, np.broadcast_to(d, (*x.shape[:2], 3)), (d[..., 0] * net)[..., None]], axis=-1)
        terms = terms + PHYSICAL_TERMS
    if phase:
        x = np.concatenate([x, np.concatenate(phases, axis=0)], axis=-1)
        terms = terms + PHASE_TERMS
    if keep_terms is not None:
        unknown = [t for t in keep_terms if t not in terms]
        if unknown:
            raise ValueError(f"unknown keep_terms {unknown}; available {terms}")
        x = x[..., [terms.index(t) for t in keep_terms]]
        terms = list(keep_terms)
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


@dataclass
class MarginModel:
    beta: np.ndarray           # (K, len(terms)) posterior draws, points per unit of each term
    sigma: np.ndarray          # (K,)
    nu: np.ndarray             # (K,)

    def predict(self, x: np.ndarray, interval: float = 0.9, n_gamma: int = 200, chunk: int = 100,
                seed: int = 0) -> pd.DataFrame:
        """P(home) = P(margin > 0) under StudentT(nu, x @ beta, sigma), averaged over rating and posterior
        draws; chunked like OutcomeModel.predict."""
        rng = np.random.default_rng(seed)
        k = min(n_gamma, len(self.beta))
        idx = rng.choice(len(self.beta), size=k, replace=False)
        beta, sigma, nu = self.beta[idx], self.sigma[idx], self.nu[idx]
        lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
        parts = []
        for start in range(0, x.shape[0], chunk):
            mu = np.einsum("gst,kt->gsk", x[start:start + chunk], beta)
            p = stdtr(nu, mu / sigma).reshape(mu.shape[0], -1)
            parts.append(np.column_stack([p.mean(1), np.quantile(p, lo, axis=1), np.quantile(p, hi, axis=1),
                                          mu.reshape(mu.shape[0], -1).mean(1)]))
        out = np.vstack(parts) if parts else np.empty((0, 4))
        return pd.DataFrame(out, columns=["p_home", "p_lo", "p_hi", "margin_mean"])


def fit_margin(gd: GameDraws, draws: int = 1000, tune: int = 1000, seed: int = 0) -> MarginModel:
    """mu is linear in beta, so the mean over rating draws of x @ beta is (mean over draws of x) @ beta."""
    keep = gd.games["completed"].fillna(False).to_numpy(bool) & gd.games["home_points"].notna().to_numpy() \
        & gd.games["away_points"].notna().to_numpy()
    xbar = gd.x[keep].mean(axis=1)
    margin = (gd.games.loc[keep, "home_points"] - gd.games.loc[keep, "away_points"]).to_numpy(float)
    terms = gd.terms
    with pm.Model(coords={"term": terms}):
        beta = pm.Normal("beta", mu=[MARGIN_PRIORS[t][0] for t in terms], sigma=[MARGIN_PRIORS[t][1] for t in terms],
                         dims="term")
        sigma = pm.HalfNormal("sigma", 20.0)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.1)
        pm.StudentT("margin", nu=nu, mu=pt.dot(xbar, beta), sigma=sigma, observed=margin)
        idata = pm.sample(draws=draws, tune=tune, chains=4, random_seed=seed, nuts_sampler="nutpie",
                          progressbar=False)
    post = idata.posterior.to_dataset().stack(sample=("chain", "draw"))
    return MarginModel(beta=post["beta"].values.T, sigma=post["sigma"].values, nu=post["nu"].values)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_recal(p: np.ndarray, a: float) -> np.ndarray:
    """sigmoid(a * logit(p)); no intercept so a neutral-site swap still gives 1 - p."""
    return 1 / (1 + np.exp(-a * _logit(np.asarray(p, float))))


def fit_recal(p: np.ndarray, y: np.ndarray) -> float:
    z, y = _logit(np.asarray(p, float)), np.asarray(y, float)

    def loss(a: float) -> float:
        q = np.clip(1 / (1 + np.exp(-a * z)), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    return float(minimize_scalar(loss, bounds=RECAL_BOUNDS, method="bounded").x)


def subset_draws(gd: GameDraws, mask: np.ndarray) -> GameDraws:
    return GameDraws(games=gd.games.loc[mask].reset_index(drop=True), x=gd.x[mask], terms=gd.terms)


def cross_fitted_recal(gd: GameDraws, fitter=fit_margin, **fit_kwargs) -> tuple[float, int]:
    """Recalibration slope fitted on the last training season only, scored by a model fitted on the
    other training seasons, so the slope sees out-of-sample predictions like the test season's."""
    season = gd.games["season"].to_numpy()
    last = season == season.max()
    if last.all():
        raise ValueError("cross-fitted recalibration needs at least two training seasons")
    model = fitter(subset_draws(gd, ~last), **fit_kwargs)
    rows = last & gd.games["home_win"].notna().to_numpy()
    p = model.predict(gd.x[rows])["p_home"].to_numpy()
    return fit_recal(p, gd.games.loc[rows, "home_win"].to_numpy()), int(rows.sum())
