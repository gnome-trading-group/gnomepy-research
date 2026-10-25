"""
The Bayesian series model, fitted with NUTS (nutpie) on one window.

    rounds_A ~ Binomial(n, sigmoid(D))
    D = mean(skill_A[t]) - mean(skill_B[t]) + beta[A, map] - beta[B, map] + rho[home A] - rho[home B] + eta + eps
    skill[p, 0] ~ N(b * strength_p, sigma0);  skill[p, t] = skill[p, t-1] + tau * N(0, 1)   (monthly random walk)
    beta ~ N(0, sigma_map);  rho[Europe] = rho[unknown] = 0, others ~ N(0, 0.3)
    eta ~ N(0, sigma_m) per map;  eps ~ N(0, sigma_s) per series
    rating_dev[p, map] ~ N(gamma * (skill[p, t] - mean(skill[own lineup, t])) + style_p, sigma_r),  gamma > 0

With several stats (BayesSeriesConfig.stats), each standardized deviation gets its own gamma_k > 0 and style, plus
AWPer/IGL offsets per stat (roles=True), and their residuals share an LKJ covariance: the stats are built from the
same kills and damage, so treating them as independent would count one piece of evidence several times.

Rounds identify a lineup's mean, not who in a stable five is better; the ratings relative to teammates do, which
is what helps most when a lineup changes (stand-ins, transfers).

Evaluated walk-forward on Mar-Sep 2026 (blend weights fitted on Sep 2025 - Feb 2026): blended with the pre-map
model it scored 0.5774 series log loss against 0.5854 for the pre-map model alone (0.5784 without the debut prior).
"""
from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from gnomepy_research.models.cs2.bayes.data import FitData, build_fit_data
from gnomepy_research.models.cs2.hltv.bayes_series import REGIONS, RankStrength, SeriesPosterior

POSTERIOR_VARS = ["theta_last", "beta", "rho", "sigma_m", "sigma_s", "b", "sigma0", "sigma_map", "gamma", "sigma_r"]
SUMMARY_VARS = ["sigma0", "b", "sigma_map", "rho", "sigma_m", "sigma_s", "gamma", "sigma_u", "sigma_r"]


@dataclass(frozen=True)
class BayesSeriesConfig:
    window_months: int = 12
    # Fixed at the mean of its estimates over Sep 2025 - Feb 2026 (0.038-0.077): free, it sat in a funnel with
    # tens of thousands of random-walk steps (r_hat 1.34, ess 10) while the skills themselves converged.
    tau: float = 0.055
    draws: int = 500
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    seed: int = 0
    # Debutants start from their own prior (offset a_new, about -0.35 round-logit, spread sigma_new) instead of the
    # regular one, which rates an unknown newcomer on a good team too highly. Confirm Mar-Sep 2026: -0.0010* vs off.
    debut_prior: bool = True
    # Per-player map stats in the skill likelihood (data.STAT_NAMES); ("rating",) is the model above.
    stats: tuple[str, ...] = ("rating",)
    # Per-stat offsets for the AWPer and the in-game leader, whose stat lines differ by role rather than skill.
    roles: bool = False

    @property
    def multi_stat(self) -> bool:
        return self.stats != ("rating",) or self.roles


def build_model(data: FitData, cfg: BayesSeriesConfig) -> pm.Model:
    n_p, n_t = data.n_players, len(data.months)
    with pm.Model() as model:
        sigma0 = pm.HalfNormal("sigma0", 0.5)
        b = pm.Normal("b", 0, 1)
        z0 = pm.Normal("z0", 0, 1, shape=n_p)
        steps = pm.Normal("steps", 0, 1, shape=(n_p, max(n_t - 1, 0)))
        walk = pt.concatenate([pt.zeros((n_p, 1)), pt.cumsum(steps, axis=1)], axis=1)
        start = b * data.strength + sigma0 * z0
        if cfg.debut_prior:
            a_new = pm.Normal("a_new", 0, 0.5)
            sigma_new = pm.HalfNormal("sigma_new", 0.5)
            start = pt.switch(data.debutant, a_new + b * data.strength + sigma_new * z0, start)
        skill = start[:, None] + cfg.tau * walk
        pm.Deterministic("theta_last", skill[:, -1])

        sigma_map = pm.HalfNormal("sigma_map", 0.3)
        beta = pm.Deterministic("beta", sigma_map * pm.Normal("beta_z", 0, 1, shape=len(data.tm_maps)))
        rho_free = pm.Normal("rho_free", 0, 0.3, shape=len(REGIONS) - 1)
        rho = pm.Deterministic("rho", pt.concatenate([pt.zeros(2), rho_free]))
        sigma_m = pm.HalfNormal("sigma_m", 0.5)
        eta = sigma_m * pm.Normal("eta_z", 0, 1, shape=len(data.rounds))
        sigma_s = pm.HalfNormal("sigma_s", 0.5)
        eps = sigma_s * pm.Normal("eps_z", 0, 1, shape=data.n_series)

        d = (skill[data.lineup_a, data.month[:, None]].mean(axis=1) - skill[data.lineup_b, data.month[:, None]].mean(axis=1)
             + beta[data.tm_a] - beta[data.tm_b] + rho[data.region_a] - rho[data.region_b] + eta + eps[data.series])
        pm.Binomial("rounds", n=data.rounds, p=pm.math.sigmoid(d), observed=data.rounds_a)

        # Positive: a better player rates above their teammates. Unconstrained, the sign is only weakly fixed (rounds
        # see a lineup's mean, not who in it is better) and the chains split between the two modes.
        own = skill[data.rating_lineup, data.rating_month[:, None]].mean(axis=1)
        rel = skill[data.rating_player, data.rating_month] - own
        if not cfg.multi_stat:
            gamma = pm.HalfNormal("gamma", 1.0)
            sigma_u = pm.HalfNormal("sigma_u", 0.2)
            style = sigma_u * pm.Normal("style_z", 0, 1, shape=n_p)
            sigma_r = pm.HalfNormal("sigma_r", 0.5)
            pm.Normal("rating", gamma * rel + style[data.rating_player], sigma_r, observed=data.rating_dev)
            return model
        k = len(cfg.stats)
        gamma = pm.HalfNormal("gamma", 1.0, shape=k)
        sigma_u = pm.HalfNormal("sigma_u", 0.5, shape=k)
        style = sigma_u * pm.Normal("style_z", 0, 1, shape=(n_p, k))
        mu = gamma[None, :] * rel[:, None] + style[data.rating_player]
        if cfg.roles:
            role = pm.Normal("role", 0, 1, shape=(2, k))
            mu = mu + data.rating_awp[:, None] * role[0] + data.rating_igl[:, None] * role[1]
        chol, corr, sds = pm.LKJCholeskyCov("stat_cov", n=k, eta=2.0, sd_dist=pm.HalfNormal.dist(1.0, shape=k),
                                            compute_corr=True)
        pm.Deterministic("sigma_r", sds)
        pm.Deterministic("stat_corr", corr)
        pm.MvNormal("stats", mu=mu, chol=chol, observed=data.rating_dev)
    return model


def sample(model: pm.Model, cfg: BayesSeriesConfig) -> az.InferenceData:
    with model:
        extra = (["a_new", "sigma_new"] if cfg.debut_prior else []) + (["stat_corr"] if cfg.multi_stat else []) \
            + (["role"] if cfg.roles else [])
        return pm.sample(draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, random_seed=cfg.seed, nuts_sampler="nutpie",
                         target_accept=cfg.target_accept, progressbar=False,
                         var_names=POSTERIOR_VARS + ["sigma_u"] + extra)


def diagnostics(idata: az.InferenceData) -> dict:
    s = az.summary(idata, var_names=SUMMARY_VARS, kind="diagnostics")
    skills = az.summary(idata, var_names=["theta_last"], kind="diagnostics")
    return {"max_rhat": float(s["r_hat"].max()), "worst_rhat_param": str(s["r_hat"].idxmax()),
            "min_ess_bulk": float(s["ess_bulk"].min()), "skill_max_rhat": float(skills["r_hat"].max()),
            "skill_min_ess_bulk": float(skills["ess_bulk"].min()),
            "divergences": int(idata.sample_stats["diverging"].sum())}


def posterior_from(idata: az.InferenceData, data: FitData, cfg: BayesSeriesConfig, as_of: pd.Timestamp) -> SeriesPosterior:
    post = idata.posterior

    def draws(name: str) -> np.ndarray:
        return post[name].stack(s=("chain", "draw")).transpose("s", ...).to_numpy()

    return SeriesPosterior(
        player_ids=data.player_ids, theta=draws("theta_last"), tau=cfg.tau, b=draws("b"), sigma0=draws("sigma0"),
        strength_mu=data.strength_mu, strength_sd=data.strength_sd, rho=draws("rho"), sigma_m=draws("sigma_m"),
        sigma_s=draws("sigma_s"),
        home_team_ids=np.array(list(data.home), dtype=np.int64), home_region=np.array(list(data.home.values()), dtype=np.int64),
        tm_team_ids=data.tm_team_ids, tm_maps=data.tm_maps, beta=draws("beta"),
        plays_team_ids=data.plays.team_id.to_numpy(np.int64), plays_maps=data.plays.map_name.to_numpy(str),
        plays_counts=data.plays.maps.to_numpy(np.int64),
        a_new=draws("a_new") if cfg.debut_prior else np.zeros_like(draws("b")),
        sigma_new=draws("sigma_new") if cfg.debut_prior else draws("sigma0"),
        career_ids=data.career_ids,
        as_of=f"{pd.Timestamp(as_of):%Y-%m-%d}",
    )


def fit_month(maps: pd.DataFrame, player_stats: pd.DataFrame, rank_strength: RankStrength, month: pd.Timestamp,
              cfg: BayesSeriesConfig = BayesSeriesConfig()) -> tuple[SeriesPosterior, dict]:
    """Fit on the window_months before `month` and return the posterior that prices `month`'s series."""
    data = build_fit_data(maps, player_stats, rank_strength, month - pd.DateOffset(months=cfg.window_months), month,
                          cfg.stats)
    idata = sample(build_model(data, cfg), cfg)
    diag = diagnostics(idata)
    diag.update({"maps": len(data.rounds), "players": data.n_players, "rating_obs": len(data.rating_dev),
                 **{k: float(idata.posterior[k].mean()) for k in ("sigma0", "sigma_m", "sigma_s")
                    + (("a_new", "sigma_new") if cfg.debut_prior else ())},
                 **{k: np.round(idata.posterior[k].mean(("chain", "draw")).to_numpy(), 3).tolist()
                    for k in ["gamma"] + (["sigma_r"] if cfg.multi_stat else []) + (["role"] if cfg.roles else [])}})
    return posterior_from(idata, data, cfg, month), diag
