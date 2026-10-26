"""
The Bayesian series model, fitted with NUTS (nutpie) on one window.

    rounds_A ~ Binomial(n, sigmoid(D))
    D = mean(skill_A[t]) - mean(skill_B[t]) + beta[A, map] - beta[B, map] + rho[home A] - rho[home B] + eta + eps
    skill[p, 0] ~ N(b * strength_p, sigma0);  skill[p, t] = skill[p, t-1] + tau * N(0, 1)   (monthly random walk)
    beta ~ N(0, sigma_map);  rho[Europe] = rho[unknown] = 0, others ~ N(0, 0.3)
    eta ~ N(0, sigma_m) per map;  eps ~ N(0, sigma_s) per series
    rating_dev[p, map] ~ N(gamma * (skill[p, t] - mean(skill[own lineup, t])) + style_p, sigma_r),  gamma > 0
    maps_A ~ Binomial(maps, sigmoid(EXTRA_MAP_SCALE * (mean(skill[roster A, t]) - mean(skill[roster B, t]) + rho + eps_x)))
        for each extra result (HLTV-less series: maps won only), its log likelihood weighted by extra_weight

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
# A map's win logit is close to this multiple of its round-logit gap: least squares against regulation_map_win under
# the per-map noise (sigma_m ~0.37) over gaps of -1.2..1.2, within 0.02 in probability. Extra results have no rounds,
# and a fixed scale keeps them from moving the round model's own parameters.
EXTRA_MAP_SCALE = 3.22


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
    # Extra (tier-D) results are noisier evidence than HLTV maps: a tournament roster stands in for the five that
    # played, and the stakes are lower. Below 1 their log likelihood counts for less.
    extra_weight: float = 1.0


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

        # Positive: a better player rates above his teammates. Unconstrained, the sign is only weakly fixed (rounds
        # see a lineup's mean, not who in it is better) and the chains split between the two modes.
        gamma = pm.HalfNormal("gamma", 1.0)
        sigma_u = pm.HalfNormal("sigma_u", 0.2)
        style = sigma_u * pm.Normal("style_z", 0, 1, shape=n_p)
        sigma_r = pm.HalfNormal("sigma_r", 0.5)
        own = skill[data.rating_lineup, data.rating_month[:, None]].mean(axis=1)
        mu_r = gamma * (skill[data.rating_player, data.rating_month] - own) + style[data.rating_player]
        pm.Normal("rating", mu_r, sigma_r, observed=data.rating_dev)

        if len(data.extra_maps):
            def roster_mean(lineup, weight):
                return (skill[lineup, data.extra_month[:, None]] * weight).sum(axis=1)

            # Its own series noise: shared with HLTV series, sigma_s fell from ~0.17 to ~0.10 and mixed badly (r_hat
            # 1.29), and pricing an HLTV series uses sigma_s.
            sigma_x = pm.HalfNormal("sigma_extra", 0.5)
            eps_x = sigma_x * pm.Normal("eps_extra_z", 0, 1, shape=len(data.extra_maps))
            gap = (roster_mean(data.extra_lineup_a, data.extra_weight_a) - roster_mean(data.extra_lineup_b, data.extra_weight_b)
                   + rho[data.extra_region_a] - rho[data.extra_region_b] + eps_x)
            logp = pm.logp(pm.Binomial.dist(n=data.extra_maps, p=pm.math.sigmoid(EXTRA_MAP_SCALE * gap)), data.extra_maps_a)
            pm.Potential("extra_results", cfg.extra_weight * logp.sum())
    return model


def sample(model: pm.Model, cfg: BayesSeriesConfig) -> az.InferenceData:
    with model:
        kept = ["a_new", "sigma_new"] if cfg.debut_prior else []
        if "sigma_extra" in model.named_vars:
            kept.append("sigma_extra")
        return pm.sample(draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, random_seed=cfg.seed, nuts_sampler="nutpie",
                         target_accept=cfg.target_accept, progressbar=False,
                         var_names=POSTERIOR_VARS + ["sigma_u"] + kept)


def diagnostics(idata: az.InferenceData) -> dict:
    s = az.summary(idata, var_names=SUMMARY_VARS + [v for v in ("sigma_extra",) if v in idata.posterior], kind="diagnostics")
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
              cfg: BayesSeriesConfig = BayesSeriesConfig(), extra_results: pd.DataFrame | None = None) -> tuple[SeriesPosterior, dict]:
    """Fit on the window_months before `month` and return the posterior that prices `month`'s series."""
    data = build_fit_data(maps, player_stats, rank_strength, month - pd.DateOffset(months=cfg.window_months), month,
                          extra_results)
    idata = sample(build_model(data, cfg), cfg)
    diag = diagnostics(idata)
    diag.update({"maps": len(data.rounds), "players": data.n_players, "rating_obs": len(data.rating_dev),
                 "extra_results": len(data.extra_maps),
                 **{k: float(idata.posterior[k].mean()) for k in ("gamma", "sigma0", "sigma_m", "sigma_s")
                    + (("a_new", "sigma_new") if cfg.debut_prior else ()) + (("sigma_extra",) if len(data.extra_maps) else ())}})
    return posterior_from(idata, data, cfg, month), diag
