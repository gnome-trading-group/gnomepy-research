"""Dynamic hierarchical Bayesian model of opponent-adjusted offensive (O) and defensive (D) strength.

    y[g]   ~ StudentT(nu, mu[g], sigma[g])          (or Normal)
    mu[g]  = alpha[season] + O[off, k] - D[def, k] + h * venue
    alpha[season] ~ Normal(alpha, sigma_alpha)    (PPA's league baseline drifts between seasons)
    sigma[g]^2 = sigma_game^2 + sigma_play^2 / plays
    O[i, 0] ~ Normal(x_off[i] @ beta_off, s_off * exp(z[i] @ gamma))
    O[i, k] = O[i, k-1] + tau_off * eps          (non-centered random walk)
    D analogously. Higher O = better offense, higher D = better defense.

Identifiability: features are centred within season, there is no intercept in the state priors,
and alpha[season] carries the league baseline. A season seen for the first time in a cutoff fit
gets alpha as its prior mean, so its baseline is learned from its own games.

Two uses:
- hyperparameter fit: all of the above are free, fitted jointly over training seasons.
- cutoff fit: hyperparameters fixed (posterior means of the hyperparameter fit); only the states
  of one season are sampled, from that season's data before the cutoff.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from gnomepy_research.models.cfb.efficiency.data import ModelData

LIKELIHOODS = ("studentt", "normal")


@dataclass
class Hyperparameters:
    alpha: float
    sigma_alpha: float
    hfa: float
    beta_off: np.ndarray
    beta_def: np.ndarray
    gamma: np.ndarray
    s_off: float
    s_def: float
    tau_off: float
    tau_def: float
    sigma_game: float
    sigma_play: float
    nu: float
    off_names: list[str]
    def_names: list[str]
    likelihood: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Hyperparameters:
        d = dict(d)
        for k in ("beta_off", "beta_def", "gamma"):
            d[k] = np.asarray(d[k], dtype=float)
        return cls(**d)


def build_model(data: ModelData, hypers: Hyperparameters | None = None, likelihood: str = "studentt",
                store_states: bool = True) -> pm.Model:
    """store_states=False skips saving O and D per draw: a multi-season hyperparameter fit would
    otherwise keep GBs of states it never uses."""
    if likelihood not in LIKELIHOODS:
        raise ValueError(likelihood)
    n_ts, n_k = len(data.ts_keys), data.n_periods
    coords = {"ts": np.arange(n_ts), "period": np.arange(n_k), "step": np.arange(n_k - 1), "season": data.seasons,
              "f_off": data.off_names, "f_def": data.def_names, "unc": np.arange(data.z_unc.shape[1])}
    with pm.Model(coords=coords) as model:
        if hypers is None:
            alpha = pm.Normal("alpha", 0.1, 0.1)
            sigma_alpha = pm.HalfNormal("sigma_alpha", 0.05)
            hfa = pm.Normal("hfa", 0.03, 0.03)
            beta_off = pm.Normal("beta_off", 0.0, 0.05, dims="f_off")
            beta_def = pm.Normal("beta_def", 0.0, 0.05, dims="f_def")
            gamma = pm.Normal("gamma", 0.0, 0.3, dims="unc")
            s_off = pm.HalfNormal("s_off", 0.1)
            s_def = pm.HalfNormal("s_def", 0.1)
            tau_off = pm.HalfNormal("tau_off", 0.02)
            tau_def = pm.HalfNormal("tau_def", 0.02)
            sigma_game = pm.HalfNormal("sigma_game", 0.2)
            sigma_play = pm.HalfNormal("sigma_play", 1.5)
            nu = pm.Gamma("nu", 2.0, 0.1) if likelihood == "studentt" else None
        else:
            h = hypers
            alpha, sigma_alpha, hfa = h.alpha, h.sigma_alpha, h.hfa
            beta_off, beta_def, gamma = (pt.as_tensor_variable(h.beta_off), pt.as_tensor_variable(h.beta_def),
                                         pt.as_tensor_variable(h.gamma))
            s_off, s_def, tau_off, tau_def = h.s_off, h.s_def, h.tau_off, h.tau_def
            sigma_game, sigma_play, nu = h.sigma_game, h.sigma_play, h.nu

        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="season")
        alpha_season = pm.Deterministic("alpha_season", alpha + sigma_alpha * z_alpha, dims="season")
        widen = pt.exp(pt.dot(data.z_unc, gamma))
        prior_off = pt.dot(data.x_off, beta_off)
        prior_def = pt.dot(data.x_def, beta_def)

        z0_off = pm.Normal("z0_off", 0.0, 1.0, dims="ts")
        z0_def = pm.Normal("z0_def", 0.0, 1.0, dims="ts")
        eps_off = pm.Normal("eps_off", 0.0, 1.0, dims=("ts", "step"))
        eps_def = pm.Normal("eps_def", 0.0, 1.0, dims=("ts", "step"))

        start_off = prior_off + s_off * widen * z0_off
        start_def = prior_def + s_def * widen * z0_def
        walk_off = pt.concatenate([pt.zeros((n_ts, 1)), pt.cumsum(tau_off * eps_off, axis=1)], axis=1)
        walk_def = pt.concatenate([pt.zeros((n_ts, 1)), pt.cumsum(tau_def * eps_def, axis=1)], axis=1)
        off = start_off[:, None] + walk_off
        de = start_def[:, None] + walk_def
        if store_states:
            off = pm.Deterministic("O", off, dims=("ts", "period"))
            de = pm.Deterministic("D", de, dims=("ts", "period"))

        mu = (alpha_season[data.obs_season] + off[data.obs_off, data.obs_period] - de[data.obs_def, data.obs_period]
              + hfa * data.obs_venue)
        sigma = pt.sqrt(sigma_game ** 2 + sigma_play ** 2 / data.obs_plays)
        if likelihood == "studentt":
            pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=data.obs_y)
        else:
            pm.Normal("y", mu=mu, sigma=sigma, observed=data.obs_y)
    return model


def sample(model: pm.Model, draws: int = 500, tune: int = 1000, chains: int = 4, seed: int = 0,
           target_accept: float = 0.9) -> az.InferenceData:
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, nuts_sampler="nutpie",
                         target_accept=target_accept, progressbar=False)


def hyperparameters_from(idata: az.InferenceData, data: ModelData, likelihood: str) -> Hyperparameters:
    post = idata.posterior

    def m(name: str):
        return post[name].mean(("chain", "draw")).values

    return Hyperparameters(
        alpha=float(m("alpha")), sigma_alpha=float(m("sigma_alpha")), hfa=float(m("hfa")), beta_off=m("beta_off"), beta_def=m("beta_def"),
        gamma=m("gamma"), s_off=float(m("s_off")), s_def=float(m("s_def")), tau_off=float(m("tau_off")),
        tau_def=float(m("tau_def")), sigma_game=float(m("sigma_game")), sigma_play=float(m("sigma_play")),
        nu=float(m("nu")) if likelihood == "studentt" else float("inf"),
        off_names=list(data.off_names), def_names=list(data.def_names), likelihood=likelihood,
    )


def diagnostics(idata: az.InferenceData, var_names: list[str]) -> dict:
    s = az.summary(idata, var_names=var_names, kind="diagnostics")
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
    return {"max_rhat": float(s["r_hat"].max()), "worst_rhat_param": str(s["r_hat"].idxmax()),
            "min_ess_bulk": float(s["ess_bulk"].min()), "worst_ess_param": str(s["ess_bulk"].idxmin()),
            "min_ess_tail": float(s["ess_tail"].min()), "divergences": div}
