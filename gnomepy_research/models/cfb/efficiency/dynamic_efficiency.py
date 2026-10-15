"""Dynamic hierarchical Bayesian model of opponent-adjusted offensive (O) and defensive (D) strength.

    y[g]   ~ StudentT(nu, mu[g], sigma[g])          (or Normal)
    mu[g]  = alpha[season] + O[off, k] - D[def, k] + h * venue
    alpha[season, unit] ~ Normal(alpha[unit], sigma_alpha)    (PPA's league baseline drifts between seasons)
    sigma[g]^2 = sigma_game^2 + sigma_play^2 / plays
    O[i, 0] ~ Normal(x_off[i] @ beta_off, s_off * exp(z[i] @ gamma))   (StudentT(nu_team, ...) with heavy_tails)
    O[i, k] = O[i, k-1] + tau_off * eps          (non-centered random walk)
    D analogously. Higher O = better offense, higher D = better defense.

Optional drive channel (drives=True), a second observation of the same states:
    ppd[g] ~ Normal(alpha_ppd[season] + lam_ppd * (Ocomb[off, k] - Dcomb[def, k]) + hfa_ppd * venue,
                    sqrt(s_game_ppd^2 + s_drive_ppd^2 / n_eff))
with Ocomb/Dcomb the pass-share-weighted mean of a team's unit states (= O/D with units ("all",)).

Optional observation-model extensions (data.EXTENSIONS):
    couple: ppd[g] | PPA rows of the same (game, offense) adds sum_u kappa_ppd[u] * (y_u - mu_u) to its mean,
            the Gaussian conditional when ppd's game noise loads on that game's PPA noise
    fcs:    sigma^2 += s_fcs^2 on PPA rows with an FCS side (s_fcs_ppd^2 on drive rows)
    finish: ppd mean += FIN_O[off team-season] - FIN_D[def team-season], FIN ~ Normal(0, s_fin_off / s_fin_def)

Identifiability: features are centred within season, there is no intercept in the state priors,
and alpha[season] carries the league baseline. A season seen for the first time in a cutoff fit
gets alpha as its prior mean, so its baseline is learned from its own games.

Two uses:
- hyperparameter fit: all of the above are free, fitted jointly over training seasons.
- cutoff fit: hyperparameters fixed (posterior means of the hyperparameter fit); only the states
  of one season are sampled, from that season's data before the cutoff.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from gnomepy_research.models.cfb.efficiency.data import (
    EXT_COUPLE, EXT_FCS, EXT_FINISH, TAU_PHASES, ModelData, step_phases)
from gnomepy_research.models.cfb.features.drives import GARBAGE_WEIGHT

LIKELIHOODS = ("studentt", "normal")
CONFERENCE_FIELDS = ("s_conf_off", "s_conf_def", "conference")
TAU_PHASE_PRIORS = (0.03, 0.03, 0.03, 0.06)
DRIVE_FIELDS = ("alpha_ppd", "sigma_alpha_ppd", "lam_ppd", "hfa_ppd", "s_game_ppd", "s_drive_ppd", "drives")
EXTENSION_FIELDS = {EXT_COUPLE: ("kappa_ppd",), EXT_FCS: ("s_fcs", "s_fcs_ppd"), EXT_FINISH: ("s_fin_off", "s_fin_def")}


@dataclass
class Hyperparameters:
    alpha: np.ndarray
    sigma_alpha: float
    hfa: float
    beta_off: np.ndarray
    beta_def: np.ndarray
    gamma: np.ndarray
    s_off: float
    s_def: float
    tau_off: float | np.ndarray
    tau_def: float | np.ndarray
    sigma_game: float
    sigma_play: float
    nu: float
    off_names: list[str]
    def_names: list[str]
    likelihood: str
    units: list[str] = field(default_factory=lambda: ["all"])
    s_conf_off: float = 0.0
    s_conf_def: float = 0.0
    conference: bool = False
    projection: bool = False
    drop_features: list[str] = field(default_factory=list)
    alpha_ppd: float = 0.0
    sigma_alpha_ppd: float = 0.0
    lam_ppd: float = 0.0
    hfa_ppd: float = 0.0
    s_game_ppd: float = 0.0
    s_drive_ppd: float = 0.0
    drives: bool = False
    independents_separate: bool = False
    tau_phases: bool = False
    heavy_tails: bool = False
    nu_team: float = float("inf")
    garbage_weight: float = GARBAGE_WEIGHT
    extensions: list[str] = field(default_factory=list)
    kappa_ppd: list[float] = field(default_factory=list)
    s_fcs: float = 0.0
    s_fcs_ppd: float = 0.0
    s_fin_off: float = 0.0
    s_fin_def: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if not self.extensions:
            d.pop("extensions")
        for ext, keys in EXTENSION_FIELDS.items():
            if ext not in self.extensions:
                for k in keys:
                    d.pop(k)
        if not self.conference:
            for k in CONFERENCE_FIELDS:
                d.pop(k)
        if not self.projection:
            d.pop("projection")
        if not self.drop_features:
            d.pop("drop_features")
        if not self.drives:
            for k in DRIVE_FIELDS:
                d.pop(k)
        if not self.independents_separate:
            d.pop("independents_separate")
        if not self.tau_phases:
            d.pop("tau_phases")
        if not self.heavy_tails:
            d.pop("heavy_tails")
            d.pop("nu_team")
        if self.garbage_weight == GARBAGE_WEIGHT:
            d.pop("garbage_weight")
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Hyperparameters:
        d = dict(d)
        for k in ("beta_off", "beta_def", "gamma"):
            d[k] = np.asarray(d[k], dtype=float)
        d["alpha"] = np.atleast_1d(np.asarray(d["alpha"], dtype=float))
        d.setdefault("units", ["all"])
        if d.get("tau_phases"):
            d["tau_off"] = np.asarray(d["tau_off"], dtype=float)
            d["tau_def"] = np.asarray(d["tau_def"], dtype=float)
        if "kappa_ppd" in d:
            d["kappa_ppd"] = [float(v) for v in d["kappa_ppd"]]
        return cls(**d)


def build_model(data: ModelData, hypers: Hyperparameters | None = None, likelihood: str = "studentt",
                store_states: bool = True, tau_phases: bool = False, heavy_tails: bool = False) -> pm.Model:
    """store_states=False skips saving O and D per draw: a multi-season hyperparameter fit would
    otherwise keep GBs of states it never uses. tau_phases (free fit; fixed hypers carry their own
    flag) gives the walk one drift rate per season phase, TAU_PHASES. heavy_tails (likewise) gives
    the start states a StudentT(nu_team) prior so genuine outliers are shrunk less."""
    if likelihood not in LIKELIHOODS:
        raise ValueError(likelihood)
    if hypers is not None and tuple(hypers.extensions) != tuple(data.extensions):
        raise ValueError(f"hyperparameter extensions {hypers.extensions} != data extensions {data.extensions}")
    n_ts, n_k = len(data.ts_keys), data.n_periods
    has_conf = len(data.conf_keys) > 0
    ext = set(data.extensions)
    coords = {"ts": np.arange(n_ts), "period": np.arange(n_k), "step": np.arange(n_k - 1), "alpha_group": np.arange(len(data.alpha_keys)), "unit": list(data.units),

              "f_off": data.off_names, "f_def": data.def_names, "unc": np.arange(data.z_unc.shape[1])}
    if has_conf:
        coords["conf"] = np.arange(len(data.conf_keys))
    if data.drives:
        coords["season"] = list(data.seasons)
    if EXT_FINISH in ext:
        coords["base"] = np.arange(data.n_base)
    if hypers is not None:
        tau_phases = hypers.tau_phases
        heavy_tails = hypers.heavy_tails
    if tau_phases:
        coords["tau_phase"] = list(TAU_PHASES)
    with pm.Model(coords=coords) as model:
        if hypers is None:
            alpha = pm.Normal("alpha", 0.1, 0.1, dims="unit")
            sigma_alpha = pm.HalfNormal("sigma_alpha", 0.05)
            hfa = pm.Normal("hfa", 0.03, 0.03)
            beta_off = pm.Normal("beta_off", 0.0, 0.05, dims="f_off")
            beta_def = pm.Normal("beta_def", 0.0, 0.05, dims="f_def")
            gamma = pm.Normal("gamma", 0.0, 0.3, dims="unc")
            s_off = pm.HalfNormal("s_off", 0.1)
            s_def = pm.HalfNormal("s_def", 0.1)
            if tau_phases:
                tau_off = pm.HalfNormal("tau_off", np.array(TAU_PHASE_PRIORS), dims="tau_phase")
                tau_def = pm.HalfNormal("tau_def", np.array(TAU_PHASE_PRIORS), dims="tau_phase")
            else:
                tau_off = pm.HalfNormal("tau_off", 0.02)
                tau_def = pm.HalfNormal("tau_def", 0.02)
            sigma_game = pm.HalfNormal("sigma_game", 0.2)
            sigma_play = pm.HalfNormal("sigma_play", 1.5)
            nu = pm.Gamma("nu", 2.0, 0.1) if likelihood == "studentt" else None
            nu_team = pm.Gamma("nu_team", 2.0, 0.2) if heavy_tails else None
            if has_conf:
                s_conf_off = pm.HalfNormal("s_conf_off", 0.05)
                s_conf_def = pm.HalfNormal("s_conf_def", 0.05)
            if data.drives:
                alpha_ppd = pm.Normal("alpha_ppd", 2.2, 0.5)
                sigma_alpha_ppd = pm.HalfNormal("sigma_alpha_ppd", 0.3)
                lam_ppd = pm.Normal("lam_ppd", 12.0, 6.0)
                hfa_ppd = pm.Normal("hfa_ppd", 0.2, 0.2)
                s_game_ppd = pm.HalfNormal("s_game_ppd", 1.0)
                s_drive_ppd = pm.HalfNormal("s_drive_ppd", 3.0)
            if EXT_COUPLE in ext:
                kappa_ppd = pm.Normal("kappa_ppd", 2.0, 2.0, dims="unit")
            if EXT_FCS in ext:
                s_fcs = pm.HalfNormal("s_fcs", 0.1)
                s_fcs_ppd = pm.HalfNormal("s_fcs_ppd", 1.0) if data.drives else 0.0
            if EXT_FINISH in ext:
                s_fin_off = pm.HalfNormal("s_fin_off", 0.5)
                s_fin_def = pm.HalfNormal("s_fin_def", 0.5)
        else:
            h = hypers
            alpha, sigma_alpha, hfa = pt.as_tensor_variable(np.asarray(h.alpha, dtype=float)), h.sigma_alpha, h.hfa
            beta_off, beta_def, gamma = (pt.as_tensor_variable(h.beta_off), pt.as_tensor_variable(h.beta_def),
                                         pt.as_tensor_variable(h.gamma))
            s_off, s_def, tau_off, tau_def = h.s_off, h.s_def, h.tau_off, h.tau_def
            if tau_phases:
                tau_off, tau_def = pt.as_tensor_variable(np.asarray(tau_off)), pt.as_tensor_variable(np.asarray(tau_def))
            sigma_game, sigma_play, nu, nu_team = h.sigma_game, h.sigma_play, h.nu, h.nu_team
            s_conf_off, s_conf_def = h.s_conf_off, h.s_conf_def
            alpha_ppd, sigma_alpha_ppd, lam_ppd = h.alpha_ppd, h.sigma_alpha_ppd, h.lam_ppd
            hfa_ppd, s_game_ppd, s_drive_ppd = h.hfa_ppd, h.s_game_ppd, h.s_drive_ppd
            kappa_ppd = pt.as_tensor_variable(np.asarray(h.kappa_ppd, dtype=float))
            s_fcs, s_fcs_ppd, s_fin_off, s_fin_def = h.s_fcs, h.s_fcs_ppd, h.s_fin_off, h.s_fin_def

        group_unit = data.alpha_keys["unit"].map({u: i for i, u in enumerate(data.units)}).to_numpy(int)
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="alpha_group")
        alpha_season = pm.Deterministic("alpha_season", alpha[group_unit] + sigma_alpha * z_alpha, dims="alpha_group")
        widen = pt.exp(pt.dot(data.z_unc, gamma))
        prior_off = pt.dot(data.x_off, beta_off)
        prior_def = pt.dot(data.x_def, beta_def)

        if heavy_tails:
            z0_off = pm.StudentT("z0_off", nu=nu_team, mu=0.0, sigma=1.0, dims="ts")
            z0_def = pm.StudentT("z0_def", nu=nu_team, mu=0.0, sigma=1.0, dims="ts")
        else:
            z0_off = pm.Normal("z0_off", 0.0, 1.0, dims="ts")
            z0_def = pm.Normal("z0_def", 0.0, 1.0, dims="ts")
        eps_off = pm.Normal("eps_off", 0.0, 1.0, dims=("ts", "step"))
        eps_def = pm.Normal("eps_def", 0.0, 1.0, dims=("ts", "step"))

        start_off = prior_off + s_off * widen * z0_off
        start_def = prior_def + s_def * widen * z0_def
        if has_conf:
            z_conf_off = pm.Normal("z_conf_off", 0.0, 1.0, dims="conf")
            z_conf_def = pm.Normal("z_conf_def", 0.0, 1.0, dims="conf")
            start_off = start_off + s_conf_off * z_conf_off[data.ts_conf]
            start_def = start_def + s_conf_def * z_conf_def[data.ts_conf]
        if tau_phases:
            phase = step_phases(data.periods, data.ts_keys["season"].to_numpy(), n_k)
            tau_off, tau_def = tau_off[phase], tau_def[phase]
        walk_off = pt.concatenate([pt.zeros((n_ts, 1)), pt.cumsum(tau_off * eps_off, axis=1)], axis=1)
        walk_def = pt.concatenate([pt.zeros((n_ts, 1)), pt.cumsum(tau_def * eps_def, axis=1)], axis=1)
        off = start_off[:, None] + walk_off
        de = start_def[:, None] + walk_def
        if store_states:
            off = pm.Deterministic("O", off, dims=("ts", "period"))
            de = pm.Deterministic("D", de, dims=("ts", "period"))

        mu = (alpha_season[data.obs_alpha] + off[data.obs_off, data.obs_period] - de[data.obs_def, data.obs_period]
              + hfa * data.obs_venue)
        var = sigma_game ** 2 + sigma_play ** 2 / data.obs_plays
        if EXT_FCS in ext:
            var = var + s_fcs ** 2 * data.obs_fcs
        sigma = pt.sqrt(var)
        if likelihood == "studentt":
            pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=data.obs_y)
        else:
            pm.Normal("y", mu=mu, sigma=sigma, observed=data.obs_y)
        if data.drives:
            z_alpha_ppd = pm.Normal("z_alpha_ppd", 0.0, 1.0, dims="season")
            alpha_ppd_season = alpha_ppd + sigma_alpha_ppd * z_alpha_ppd
            k = data.drv_period[:, None]
            o_comb = (off[data.drv_off, k] * data.drv_off_w).sum(axis=1)
            d_comb = (de[data.drv_def, k] * data.drv_def_w).sum(axis=1)
            mu_ppd = alpha_ppd_season[data.drv_season] + lam_ppd * (o_comb - d_comb) + hfa_ppd * data.drv_venue
            if EXT_FINISH in ext:
                fin_o = s_fin_off * pm.Normal("z_fin_off", 0.0, 1.0, dims="base")
                fin_d = s_fin_def * pm.Normal("z_fin_def", 0.0, 1.0, dims="base")
                if store_states:
                    fin_o = pm.Deterministic("FIN_O", fin_o, dims="base")
                    fin_d = pm.Deterministic("FIN_D", fin_d, dims="base")
                mu_ppd = mu_ppd + fin_o[data.drv_base_off] - fin_d[data.drv_base_def]
            if EXT_COUPLE in ext:
                resid = pt.concatenate([data.obs_y - mu, pt.zeros(1)])
                link = np.where(data.drv_obs < 0, len(data.obs_y), data.drv_obs)
                mu_ppd = mu_ppd + (resid[link] * kappa_ppd[None, :]).sum(axis=1)
            var_ppd = s_game_ppd ** 2 + s_drive_ppd ** 2 / data.drv_n
            if EXT_FCS in ext:
                var_ppd = var_ppd + s_fcs_ppd ** 2 * data.drv_fcs
            pm.Normal("ppd", mu=mu_ppd, sigma=pt.sqrt(var_ppd), observed=data.drv_y)
    return model


def sample(model: pm.Model, draws: int = 500, tune: int = 1000, chains: int = 4, seed: int = 0,
           target_accept: float = 0.9) -> az.InferenceData:
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, nuts_sampler="nutpie",
                         target_accept=target_accept, progressbar=False)


def hyperparameters_from(idata: az.InferenceData, data: ModelData, likelihood: str,
                         tau_phases: bool = False, heavy_tails: bool = False,
                         garbage_weight: float = GARBAGE_WEIGHT) -> Hyperparameters:
    post = idata.posterior

    def m(name: str):
        return post[name].mean(("chain", "draw")).values

    return Hyperparameters(
        alpha=np.atleast_1d(m("alpha")), sigma_alpha=float(m("sigma_alpha")), hfa=float(m("hfa")), beta_off=m("beta_off"), beta_def=m("beta_def"),
        gamma=m("gamma"), s_off=float(m("s_off")), s_def=float(m("s_def")), tau_off=m("tau_off") if tau_phases else float(m("tau_off")),
        tau_def=m("tau_def") if tau_phases else float(m("tau_def")), sigma_game=float(m("sigma_game")), sigma_play=float(m("sigma_play")),
        nu=float(m("nu")) if likelihood == "studentt" else float("inf"),
        off_names=list(data.off_names), def_names=list(data.def_names), likelihood=likelihood,
        units=list(data.units), projection=data.projection, drop_features=list(data.drop_features),
        **({"s_conf_off": float(m("s_conf_off")), "s_conf_def": float(m("s_conf_def")), "conference": True}
           if len(data.conf_keys) else {}),
        **({k: float(m(k)) for k in DRIVE_FIELDS[:-1]} | {"drives": True} if data.drives else {}),
        independents_separate=data.independents_separate, tau_phases=tau_phases,
        **({"heavy_tails": True, "nu_team": float(m("nu_team"))} if heavy_tails else {}),
        garbage_weight=garbage_weight,
        **_extension_values(m, data),
    )


def _extension_values(m, data: ModelData) -> dict:
    ext = list(data.extensions)
    out = {"extensions": ext} if ext else {}
    if EXT_COUPLE in ext:
        out["kappa_ppd"] = [float(v) for v in np.atleast_1d(m("kappa_ppd"))]
    if EXT_FCS in ext:
        out["s_fcs"] = float(m("s_fcs"))
        out["s_fcs_ppd"] = float(m("s_fcs_ppd")) if data.drives else 0.0
    if EXT_FINISH in ext:
        out["s_fin_off"], out["s_fin_def"] = float(m("s_fin_off")), float(m("s_fin_def"))
    return out


def diagnostics(idata: az.InferenceData, var_names: list[str]) -> dict:
    s = az.summary(idata, var_names=var_names, kind="diagnostics")
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
    return {"max_rhat": float(s["r_hat"].max()), "worst_rhat_param": str(s["r_hat"].idxmax()),
            "min_ess_bulk": float(s["ess_bulk"].min()), "worst_ess_param": str(s["ess_bulk"].idxmin()),
            "min_ess_tail": float(s["ess_tail"].min()), "divergences": div}
