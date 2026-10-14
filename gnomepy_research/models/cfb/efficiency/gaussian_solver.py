"""Exact posterior of the cutoff fit under the Gaussian likelihood.

With hyperparameters fixed, every latent in the non-centred model (z_alpha, z0_off, z0_def,
eps_off, eps_def) has a standard-normal prior and each observation is linear in them:

    y = c + A @ theta + noise,   noise ~ N(0, diag(sigma^2)),   theta ~ N(0, I)

so the posterior is N(m, P^-1) with P = I + A' W A and m = P^-1 A' W (y - c). One Cholesky
of a ~4.5k x 4.5k matrix replaces an MCMC run. Must agree with the PyMC fit of the same model
(tests/sessions/test_cfb_win_probability_models.py and the validation report check it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from gnomepy_research.models.cfb.efficiency.data import ModelData, build_model_data
from gnomepy_research.models.cfb.efficiency.dynamic_efficiency import Hyperparameters
from gnomepy_research.models.cfb.efficiency.inference import CutoffPosterior, Inputs


def _layout(data: ModelData) -> dict[str, int]:
    n_ts, n_k = len(data.ts_keys), data.n_periods
    lay = {"alpha": 0, "z0_off": len(data.alpha_keys)}
    lay["z0_def"] = lay["z0_off"] + n_ts
    lay["eps_off"] = lay["z0_def"] + n_ts
    lay["eps_def"] = lay["eps_off"] + n_ts * (n_k - 1)
    lay["z_conf_off"] = lay["eps_def"] + n_ts * (n_k - 1)
    lay["z_conf_def"] = lay["z_conf_off"] + len(data.conf_keys)
    lay["size"] = lay["z_conf_def"] + len(data.conf_keys)
    if data.drives:
        lay["z_alpha_ppd"] = lay["size"]
        lay["size"] += len(data.seasons)
    return lay


def _state_map(data: ModelData, h: Hyperparameters, lay: dict[str, int], ts: np.ndarray, k: np.ndarray,
               offense: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rows expressing O[ts, k] (or D) as prior mean + B @ theta."""
    n_ts, n_k = len(data.ts_keys), data.n_periods
    x, beta, s, tau, z0, eps, zc, s_conf = (
        (data.x_off, h.beta_off, h.s_off, h.tau_off, "z0_off", "eps_off", "z_conf_off", h.s_conf_off) if offense
        else (data.x_def, h.beta_def, h.s_def, h.tau_def, "z0_def", "eps_def", "z_conf_def", h.s_conf_def))
    widen = np.exp(data.z_unc @ h.gamma)
    B = np.zeros((len(ts), lay["size"]))
    rows = np.arange(len(ts))
    B[rows, lay[z0] + ts] = s * widen[ts]
    if len(data.conf_keys):
        B[rows, lay[zc] + data.ts_conf[ts]] = s_conf
    for r, (i, kk) in enumerate(zip(ts, k)):
        if kk > 0:
            B[r, lay[eps] + i * (n_k - 1): lay[eps] + i * (n_k - 1) + kk] = tau
    return (x @ beta)[ts], B


def _group_unit(data: ModelData) -> np.ndarray:
    return data.alpha_keys["unit"].map({u: i for i, u in enumerate(data.units)}).to_numpy(int)


def _mixed_state_map(data: ModelData, h: Hyperparameters, lay: dict[str, int], ts: np.ndarray,
                     share: np.ndarray, offense: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rows for the share-weighted mix of a team's unit states at each drive observation's period."""
    c = np.zeros(len(data.drv_y))
    B = np.zeros((len(data.drv_y), lay["size"]))
    for u in range(ts.shape[1]):
        cu, Bu = _state_map(data, h, lay, ts[:, u], data.drv_period, offense)
        c += share[:, u] * cu
        B += share[:, u][:, None] * Bu
    return c, B


def _drive_rows(data: ModelData, h: Hyperparameters, lay: dict[str, int]):
    c_off, A_off = _mixed_state_map(data, h, lay, data.drv_off, data.drv_off_w, offense=True)
    c_def, A_def = _mixed_state_map(data, h, lay, data.drv_def, data.drv_def_w, offense=False)
    A = h.lam_ppd * (A_off - A_def)
    A[np.arange(len(data.drv_y)), lay["z_alpha_ppd"] + data.drv_season] += h.sigma_alpha_ppd
    c = h.alpha_ppd + h.lam_ppd * (c_off - c_def) + h.hfa_ppd * data.drv_venue
    w = 1.0 / (h.s_game_ppd ** 2 + h.s_drive_ppd ** 2 / data.drv_n)
    return A, c, w


def solve(data: ModelData, h: Hyperparameters) -> tuple[np.ndarray, tuple, dict[str, int]]:
    lay = _layout(data)
    c_off, A_off = _state_map(data, h, lay, data.obs_off, data.obs_period, offense=True)
    c_def, A_def = _state_map(data, h, lay, data.obs_def, data.obs_period, offense=False)
    A = A_off - A_def
    A[np.arange(len(data.obs_y)), lay["alpha"] + data.obs_alpha] = h.sigma_alpha
    c = np.asarray(h.alpha)[_group_unit(data)][data.obs_alpha] + c_off - c_def + h.hfa * data.obs_venue
    w = 1.0 / (h.sigma_game ** 2 + h.sigma_play ** 2 / data.obs_plays)
    y = data.obs_y
    if data.drives:
        A_d, c_d, w_d = _drive_rows(data, h, lay)
        A, c, w, y = np.vstack([A, A_d]), np.r_[c, c_d], np.r_[w, w_d], np.r_[y, data.drv_y]
    P = np.eye(lay["size"]) + A.T @ (A * w[:, None])
    chol = cho_factor(P, lower=True)
    mean = cho_solve(chol, A.T @ (w * (y - c)))
    return mean, chol, lay


def fit_cutoff_exact(inputs: Inputs, season: int, cutoff: pd.Timestamp, hypers: Hyperparameters,
                     n_draws: int = 2000, seed: int = 0) -> tuple[CutoffPosterior, ModelData]:
    if hypers.likelihood != "normal":
        raise ValueError("the exact solver is only valid for the Gaussian likelihood")
    data = build_model_data(inputs.preseason, inputs.visible(cutoff), inputs.games, [season],
                            units=tuple(hypers.units), conference=hypers.conference,
                            projection=inputs.projection_for(hypers), drop_features=tuple(hypers.drop_features),
                            drives=inputs.drives_for(hypers, cutoff),
                            independents_separate=hypers.independents_separate)
    mean, chol, lay = solve(data, hypers)
    rng = np.random.default_rng(seed)
    # theta = mean + L^-T z, with P = L L^T, gives draws with covariance P^-1.
    L = np.tril(chol[0])
    theta = mean[:, None] + solve_triangular(L.T, rng.standard_normal((lay["size"], n_draws)), lower=False)

    period = int(data.periods.loc[data.periods["cutoff"] == cutoff, "period"].iloc[0])
    ts = np.arange(len(data.ts_keys))
    k = np.full(len(ts), period)
    c_off, B_off = _state_map(data, hypers, lay, ts, k, offense=True)
    c_def, B_def = _state_map(data, hypers, lay, ts, k, offense=False)
    off = (c_off[:, None] + B_off @ theta).T
    de = (c_def[:, None] + B_def @ theta).T
    n_g = len(data.alpha_keys)
    alpha = (np.asarray(hypers.alpha)[_group_unit(data)][:, None]
             + hypers.sigma_alpha * theta[lay["alpha"]:lay["alpha"] + n_g]).T
    return CutoffPosterior(
        season=season, cutoff=cutoff, period=period, team_ids=data.ts_keys["team_id"].to_numpy(),
        units=data.ts_keys["unit"].to_numpy(),
        off=off, de=de, alpha=alpha, hypers=hypers,
        diagnostics={"max_rhat": 1.0, "min_ess_bulk": float(n_draws), "min_ess_tail": float(n_draws),
                     "divergences": 0, "method": "exact_gaussian"},
    ), data
