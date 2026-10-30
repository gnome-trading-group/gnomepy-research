"""A ridge-penalised binomial paired comparison on regulation rounds, solved deterministically with L-BFGS.

    rounds_A ~ Binomial(n, sigmoid(D)),  D = mean(s_A) - mean(s_B) + t_A - t_B
    s_p = theta_p + gamma . x_p          player skill: a ridge deviation around a covariate prior
    t_team                               ridge team effect, what a five does beyond its players

A FACEIT map is a separate measurement: it sees theta_p + delta_p per player, t_team + u_team per team and its own
covariate slopes, where delta and u are ridge-penalised FACEIT-specific deviations. FACEIT rounds therefore move the
HLTV-level parameters only through the coupling lam_delta sets (large: one shared skill; small: FACEIT results
barely transfer), and HLTV series are priced from theta, t and the HLTV slopes alone. Sharing everything distorted
HLTV pricing in the Jan-Mar 2026 walk-forward (0.634 vs 0.612 with identical lineups), while FACEIT lineups helped.

Each map's log likelihood is weighted by an exponential decay in its age (drift), a source weight (FACEIT rounds are
a noisier measurement of HLTV-level strength) and kappa (rounds within a map are correlated, so a map carries fewer
than n independent rounds). The covariate slopes gamma are shared, so a player with no maps in the window is priced
from the prior alone, which is the hierarchical fallback for thin coverage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e18.data import LINEUP, Observations, covariates, player_prior_inputs

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
_GH_W = _GH_W / _GH_W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)


@dataclass(frozen=True)
class Config:
    half_life_days: float = 365.0
    window_days: int = 365
    lam_player: float = 4.0
    lam_team: float = 4.0
    lam_cov: float = 0.01
    kappa: float = 0.35
    faceit_weight: float = 0.5
    lam_delta: float = 0.5
    sigma_map: float = 0.37
    sigma_series: float = 0.3
    faceit: bool = True
    # With faceit_rows off, FACEIT still supplies lineups and the skill-level and division prior covariates, but its
    # rounds are not fitted.
    faceit_rows: bool = True
    # FACEIT skill level and division as prior covariates in the HLTV block. Off, HLTV priors rest on rank alone
    # (the FACEIT block keeps its own slopes on every covariate).
    faceit_priors: bool = False
    roster_moves: bool = True


@dataclass
class Fit:
    cfg: Config
    obs: Observations
    cutoff: pd.Timestamp
    players: dict[str, int]
    theta: np.ndarray
    teams: dict[str, int]
    team_effect: np.ndarray
    gamma: np.ndarray
    mu: float
    sd: float
    player_cov: np.ndarray
    cov_mask: np.ndarray
    map_table: np.ndarray
    info: dict


def fit_model(obs: Observations, cutoff: pd.Timestamp, cfg: Config) -> Fit:
    t0 = time.time()
    maps = obs.maps if cfg.faceit_rows else obs.maps[obs.maps.source == "hltv"].reset_index(drop=True)
    pri = player_prior_inputs(obs, maps).sort_values("player", kind="stable").reset_index(drop=True)
    x_cov, mu, sd = covariates(obs, pri.player.tolist(), pri.team.tolist(), pri.day.tolist())
    cov_mask = np.array([1.0, 1.0] + [float(cfg.faceit_priors)] * (x_cov.shape[1] - 2))
    pidx = {p: i for i, p in enumerate(pri.player)}
    teams = sorted(set(maps.team_a) | set(maps.team_b))
    tidx = {t: i for i, t in enumerate(teams)}
    n_maps, n_p, n_t, n_c = len(maps), len(pidx), len(teams), x_cov.shape[1]

    rows = np.repeat(np.arange(n_maps), LINEUP)
    cols_a = np.array([pidx[p] for five in maps.lineup_a for p in five])
    cols_b = np.array([pidx[p] for five in maps.lineup_b for p in five])
    x_players = sp.csr_matrix((np.concatenate([np.full(n_maps * LINEUP, 1 / LINEUP), np.full(n_maps * LINEUP, -1 / LINEUP)]),
                               (np.concatenate([rows, rows]), np.concatenate([cols_a, cols_b]))), shape=(n_maps, n_p))
    x_teams = sp.csr_matrix((np.concatenate([np.ones(n_maps), -np.ones(n_maps)]),
                             (np.concatenate([np.arange(n_maps)] * 2), np.concatenate([maps.team_a.map(tidx).to_numpy(), maps.team_b.map(tidx).to_numpy()]))),
                            shape=(n_maps, n_t))
    is_faceit = (maps.source.to_numpy() == "faceit")
    faceit_rows = sp.diags(is_faceit.astype(float))
    hltv_rows = sp.diags((~is_faceit).astype(float))
    x_covs = x_players @ x_cov
    x = sp.hstack([x_players, faceit_rows @ x_players, x_teams, faceit_rows @ x_teams,
                   sp.csr_matrix(hltv_rows @ (x_covs * cov_mask)), sp.csr_matrix(faceit_rows @ x_covs)]).tocsr()

    age = (cutoff - maps.day).dt.days.to_numpy(float)
    w = 0.5 ** (age / cfg.half_life_days) * np.where(is_faceit, cfg.faceit_weight, 1.0) * cfg.kappa
    n = maps.rounds.to_numpy(float)
    k = maps.rounds_a.to_numpy(float)
    lam = np.concatenate([np.full(n_p, cfg.lam_player), np.full(n_p, cfg.lam_delta), np.full(n_t, cfg.lam_team),
                          np.full(n_t, cfg.lam_team), np.full(2 * n_c, cfg.lam_cov)])
    xt = x.T.tocsr()

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        log_p = -np.logaddexp(0.0, -eta)
        log_q = -np.logaddexp(0.0, eta)
        p = np.exp(log_p)
        value = -(w * (k * log_p + (n - k) * log_q)).sum() + 0.5 * (lam * beta * beta).sum()
        grad = -(xt @ (w * (k - n * p))) + lam * beta
        return value, grad

    res = minimize(objective, np.zeros(2 * n_p + 2 * n_t + 2 * n_c), jac=True, method="L-BFGS-B",
                   options={"maxiter": 4000, "maxfun": 8000, "gtol": 1e-5, "ftol": 1e-14})
    beta = res.x
    map_table = regulation_map_win(_GRID[:, None] + cfg.sigma_map * _GH_X[None, :]) @ _GH_W
    info = {"maps": n_maps, "players": n_p, "teams": n_t, "iterations": int(res.nit), "converged": bool(res.success),
            "faceit_maps": int((maps.source == "faceit").sum()), "seconds": round(time.time() - t0, 1),
            "gamma": [round(float(g), 4) for g in beta[2 * n_p + 2 * n_t:2 * n_p + 2 * n_t + n_c]],
            "gamma_faceit": [round(float(g), 4) for g in beta[2 * n_p + 2 * n_t + n_c:]]}
    return Fit(cfg=cfg, obs=obs, cutoff=cutoff, players=pidx, theta=beta[:n_p], teams=tidx, team_effect=beta[2 * n_p:2 * n_p + n_t],
               gamma=beta[2 * n_p + 2 * n_t:2 * n_p + 2 * n_t + n_c], mu=mu, sd=sd, player_cov=x_cov * cov_mask, cov_mask=cov_mask, map_table=map_table, info=info)


def lineup_skill(fit: Fit, team: str, lineup: list[str], day: pd.Timestamp) -> float:
    """Mean skill of a five; players without maps in the window are priced from the prior of the team they play for."""
    if not lineup:
        lineup = [""] * LINEUP
    values = []
    for p in lineup:
        i = fit.players.get(p)
        if i is not None:
            values.append(fit.theta[i] + fit.player_cov[i] @ fit.gamma)
        else:
            x, _, _ = covariates(fit.obs, [p], [team], [day], fit.mu, fit.sd)
            values.append((x[0] * fit.cov_mask) @ fit.gamma)
    return float(np.mean(values))


def price(fit: Fit, team_a: str, lineup_a: list[str], team_b: str, lineup_b: list[str], day: pd.Timestamp) -> float:
    """P(team A wins the bo3), with map noise integrated per map and series noise shared by the three maps."""
    cfg = fit.cfg
    d = (lineup_skill(fit, team_a, lineup_a, day) - lineup_skill(fit, team_b, lineup_b, day)
         + _team(fit, team_a) - _team(fit, team_b))
    p = np.interp(d + cfg.sigma_series * _GH_X, _GRID, fit.map_table)
    return float((p * p * (3 - 2 * p)) @ _GH_W)


def _team(fit: Fit, team: str) -> float:
    i = fit.teams.get(team)
    return float(fit.team_effect[i]) if i is not None else 0.0


def team_key(team_id) -> str:
    return f"h{int(team_id)}"
