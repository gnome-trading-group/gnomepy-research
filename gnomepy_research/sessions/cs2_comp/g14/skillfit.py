"""
g14's batch skill fit at one cutoff: penalised maximum likelihood over a trailing window of maps, with skills that
move through time.

    rounds_A ~ Binomial(n, sigmoid(d)),  d = mean(s[five_A, k]) + o[team_A, k] - mean(s[five_B, k]) - o[team_B, k]

where k is the map's period (the window is cut into `period_days` slices ending at the cutoff). Each player cell
s[p, k] sits around a prior mean x_p . gamma (HLTV rank strength and FACEIT division shares, learned with the
skills) and adjacent cells of one player are tied by a random walk, s[p, k+1] - s[p, k] ~ N(0, tau^2): a player's
form can drift, and the latest cell, which prices the next series, leans on the recent maps more than the old ones
without throwing the old ones away. Team offsets o[t, k] carry what a five does beyond its players (system,
cohesion) and drift the same way. HLTV map ratings relative to teammates are a second measurement of who in a five
carries it, as in e12.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g14.data import DIVISIONS, FIVE, LOG_UNRANKED, SOURCE_FACEIT, SOURCE_HLTV, Inputs

PRIOR_DIVISIONS = DIVISIONS[:-1]


@dataclass(frozen=True)
class FitConfig:
    window_days: int = 365
    period_days: int = 91
    half_life_days: float = 600.0
    sigma: float = 0.5            # prior sd of a cell around its prior mean
    tau: float = 0.12             # random-walk sd between a player's adjacent periods
    sigma_team: float = 0.15      # prior sd of a team offset
    tau_team: float = 0.08
    faceit_weight: float = 0.5
    faceit_scale: float = 1.0
    rating_gamma: float = 0.6
    rating_sd: float = 0.31
    team_offsets: bool = True
    maxiter: int = 200
    method: str = "trust-krylov"


@dataclass
class SkillFit:
    cutoff: pd.Timestamp
    cfg: FitConfig
    players: np.ndarray
    skill: np.ndarray             # (P, K)
    maps_played: np.ndarray
    teams: np.ndarray
    offset: np.ndarray            # (T, K)
    gamma: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def periods(self) -> int:
        return self.skill.shape[1]

    def player(self, uid: int) -> tuple[float, int] | None:
        i = np.searchsorted(self.players, uid)
        if i < len(self.players) and self.players[i] == uid:
            return float(self.skill[i, -1]), int(self.maps_played[i])
        return None

    def unknown_prior(self, team_rank: float) -> float:
        return float(self.gamma[0] + self.gamma[1] + self.gamma[2] * (LOG_UNRANKED - np.log(max(team_rank, 1.0))))

    def lineup(self, five: tuple[int, ...], team_rank: float) -> tuple[float, int, float]:
        """Mean latest-period skill of a five, how many of its players the window barely knows, and the spread."""
        vals, thin = [], 0
        for p in five:
            known = self.player(p)
            if known is None:
                vals.append(self.unknown_prior(team_rank))
                thin += 1
            else:
                vals.append(known[0])
                thin += int(known[1] < 5)
        return float(np.mean(vals)), thin, float(np.std(vals))

    def team(self, tid: int) -> float:
        i = np.searchsorted(self.teams, tid)
        if i < len(self.teams) and self.teams[i] == tid:
            return float(self.offset[i, -1])
        return 0.0


def _prior_design(frame: pd.DataFrame, five_a: np.ndarray, five_b: np.ndarray, players: np.ndarray) -> np.ndarray:
    """Per player: [1, played on HLTV, mean HLTV rank strength, FACEIT division shares...]."""
    pl = np.concatenate([five_a.ravel(), five_b.ravel()])
    pos = np.searchsorted(players, pl)
    strength = np.concatenate([np.repeat(LOG_UNRANKED - np.log(frame.rank_a.to_numpy(float)), FIVE),
                               np.repeat(LOG_UNRANKED - np.log(frame.rank_b.to_numpy(float)), FIVE)])
    is_hltv = np.tile(np.repeat((frame.source == SOURCE_HLTV).to_numpy(), FIVE), 2)
    P = len(players)
    hltv_maps = np.bincount(pos[is_hltv], minlength=P)
    hltv_strength = np.bincount(pos[is_hltv], weights=strength[is_hltv], minlength=P)
    cols = [np.ones(P), (hltv_maps > 0).astype(float), np.divide(hltv_strength, np.maximum(hltv_maps, 1))]
    if (frame.source == SOURCE_FACEIT).any():
        div = np.tile(np.repeat(frame.division.to_numpy(), FIVE), 2)
        fmaps = np.bincount(pos[~is_hltv], minlength=P)
        for d in PRIOR_DIVISIONS:
            m = div == d
            cols.append(np.bincount(pos[m], minlength=P) / np.maximum(fmaps, 1))
    return np.column_stack(cols)


def _rating_rows(rating: np.ndarray, cells: np.ndarray, w: np.ndarray, N: int, gamma: float) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """One row per rated player-map of a fully rated five: L @ theta is the player's cell minus the five's mean cell,
    the target is the rating minus the five's mean rating, over rating_gamma."""
    ok = ~np.isnan(rating).any(axis=1)
    rt, ii, ww = rating[ok], cells[ok], w[ok]
    R = len(rt)
    if R == 0:
        return sp.csr_matrix((0, N)), np.zeros(0), np.zeros(0)
    row = np.arange(R * FIVE)
    rows_mean = np.repeat(row, FIVE)
    cols_mean = np.repeat(ii, FIVE, axis=0).ravel()
    L = (sp.csr_matrix((np.ones(R * FIVE), (row, ii.ravel())), shape=(R * FIVE, N))
         - sp.csr_matrix((np.full(R * FIVE * FIVE, 1.0 / FIVE), (rows_mean, cols_mean)), shape=(R * FIVE, N)))
    return L.tocsr(), ((rt - rt.mean(axis=1, keepdims=True)) / gamma).ravel(), np.repeat(ww, FIVE)


def _differences(n_units: int, K: int, offset: int, N: int) -> sp.csr_matrix:
    """Rows of (cell k+1 - cell k) for every unit, over the flattened (unit, period) cells starting at `offset`."""
    if K < 2:
        return sp.csr_matrix((0, N))
    u = np.repeat(np.arange(n_units), K - 1)
    k = np.tile(np.arange(K - 1), n_units)
    rows = np.arange(len(u))
    hi = offset + u * K + k + 1
    lo = offset + u * K + k
    return (sp.csr_matrix((np.ones(len(u)), (rows, hi)), shape=(len(u), N))
            - sp.csr_matrix((np.ones(len(u)), (rows, lo)), shape=(len(u), N))).tocsr()


def fit(inputs: Inputs, cutoff: pd.Timestamp, cfg: FitConfig) -> SkillFit:
    m = inputs.maps
    keep = ((m.known < cutoff) & (m.time >= cutoff - pd.Timedelta(days=cfg.window_days))).to_numpy()
    f = m[keep].reset_index(drop=True)
    five_a, five_b = inputs.five_a[keep], inputs.five_b[keep]
    rating_a, rating_b = inputs.rating_a[keep], inputs.rating_b[keep]
    K = int(np.ceil(cfg.window_days / cfg.period_days))
    age_days = (cutoff - f.time).dt.total_seconds().to_numpy() / 86400.0
    period = np.clip(K - 1 - np.floor(age_days / cfg.period_days).astype(int), 0, K - 1)
    players = np.unique(np.concatenate([five_a.ravel(), five_b.ravel()]))
    teams = np.unique(np.concatenate([f.team_a.to_numpy(), f.team_b.to_numpy()])) if cfg.team_offsets else np.zeros(0, np.int64)
    P, T, M = len(players), len(teams), len(f)
    X = _prior_design(f, five_a, five_b, players)
    G = X.shape[1]
    N = P * K + T * K + G
    off_o, off_g = P * K, P * K + T * K
    cell_a = np.searchsorted(players, five_a) * K + period[:, None]
    cell_b = np.searchsorted(players, five_b) * K + period[:, None]
    rows = np.repeat(np.arange(M), FIVE)
    A = (sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, cell_a.ravel())), shape=(M, N))
         - sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, cell_b.ravel())), shape=(M, N)))
    if T:
        ta = off_o + np.searchsorted(teams, f.team_a.to_numpy()) * K + period
        tb = off_o + np.searchsorted(teams, f.team_b.to_numpy()) * K + period
        A = A + sp.csr_matrix((np.ones(M), (np.arange(M), ta)), shape=(M, N)) - sp.csr_matrix((np.ones(M), (np.arange(M), tb)), shape=(M, N))
    scale = np.where(f.source == SOURCE_FACEIT, cfg.faceit_scale, 1.0)
    A = (sp.diags(scale) @ A).tocsr()
    At = A.T.tocsr()
    w = 0.5 ** (age_days / cfg.half_life_days) * np.where(f.source == SOURCE_FACEIT, cfg.faceit_weight, 1.0)
    n = f.rounds.to_numpy(float)
    ra = f.rounds_a.to_numpy(float)
    # penalties: cells around their prior mean, random walks, team offsets, rating rows, and a touch on gamma
    E = sp.csr_matrix((np.ones(P * K), (np.arange(P * K), np.repeat(np.arange(P), K))), shape=(P * K, P))
    B = sp.hstack([sp.identity(P * K, format="csr"), sp.csr_matrix((P * K, T * K)), -(E @ sp.csr_matrix(X))]).tocsr()
    pen = (B.T @ B) / cfg.sigma ** 2
    Ds = _differences(P, K, 0, N)
    pen = pen + (Ds.T @ Ds) / cfg.tau ** 2
    if T:
        Io = sp.csr_matrix((np.ones(T * K), (np.arange(T * K), off_o + np.arange(T * K))), shape=(T * K, N))
        pen = pen + (Io.T @ Io) / cfg.sigma_team ** 2
        Do = _differences(T, K, off_o, N)
        pen = pen + (Do.T @ Do) / cfg.tau_team ** 2
    Ig = sp.csr_matrix((np.ones(G), (np.arange(G), off_g + np.arange(G))), shape=(G, N))
    pen = pen + 1e-2 * (Ig.T @ Ig)
    lin = np.zeros(N)
    if cfg.rating_gamma:
        L, yr, wr = _rating_rows(np.concatenate([rating_a, rating_b]), np.concatenate([cell_a, cell_b]),
                                 np.concatenate([w, w]), N, cfg.rating_gamma)
        kappa = (cfg.rating_gamma / cfg.rating_sd) ** 2
        if L.shape[0]:
            Lw = L.T @ sp.diags(wr)
            pen = pen + kappa * (Lw @ L)
            lin = -kappa * (Lw @ yr)
    pen = pen.tocsr()

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        d = A @ theta
        nll = -(w * (ra * d - n * np.logaddexp(0.0, d))).sum()
        pt = pen @ theta
        obj = nll + 0.5 * (theta @ pt) + lin @ theta
        grad = -(At @ (w * (ra - n * expit(d)))) + pt + lin
        return obj, grad

    def hessp(theta: np.ndarray, v: np.ndarray) -> np.ndarray:
        pr = expit(A @ theta)
        return At @ (w * n * pr * (1 - pr) * (A @ v)) + pen @ v

    res = minimize(objective, np.zeros(N), jac=True, hessp=hessp, method=cfg.method,
                   options={"maxiter": cfg.maxiter, "gtol": 1e-6})
    played = np.bincount(np.concatenate([np.searchsorted(players, five_a).ravel(), np.searchsorted(players, five_b).ravel()]), minlength=P)
    theta = res.x
    return SkillFit(cutoff, cfg, players, theta[:P * K].reshape(P, K), played, teams,
                    theta[off_o:off_g].reshape(T, K) if T else np.zeros((0, K)), theta[off_g:],
                    {"maps": M, "players": P, "teams": T, "periods": K, "iters": int(res.nit), "converged": bool(res.success),
                     "message": str(res.message), "gnorm": float(np.abs(res.jac).max())})
