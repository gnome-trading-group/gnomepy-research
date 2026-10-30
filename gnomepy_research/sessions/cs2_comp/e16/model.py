"""
A penalised paired-comparison model of player skill, fitted by maximum a posteriori on one window, HLTV and FACEIT
maps together.

    rounds_A ~ Binomial(n, sigmoid(D)) with weight kappa[source] * 0.5 ** (age / half_life)
    D = mean(theta[five A]) - mean(theta[five B]) + team[A] - team[B]
    theta[p] ~ N(x_p . beta, sigma0^2)      x_p: team ranking, FACEIT skill level, ESEA division, missingness flags
    team[t] ~ N(0, sigma_team^2)
    impact_dev[p, map] ~ N(gamma[source] * (theta[p] - mean(theta[own five])), sigma_r[source]^2)

kappa is each source's overdispersion: the rounds of a map are not independent trials, and an ESEA map says less
about a player's HLTV-level strength than an HLTV map does. The impact channel is the HLTV-rating idea applied per
source: the rounds only identify a five's mean, a player's impact relative to his teammates says who in it is good,
which is what matters when a five changes. The prior is the preseason view (where a player plays and at what level)
and the decayed results are the in-season evidence; the fit blends them by their precision.

A series is priced from the MAP skills with the parameter uncertainty of each five (a diagonal Laplace
approximation) added to the noise shared by all its maps.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e16.data import DIVISIONS, LINEUP

UNRANKED = 300.0
RANK_MAX_AGE_DAYS = 120
_X, _W = np.polynomial.hermite_e.hermegauss(21)
_W = _W / _W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)
FEATURES = ["one", "rank", "has_hltv", "level", "no_level"] + [f"div_{d}" for d in DIVISIONS]


@dataclass(frozen=True)
class Config:
    window_days: int = 365
    half_life_days: float = 180.0
    kappa_hltv: float = 1.0
    kappa_faceit: float = 0.25
    sigma0: float = 1.2
    sigma_team: float = 0.05
    sigma_beta: float = 1.0
    impact_hltv: float = 1.0
    impact_faceit: float = 1.0
    map_noise: float = 0.37
    series_noise: float = 0.17
    faceit_window_days: int = 365
    roster_moves: int = 1


@dataclass
class Fit:
    cfg: Config
    players: pd.Index
    teams: pd.Index
    theta: np.ndarray
    var: np.ndarray
    team: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    rank_of: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def player(self, key: str, team_rank: float) -> tuple[float, float]:
        i = self.players.get_indexer([key])[0]
        if i >= 0:
            return float(self.theta[i]), float(self.var[i])
        x = np.zeros(len(FEATURES))
        x[0], x[1], x[4] = 1.0, team_rank, 1.0
        return float(x @ self.beta), self.cfg.sigma0 ** 2

    def gap(self, team_a: str, five_a, rank_a: float, team_b: str, five_b, rank_b: float) -> tuple[float, float]:
        mean, var = 0.0, 0.0
        for five, rank, sign in ((five_a, rank_a, 1.0), (five_b, rank_b, -1.0)):
            if five is None:
                m, v = self.player("", rank)
                mean += sign * m
                var += v / LINEUP
            else:
                for p in five:
                    m, v = self.player(p, rank)
                    mean += sign * m / LINEUP
                    var += v / LINEUP ** 2
        for t, sign in ((team_a, 1.0), (team_b, -1.0)):
            i = self.teams.get_indexer([t])[0]
            mean += sign * (self.team[i] if i >= 0 else 0.0)
        return mean, var

    def price(self, mean: float, var: float) -> float:
        d = mean + np.sqrt(var + self.cfg.series_noise ** 2) * _X
        p = np.interp(d, _GRID, _map_table(self.cfg.map_noise))
        return float((p * p * (3 - 2 * p)) @ _W)


@functools.lru_cache(maxsize=4)
def _map_table(noise: float) -> np.ndarray:
    """P(win a map) on a grid of round-logit gaps, with the per-map noise integrated out."""
    return regulation_map_win(_GRID[:, None] + noise * _X[None, :]) @ _W


def rank_feature(rankings: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    """log(300 / rank) of each HLTV team's latest ranking before the cutoff; 0 when unranked or stale."""
    r = rankings[(rankings.date < cutoff) & (rankings.date >= cutoff - pd.Timedelta(days=RANK_MAX_AGE_DAYS))]
    last = r.sort_values("date", kind="stable").groupby("team_id").tail(1)
    return {f"h{int(t)}": float(np.log(UNRANKED / min(k, UNRANKED))) for t, k in zip(last.team_id, last["rank"])}


def player_features(players: pd.Index, maps: pd.DataFrame, levels: pd.DataFrame | None,
                    rank_of: dict[str, float]) -> np.ndarray:
    """Prior features of each player from the window's maps (latest team, latest division) and FACEIT levels."""
    rows = pd.concat([maps[["cut", "team_a", "lineup_a", "source", "division"]].set_axis(["cut", "team", "five", "source", "division"], axis=1),
                      maps[["cut", "team_b", "lineup_b", "source", "division"]].set_axis(["cut", "team", "five", "source", "division"], axis=1)])
    rows = rows.explode("five").rename(columns={"five": "player"}).sort_values("cut", kind="stable")
    X = np.zeros((len(players), len(FEATURES)))
    X[:, 0] = 1.0
    h = rows[rows.source == "hltv"].groupby("player").tail(1)
    hi = players.get_indexer(h.player)
    X[hi, 1] = h.team.map(rank_of).fillna(0.0).to_numpy()
    X[hi, 2] = 1.0
    f = rows[(rows.source == "faceit") & (rows.division > 0)].groupby("player").tail(1)
    fi = players.get_indexer(f.player)
    X[fi, 5 + f.division.to_numpy() - 1] = 1.0
    X[:, 4] = 1.0
    if levels is not None and len(levels):
        lv = levels.sort_values("cut", kind="stable").groupby("player").tail(1)
        li = players.get_indexer(lv.player)
        ok = li >= 0
        X[li[ok], 3] = lv.skill_level.to_numpy(float)[ok] - 10.0
        X[li[ok], 4] = 0.0
    return X


def fit(maps: pd.DataFrame, ratings: pd.DataFrame, levels: pd.DataFrame | None, rank_of: dict[str, float],
        cutoff: pd.Timestamp, cfg: Config) -> Fit:
    maps = maps.sort_values(["cut", "match_id", "map_name"], kind="stable").reset_index(drop=True)
    la, lb = np.array(maps.lineup_a.tolist(), dtype=object), np.array(maps.lineup_b.tolist(), dtype=object)
    codes, players = pd.factorize(np.concatenate([la.ravel(), lb.ravel()]))
    players = pd.Index(players)
    n_p, n_m = len(players), len(maps)
    ca, cb = codes[: n_m * LINEUP].reshape(n_m, LINEUP), codes[n_m * LINEUP:].reshape(n_m, LINEUP)
    row = np.repeat(np.arange(n_m), LINEUP)
    A = sp.csr_matrix((np.concatenate([np.full(n_m * LINEUP, 1 / LINEUP), np.full(n_m * LINEUP, -1 / LINEUP)]),
                       (np.concatenate([row, row]), np.concatenate([ca.ravel(), cb.ravel()]))), shape=(n_m, n_p))
    tcodes, teams = pd.factorize(np.concatenate([maps.team_a.to_numpy(), maps.team_b.to_numpy()]))
    teams = pd.Index(teams)
    n_t = len(teams)
    T = sp.csr_matrix((np.concatenate([np.ones(n_m), -np.ones(n_m)]),
                       (np.concatenate([np.arange(n_m), np.arange(n_m)]), tcodes)), shape=(n_m, n_t))
    age = (cutoff - maps.time).dt.total_seconds().to_numpy() / 86400.0
    decay = 0.5 ** (np.clip(age, 0, None) / cfg.half_life_days)
    kappa = np.where(maps.source == "hltv", cfg.kappa_hltv, cfg.kappa_faceit)
    c = kappa * decay
    n = maps.rounds.to_numpy(float)
    ra = maps.rounds_a.to_numpy(float)

    rat = ratings[ratings.player.isin(players)]
    own = players.get_indexer(np.array(rat.own.tolist(), dtype=object).ravel()).reshape(-1, LINEUP) if len(rat) else np.zeros((0, LINEUP), int)
    keep = (own >= 0).all(axis=1)
    rat, own = rat[keep], own[keep]
    pi = players.get_indexer(rat.player)
    n_r = len(rat)
    rrow = np.arange(n_r)
    R = sp.csr_matrix((np.concatenate([np.ones(n_r), np.full(n_r * LINEUP, -1 / LINEUP)]),
                       (np.concatenate([rrow, np.repeat(rrow, LINEUP)]), np.concatenate([pi, own.ravel()]))),
                      shape=(n_r, n_p))
    src = (rat.source == "faceit").to_numpy().astype(int)
    dev = rat.dev.to_numpy(float)
    sd = np.array([max(dev[src == s].std(), 1e-3) if (src == s).any() else 1.0 for s in (0, 1)])
    rage = (cutoff - rat.time).dt.total_seconds().to_numpy() / 86400.0
    rw = 0.5 ** (np.clip(rage, 0, None) / cfg.half_life_days) * np.where(src == 1, cfg.impact_faceit, cfg.impact_hltv) / sd[src] ** 2

    X = player_features(players, maps, levels, rank_of)
    k = X.shape[1]
    s0, st, sb = cfg.sigma0 ** 2, cfg.sigma_team ** 2, cfg.sigma_beta ** 2

    def unpack(z):
        return z[:n_p], z[n_p:n_p + n_t], z[n_p + n_t:n_p + n_t + k], z[n_p + n_t + k:]

    def objective(z):
        theta, team, beta, gamma = unpack(z)
        d = A @ theta + T @ team
        sig = 1 / (1 + np.exp(-d))
        ll = c * (ra * np.log(np.clip(sig, 1e-12, None)) + (n - ra) * np.log(np.clip(1 - sig, 1e-12, None)))
        g_d = -c * (ra - n * sig)
        rel = R @ theta
        e = dev - gamma[src] * rel
        resid = theta - X @ beta
        loss = -ll.sum() + 0.5 * (rw * e * e).sum() + 0.5 * (resid @ resid) / s0 + 0.5 * (team @ team) / st \
            + 0.5 * (beta[1:] @ beta[1:]) / sb + 0.5 * (gamma @ gamma)
        g_theta = A.T @ g_d - R.T @ (rw * e * gamma[src]) + resid / s0
        g_team = T.T @ g_d + team / st
        g_beta = -X.T @ resid / s0
        g_beta[1:] += beta[1:] / sb
        g_gamma = np.array([-(rw * e * rel)[src == s].sum() for s in (0, 1)]) + gamma
        return loss, np.concatenate([g_theta, g_team, g_beta, g_gamma])

    z0 = np.concatenate([np.zeros(n_p + n_t + k), np.array([0.5, 0.5])])
    res = minimize(objective, z0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "maxcor": 20, "gtol": 1e-6})
    theta, team, beta, gamma = unpack(res.x)
    sig = 1 / (1 + np.exp(-(A @ theta + T @ team)))
    fisher = np.asarray((A.multiply(A)).T @ (c * n * sig * (1 - sig))).ravel()
    fisher += np.asarray((R.multiply(R)).T @ (rw * gamma[src] ** 2)).ravel()
    var = 1 / (fisher + 1 / s0)
    info = {"maps": n_m, "faceit_maps": int((maps.source == "faceit").sum()), "players": n_p, "teams": n_t,
            "rating_rows": n_r, "iters": int(res.nit), "converged": bool(res.success), "gamma": gamma.round(3).tolist(),
            "beta": dict(zip(FEATURES, beta.round(3).tolist()))}
    return Fit(cfg, players, teams, theta, var, team, beta, gamma, rank_of, info)
