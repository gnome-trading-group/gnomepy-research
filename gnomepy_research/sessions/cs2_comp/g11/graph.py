"""
g11's strength model: round-level player skills over the match graph, with community-level effects.

    rounds_A ~ Binomial(n, sigmoid(mean(s_A) - mean(s_B))),   each map weighted by 0.5^(age / half_life) * w_source
    s_p = x_p . gamma + c_k(p) + u_p,   c_k ~ N(0, tau^2),   u_p ~ N(0, sigma^2)

k(p) is the player's community in the recency-weighted map graph (spectral clustering of the player graph). Scenes
(regions, tiers, online leagues) play each other rarely, so an iid player prior pins every scene's mean to the same
prior mean, and the fit is wrong in a structured way whenever it compares across scenes. With c_k the players of a
scene move together, and a scene's level is set by the cross-scene maps there are.

x_p is the prior mean design: the HLTV rank of the teams the player played for and the player's latest FACEIT skill
level. HLTV map ratings relative to teammates are a further measurement of who carries a five (as in e12).

A series is priced from the gap in mean skill of the two announced fives, shifted by each team's map offsets averaged
over the active map pool. Optionally a Laplace approximation gives the posterior variance of each gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh, splu
from scipy.special import expit
from sklearn.cluster import KMeans

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.g11.e12.data import FIVE, UNRANKED, Inputs
from gnomepy_research.sessions.cs2_comp.g11.e12.model import _prior_design, _rating_rows

@dataclass(frozen=True)
class Config:
    window_days: int = 365
    half_life_days: float = 300.0
    sigma: float = 0.5
    tau: float = 0.3
    # One set of community effects per resolution (number of spectral communities).
    communities: tuple = (32,)
    faceit_weight: float = 0.5
    rating_gamma: float = 0.6
    rating_sd: float = 0.31
    # Prior sd of a team's map-specific round-logit offset, and the window that sets the active map pool.
    map_sd: float = 0.1
    pool_days: int = 60
    # The Laplace factorisation is only needed for gap variances.
    laplace: bool = False
    # With FACEIT on, False keeps FACEIT to the skill prior (levels) and fits on HLTV maps only.
    faceit_maps: bool = False
    community_seed: int = 0
    # Price a player the fit never saw with his known teammates' community effects.
    unseen_community: bool = True


def player_graph(ia: np.ndarray, ib: np.ndarray, w: np.ndarray, P: int) -> sp.csr_matrix:
    """The player graph: teammates and opponents on a map are linked, by the map's weight."""
    ten = np.concatenate([ia, ib], axis=1)
    r = np.repeat(ten, 10, axis=1).ravel()
    c = np.tile(ten, (1, 10)).ravel()
    W = sp.csr_matrix((np.repeat(w, 100), (r, c)), shape=(P, P))
    W.setdiag(0)
    W.eliminate_zeros()
    return W


def normalized_adjacency(W: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    return (sp.diags(dinv) @ W @ sp.diags(dinv)).tocsr(), dinv


def communities(W: sp.csr_matrix, k: int, seed: int = 0) -> np.ndarray:
    """Spectral communities of the player graph."""
    P = W.shape[0]
    if k <= 1 or P < 10 * k:
        return np.zeros(P, dtype=int)
    N, dinv = normalized_adjacency(W)
    vals, vecs = eigsh(N, k=k, which="LA", v0=np.full(P, 1.0 / np.sqrt(P)))
    emb = vecs * dinv[:, None]
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(emb)


@dataclass
class GraphFit:
    cutoff: pd.Timestamp
    players: np.ndarray
    skill: np.ndarray
    maps_played: np.ndarray
    community: np.ndarray
    gamma: np.ndarray
    c: np.ndarray
    sigma: float
    tau: float
    lu: object
    n_theta: int
    X: np.ndarray
    map_offset: dict = field(default_factory=dict)
    pool: tuple = ()
    levels: dict = field(default_factory=dict)
    resolutions: int = 1
    membership: sp.csr_matrix | None = None
    unseen_community: bool = False
    info: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, inputs: Inputs, cutoff: pd.Timestamp, cfg: Config, faceit: bool) -> GraphFit:
        maps = inputs.maps(cutoff, cutoff - pd.Timedelta(days=cfg.window_days), faceit and cfg.faceit_maps)
        f = maps.frame
        players = np.unique(np.concatenate([maps.five_a.ravel(), maps.five_b.ravel()]))
        P, M = len(players), len(f)
        ia = np.searchsorted(players, maps.five_a)
        ib = np.searchsorted(players, maps.five_b)
        rows = np.repeat(np.arange(M), FIVE)
        A = (sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, ia.ravel())), shape=(M, P))
             - sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, ib.ravel())), shape=(M, P))).tocsr()
        At = A.T.tocsr()
        age = (cutoff - f.time).dt.total_seconds().to_numpy() / 86400.0
        w = 0.5 ** (age / cfg.half_life_days) * np.where(f.source == "faceit", cfg.faceit_weight, 1.0)
        n = f.rounds.to_numpy(float)
        ra = f.rounds_a.to_numpy(float)
        X0 = _prior_design(maps, players, inputs, cutoff, faceit)
        # Division shares are all zero without FACEIT maps; an all-zero column only slows the solver.
        X0 = X0[:, np.r_[np.ones(min(5, X0.shape[1]), bool), X0[:, 5:].any(axis=0)]]
        K = X0.shape[1]
        Xs = sp.csr_matrix(X0)
        W = player_graph(ia, ib, w, P)
        levels = [communities(W, k, cfg.community_seed) for k in cfg.communities]
        comm = levels[0]
        C = sp.hstack([sp.csr_matrix((np.ones(P), (np.arange(P), lv)), shape=(P, int(lv.max()) + 1)) for lv in levels]).tocsr()
        NC = C.shape[1]
        inv_tau = 1.0 / cfg.tau ** 2
        prec = np.full(P, 1.0 / cfg.sigma ** 2)
        L, yr, wr = _rating_rows(f, ia, ib, w, P, cfg)
        kappa = (cfg.rating_gamma / cfg.rating_sd) ** 2

        def split(theta):
            return theta[:P], theta[P:P + K], theta[P + K:]

        def objective(theta):
            s, g, c = split(theta)
            d = A @ s
            r = s - Xs @ g - C @ c
            pr_ = prec * r
            obj = -(w * (ra * d - n * np.logaddexp(0.0, d))).sum() + 0.5 * (r @ pr_) + 0.5e-2 * (g @ g) \
                + 0.5 * inv_tau * (c @ c)
            grad_s = -(At @ (w * (ra - n * expit(d)))) + pr_
            if kappa:
                e = yr - L @ s
                obj += 0.5 * kappa * (wr * e * e).sum()
                grad_s -= kappa * (L.T @ (wr * e))
            grad_g = -(Xs.T @ pr_) + 1e-2 * g
            grad_c = -(C.T @ pr_) + inv_tau * c
            return obj, np.concatenate([grad_s, grad_g, grad_c])

        def hessp(theta, v):
            pr = expit(A @ theta[:P])
            vs, vg, vc = split(v)
            pr_ = prec * (vs - Xs @ vg - C @ vc)
            hs = At @ (w * n * pr * (1 - pr) * (A @ vs)) + pr_
            if kappa:
                hs += kappa * (L.T @ (wr * (L @ vs)))
            hg = -(Xs.T @ pr_) + 1e-2 * vg
            hc = -(C.T @ pr_) + inv_tau * vc
            return np.concatenate([hs, hg, hc])

        res = minimize(objective, np.zeros(P + K + NC), jac=True, hessp=hessp, method="trust-krylov",
                       options={"maxiter": 1000, "gtol": 1e-6})
        s, g, c = split(res.x)
        pr = expit(A @ s)
        Pm = sp.diags(prec)
        n_theta = P + K + NC
        Hss = At @ sp.diags(w * n * pr * (1 - pr)) @ A + Pm
        if kappa:
            Hss = Hss + kappa * (L.T @ sp.diags(wr) @ L)
        H = sp.bmat([[Hss, -Pm @ Xs, -Pm @ C],
                     [-(Pm @ Xs).T, Xs.T @ Pm @ Xs + 1e-2 * sp.eye(K), Xs.T @ Pm @ C],
                     [-(Pm @ C).T, C.T @ Pm @ Xs, C.T @ Pm @ C + inv_tau * sp.eye(NC)]]).tocsc()
        lu = splu(H) if cfg.laplace else None
        played = np.bincount(np.concatenate([ia.ravel(), ib.ravel()]), minlength=P)
        offsets, pool = map_offsets(f, w, ra, n, pr, cutoff, cfg)
        levels = {}
        if faceit:
            lv = inputs.levels[inputs.levels.known < cutoff]
            levels = {int(u): float(x) for u, x in lv.groupby("uid")["level"].last().items()}
        return cls(cutoff, players, s, played, comm, g, c, cfg.sigma, cfg.tau, lu, n_theta, X0, offsets, pool, levels,
                   len(cfg.communities), C, cfg.unseen_community,
                   {"maps": M, "players": P, "iters": int(res.nit), "gnorm": float(np.abs(res.jac).max())})

    def _vec(self, five: tuple[int, ...] | None, rank: float, sign: float, v: np.ndarray) -> tuple[float, float, int, list]:
        """Adds sign/5 per known player of the five into v; returns the five's mean skill, the variance it adds for
        players the fit never saw, how many it barely knows, and the communities of its known players."""
        P, K = len(self.players), len(self.gamma)
        g_start = self.n_theta - K - len(self.c)
        base = np.zeros(K)
        base[:3] = [1.0, 1.0, np.log(UNRANKED) - np.log(rank)]
        unknown_var = self.sigma ** 2 + self.tau ** 2 * self.resolutions
        if not five:
            v[g_start:g_start + K] += sign * base
            return float(self.gamma @ base), unknown_var, FIVE, []
        vals, extra, thin, comms = [], 0.0, 0, []
        known_idx = [i for i in np.searchsorted(self.players, five) if i < P and self.players[i] in five]
        team_comm = None
        if self.unseen_community and known_idx and self.membership is not None:
            # A player the fit never saw joins his known teammates' scenes.
            team_comm = np.asarray(self.membership[known_idx].mean(axis=0)).ravel()
        for p in five:
            i = np.searchsorted(self.players, p)
            if i < P and self.players[i] == p:
                vals.append(self.skill[i])
                v[i] += sign / FIVE
                thin += int(self.maps_played[i] < 5)
                comms.append(int(self.community[i]))
            else:
                # The prior mean of a player the fit never saw moves with the fitted gamma, so it goes in v too.
                xu = base.copy()
                level = self.levels.get(int(p))
                if level is not None and K > 4:
                    xu[3:5] = [1.0, level - 8.0]
                prior = float(self.gamma @ xu)
                if team_comm is not None:
                    prior += float(team_comm @ self.c)
                    v[self.n_theta - len(self.c):] += sign * team_comm / FIVE
                vals.append(prior)
                v[g_start:g_start + K] += sign * xu / FIVE
                extra += unknown_var / FIVE ** 2
                thin += 1
        return float(np.mean(vals)), extra, thin, comms

    def gap(self, five_a, rank_a, five_b, rank_b, team_a: int = -1, team_b: int = -1) -> dict:
        v = np.zeros(self.n_theta)
        ma, ea, ta, ca = self._vec(five_a, rank_a, 1.0, v)
        mb, eb, tb, cb = self._vec(five_b, rank_b, -1.0, v)
        var = float(v @ self.lu.solve(v)) + ea + eb if self.lu is not None else float("nan")
        dom_a = max(set(ca), key=ca.count) if ca else -1
        dom_b = max(set(cb), key=cb.count) if cb else -1
        known = sum(1 for m in self.pool for t in (team_a, team_b) if (t, m) in self.map_offset)
        return {"gap": ma - mb, "q": pool_price(ma - mb, team_a, team_b, self.map_offset, self.pool), "gap_var": var,
                "pool_known": known, "thin_a": ta, "thin_b": tb, "comm_a": dom_a, "comm_b": dom_b,
                "cross": int(dom_a != dom_b or dom_a < 0)}


def map_offsets(f: pd.DataFrame, w: np.ndarray, ra: np.ndarray, n: np.ndarray, pr: np.ndarray, cutoff: pd.Timestamp,
                cfg: Config) -> tuple[dict, tuple]:
    """Each team's round-logit offset on each map beyond its five's skill: one Newton step from zero on the map's
    residual rounds, shrunk by a N(0, map_sd^2) prior. Also the active pool: the seven most played maps lately."""
    team = np.concatenate([f.team_a.to_numpy(), f.team_b.to_numpy()])
    mp = np.concatenate([f.map_name.to_numpy(), f.map_name.to_numpy()])
    res = w * (ra - n * pr)
    info = w * n * pr * (1 - pr)
    d = pd.DataFrame({"team": team, "map": mp, "res": np.concatenate([res, -res]), "fisher": np.concatenate([info, info])})
    d = d[d.team >= 0]
    g = d.groupby(["team", "map"])[["res", "fisher"]].sum()
    b = g.res / (g.fisher + 1.0 / cfg.map_sd ** 2)
    offsets = {(int(t), m): float(v) for (t, m), v in b.items()}
    recent = f[(f.time >= cutoff - pd.Timedelta(days=cfg.pool_days)) & (f.source == "hltv")]
    counts = recent.map_name.value_counts()
    pool = tuple(sorted(counts.index[:7]))
    return offsets, pool


def pool_price(gap: float, team_a: int, team_b: int, offsets: dict, pool: tuple) -> float:
    """P(A wins a bo3) with the gap shifted by the pool-average map offset of each side. The veto is never known
    before a series, and simulating one from the offsets priced no better than this average."""
    if not pool:
        return float(series_price(np.array([gap]))[0])
    shift = np.mean([offsets.get((team_a, m), 0.0) - offsets.get((team_b, m), 0.0) for m in pool])
    return float(series_price(np.array([gap + shift]))[0])


def series_price(gap: np.ndarray) -> np.ndarray:
    p = regulation_map_win(np.asarray(gap, float))
    return p * p * (3 - 2 * p)
