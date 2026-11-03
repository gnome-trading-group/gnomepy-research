"""
g11's strength model: round-level player skills over the match graph, with community-level effects and a Laplace
posterior, so a series price carries how well the two fives are actually connected by results.

    rounds_A ~ Binomial(n, sigmoid(mean(s_A) - mean(s_B))),   each map weighted by 0.5^(age / half_life) * w_source
    s_p = x_p . gamma + c_k(p) + u_p,   c_k ~ N(0, tau^2),   u_p ~ N(0, sigma^2)

k(p) is the player's community in the recency-weighted map graph (spectral clustering of the player graph). Regional
scenes play each other rarely, so without c_k an iid player prior pins every scene's mean to the same prior mean and
the fit is overconfident when it compares across scenes. With c_k a scene's level is set by the few cross-scene maps
there are, and the Laplace variance of a gap grows with how weakly the two fives' scenes are linked.

HLTV map ratings relative to teammates are a further measurement of who carries a five (as in e12).
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

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(21)
GH_W = GH_W / GH_W.sum()


@dataclass(frozen=True)
class Config:
    window_days: int = 365
    half_life_days: float = 300.0
    sigma: float = 0.5
    tau: float = 0.3
    communities: int = 8
    faceit_weight: float = 0.5
    rating_gamma: float = 0.6
    rating_sd: float = 0.31
    # Prior sd of a team's map-specific round-logit offset, and the window that sets the active map pool.
    map_sds: tuple = (0.1, 0.2, 0.3)
    pool_days: int = 60


def communities(ia: np.ndarray, ib: np.ndarray, w: np.ndarray, P: int, k: int) -> np.ndarray:
    """Spectral communities of the player graph (teammates and opponents on a map are linked, by map weight)."""
    if k <= 1 or P < 10 * k:
        return np.zeros(P, dtype=int)
    ten = np.concatenate([ia, ib], axis=1)
    r = np.repeat(ten, 10, axis=1).ravel()
    c = np.tile(ten, (1, 10)).ravel()
    ww = np.repeat(w, 100)
    W = sp.csr_matrix((ww, (r, c)), shape=(P, P))
    W.setdiag(0)
    W.eliminate_zeros()
    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    N = sp.diags(dinv) @ W @ sp.diags(dinv)
    vals, vecs = eigsh(N, k=k, which="LA", v0=np.full(P, 1.0 / np.sqrt(P)))
    emb = vecs * dinv[:, None]
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(emb)


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
    lu: object
    X: np.ndarray
    map_offset: dict = field(default_factory=dict)
    pool: tuple = ()
    prefs: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, inputs: Inputs, cutoff: pd.Timestamp, cfg: Config, faceit: bool) -> GraphFit:
        maps = inputs.maps(cutoff, cutoff - pd.Timedelta(days=cfg.window_days), faceit)
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
        X = _prior_design(maps, players, inputs, cutoff, faceit)
        K = X.shape[1]
        comm = communities(ia, ib, w, P, cfg.communities)
        C = sp.csr_matrix((np.ones(P), (np.arange(P), comm)), shape=(P, int(comm.max()) + 1))
        NC = C.shape[1]
        inv_var = 1.0 / cfg.sigma ** 2
        inv_tau = 1.0 / cfg.tau ** 2
        L, yr, wr = _rating_rows(f, ia, ib, w, P, cfg)
        kappa = (cfg.rating_gamma / cfg.rating_sd) ** 2

        def split(theta):
            return theta[:P], theta[P:P + K], theta[P + K:]

        def objective(theta):
            s, g, c = split(theta)
            d = A @ s
            r = s - X @ g - C @ c
            obj = -(w * (ra * d - n * np.logaddexp(0.0, d))).sum() + 0.5 * inv_var * (r @ r) + 0.5e-2 * (g @ g) \
                + 0.5 * inv_tau * (c @ c)
            grad_s = -(At @ (w * (ra - n * expit(d)))) + inv_var * r
            if kappa:
                e = yr - L @ s
                obj += 0.5 * kappa * (wr * e * e).sum()
                grad_s -= kappa * (L.T @ (wr * e))
            grad_g = -inv_var * (X.T @ r) + 1e-2 * g
            grad_c = -inv_var * (C.T @ r) + inv_tau * c
            return obj, np.concatenate([grad_s, grad_g, grad_c])

        def hessp(theta, v):
            pr = expit(A @ theta[:P])
            vs, vg, vc = split(v)
            r = vs - X @ vg - C @ vc
            hs = At @ (w * n * pr * (1 - pr) * (A @ vs)) + inv_var * r
            if kappa:
                hs += kappa * (L.T @ (wr * (L @ vs)))
            hg = -inv_var * (X.T @ r) + 1e-2 * vg
            hc = -inv_var * (C.T @ r) + inv_tau * vc
            return np.concatenate([hs, hg, hc])

        res = minimize(objective, np.zeros(P + K + NC), jac=True, hessp=hessp, method="trust-krylov",
                       options={"maxiter": 200, "gtol": 1e-6})
        s, g, c = split(res.x)
        pr = expit(A @ s)
        Hss = At @ sp.diags(w * n * pr * (1 - pr)) @ A + inv_var * sp.eye(P)
        if kappa:
            Hss = Hss + kappa * (L.T @ sp.diags(wr) @ L)
        Xs = sp.csr_matrix(X)
        H = sp.bmat([[Hss, -inv_var * Xs, -inv_var * C],
                     [-inv_var * Xs.T, inv_var * (Xs.T @ Xs) + 1e-2 * sp.eye(K), inv_var * (Xs.T @ C)],
                     [-inv_var * C.T, inv_var * (C.T @ Xs), inv_var * (C.T @ C) + inv_tau * sp.eye(NC)]]).tocsc()
        lu = splu(H)
        played = np.bincount(np.concatenate([ia.ravel(), ib.ravel()]), minlength=P)
        offsets, pool = map_offsets(f, w, ra, n, pr, cutoff, cfg)
        prefs = map_preferences(f, cutoff, pool)
        return cls(cutoff, players, s, played, comm, g, c, cfg.sigma, lu, X, offsets, pool, prefs,
                   {"maps": M, "players": P, "iters": int(res.nit), "converged": bool(res.success),
                    "c": np.round(c, 3).tolist()})

    def _vec(self, five: tuple[int, ...] | None, rank: float, sign: float, v: np.ndarray) -> tuple[float, float, int, list]:
        """Adds sign/5 per known player of the five into v; returns the five's mean skill, the variance it adds for
        players the fit never saw, how many it barely knows, and the communities of its known players."""
        P, K = len(self.players), len(self.gamma)
        prior_unknown = float(self.gamma[0] + self.gamma[1] * (rank < UNRANKED)
                              + self.gamma[2] * (np.log(UNRANKED) - np.log(rank)))
        if not five:
            return prior_unknown, self.sigma ** 2 + 0.3 ** 2, FIVE, []
        vals, extra, thin, comms = [], 0.0, 0, []
        for p in five:
            i = np.searchsorted(self.players, p)
            if i < P and self.players[i] == p:
                vals.append(self.skill[i])
                v[i] += sign / FIVE
                thin += int(self.maps_played[i] < 5)
                comms.append(int(self.community[i]))
            else:
                vals.append(prior_unknown)
                extra += (self.sigma ** 2 + 0.3 ** 2) / FIVE ** 2
                thin += 1
        return float(np.mean(vals)), extra, thin, comms

    def gap(self, five_a, rank_a, five_b, rank_b, team_a: int = -1, team_b: int = -1) -> dict:
        n = self.lu.shape[0]
        v = np.zeros(n)
        ma, ea, ta, ca = self._vec(five_a, rank_a, 1.0, v)
        mb, eb, tb, cb = self._vec(five_b, rank_b, -1.0, v)
        var = float(v @ self.lu.solve(v)) + ea + eb
        dom_a = max(set(ca), key=ca.count) if ca else -1
        dom_b = max(set(cb), key=cb.count) if cb else -1
        first = next(iter(self.map_offset.values()), {})
        known = sum(1 for m in self.pool if (team_a, m) in first) + sum(1 for m in self.pool if (team_b, m) in first)
        out = {f"q_veto_{int(round(sd * 100))}": veto_price(ma - mb, team_a, team_b, o, self.pool)
               for sd, o in self.map_offset.items()}
        out |= {f"q_pref_{int(round(sd * 100))}": veto_price(ma - mb, team_a, team_b, o, self.pool, self.prefs)
                for sd, o in self.map_offset.items()}
        out |= {f"q_uni_{int(round(sd * 100))}": uniform_price(ma - mb, team_a, team_b, o, self.pool)
                for sd, o in self.map_offset.items()}
        out["pref_n_a"] = self.prefs.get(("n", team_a), 0.0)
        out["pref_n_b"] = self.prefs.get(("n", team_b), 0.0)
        return out | {"gap": ma - mb, "gap_var": var, "pool_known": known, "thin_a": ta, "thin_b": tb, "comm_a": dom_a, "comm_b": dom_b,
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
    offsets = {}
    for sd in cfg.map_sds:
        b = g.res / (g.fisher + 1.0 / sd ** 2)
        offsets[sd] = {(int(t), m): float(v) for (t, m), v in b.items()}
    recent = f[(f.time >= cutoff - pd.Timedelta(days=cfg.pool_days)) & (f.source == "hltv")]
    counts = recent.map_name.value_counts()
    pool = tuple(sorted(counts.index[:7]))
    return offsets, pool


def map_preferences(f: pd.DataFrame, cutoff: pd.Timestamp, pool: tuple, half_life_days: float = 90.0) -> dict:
    """How each team vetoes, from what it has done: ("pick", team, map) is the log share of its own picks on the map
    and ("play", team, map) the log share of its maps played there (a map it never plays is one it bans), both
    recency weighted and smoothed toward an even spread over the pool."""
    h = f[(f.source == "hltv") & (f.time >= cutoff - pd.Timedelta(days=4 * half_life_days)) & f.map_name.isin(pool)]
    w = 0.5 ** ((cutoff - h.time).dt.total_seconds().to_numpy() / 86400.0 / half_life_days)
    pa = h.picked_a.to_numpy(float)
    d = pd.DataFrame({"team": np.concatenate([h.team_a.to_numpy(), h.team_b.to_numpy()]),
                      "map": np.concatenate([h.map_name.to_numpy(), h.map_name.to_numpy()]),
                      "play": np.concatenate([w, w]),
                      "pick": np.concatenate([w * (pa == 1), w * (pa == 0)])})
    g = d.groupby(["team", "map"])[["play", "pick"]].sum()
    tot = g.groupby(level=0).sum()
    k = len(pool)
    out = {}
    for (t, m), r in g.iterrows():
        tt = tot.loc[t]
        out[("play", int(t), m)] = float(np.log((r.play + 0.5) / (tt.play + 0.5 * k)))
        out[("pick", int(t), m)] = float(np.log((r.pick + 0.25) / (tt.pick + 0.25 * k)))
    for t, tt in tot.iterrows():
        out[("n", int(t))] = float(tt.play)
    return out


def uniform_price(gap: float, team_a: int, team_b: int, offsets: dict, pool: tuple) -> float:
    """P(A wins a bo3) with the gap shifted by the pool-average map offset of each side."""
    if not pool:
        return float(series_price(np.array([gap]))[0])
    shift = np.mean([offsets.get((team_a, m), 0.0) - offsets.get((team_b, m), 0.0) for m in pool])
    return float(series_price(np.array([gap + shift]))[0])


def veto_price(gap: float, team_a: int, team_b: int, offsets: dict, pool: tuple, prefs: dict | None = None) -> float:
    """P(A wins a bo3) when both sides veto greedily (ban, ban, pick, pick, ban, ban, decider), averaged over which
    side starts the veto. Without prefs each side vetoes on its true map advantages; with prefs each picks the maps it
    has picked most and bans the maps it has played least."""
    if len(pool) < 7:
        return float(series_price(np.array([gap]))[0])
    adv = {m: gap + offsets.get((team_a, m), 0.0) - offsets.get((team_b, m), 0.0) for m in pool}
    teams = {1.0: team_a, -1.0: team_b}
    out = 0.0
    for first in (1.0, -1.0):
        left = list(pool)
        picks = []
        for step, actor in enumerate((first, -first, first, -first, first, -first)):
            ban = step in (0, 1, 4, 5)
            if prefs is None:
                key = lambda m: actor * adv[m]
            else:
                kind = "play" if ban else "pick"
                key = lambda m: prefs.get((kind, teams[actor], m), 0.0) + 1e-3 * actor * adv[m]
            m = min(left, key=key) if ban else max(left, key=key)
            left.remove(m)
            if step in (2, 3):
                picks.append(m)
        p1, p2, p3 = regulation_map_win(np.array([adv[picks[0]], adv[picks[1]], adv[left[0]]]))
        out += 0.5 * (p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3)
    return float(out)


def series_price(gap: np.ndarray, var: np.ndarray | None = None) -> np.ndarray:
    gap = np.asarray(gap, float)
    if var is None:
        p = regulation_map_win(gap)
        return p * p * (3 - 2 * p)
    d = gap[:, None] + np.sqrt(np.maximum(var, 0.0))[:, None] * GH_X[None, :]
    p = regulation_map_win(d)
    return (p * p * (3 - 2 * p)) @ GH_W
