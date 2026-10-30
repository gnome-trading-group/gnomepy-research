"""
One player-skill graph measured by two sources, fitted by MAP each week.

    rounds_a[m] ~ Binomial(n[m], sigmoid(s[src] * (mean theta[A] - mean theta[B])))
    dev[r] ~ N(gamma[src] * (theta[player] - mean theta[own five]) + style[player], sigma_r[src]^2)
    theta[p] ~ N(mu[group of p's first map], sigma0^2);  mu[g] ~ N(0, sigma_group^2);  log gamma[src] ~ N(0, 1)
    style[p] ~ N(0, sigma_style^2)

Each map's log likelihood is weighted by 0.5^(age / halflife) (skill drifts, so old maps say less) and by a
per-source weight (FACEIT rounds are noisier than their binomial count suggests). s[FACEIT] lets a round gap on
FACEIT mean a different skill gap than on HLTV; it is a fixed setting because, fitted jointly with the ridge on
skills, it runs away (tiny FACEIT skills times a huge scale cost the ridge nothing). The Laplace diagonal gives
each player a variance, which widens the price of a thin or unfamiliar five.

The style offset matters: a player's stat sits above or below teammates for reasons that are not skill (role,
weapon), and over hundreds of maps that persistent offset would otherwise be read as a large skill gap, which the
ridge then answers by compressing every skill and inflating gamma.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.sparse
from threadpoolctl import threadpool_limits

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e20.data import FACEIT, GROUPS, LINEUP_SIZE, MapRows, RatingRows

_OUTER_X, _OUTER_W = np.polynomial.hermite_e.hermegauss(21)
_OUTER_W = _OUTER_W / _OUTER_W.sum()
_INNER_X, _INNER_W = np.polynomial.hermite_e.hermegauss(21)
_INNER_W = _INNER_W / _INNER_W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)


@dataclass(frozen=True)
class Config:
    """Defaults are the submitted entry, chosen on the Jan-Mar 2026 walk-forward (see the submission README)."""
    window_days: int = 450
    halflife_days: float = 150.0
    # Wider than the first guess of 0.35: with 60k players the ridge toward the group mean was shrinking real
    # gaps (under-confident at 0.35, calibrated at 0.7, over-confident at 1.0).
    sigma0: float = 0.7
    sigma_group: float = 1.0
    scale_faceit: float = 1.0
    # A FACEIT round is noisier evidence than an HLTV round: half weight beat full and quarter weight.
    w_faceit: float = 0.5
    sigma_m: float = 0.35
    sigma_s: float = 0.15
    ratings: bool = True
    # The rating channel compresses skills when it carries full weight.
    w_ratings: float = 0.25
    sigma_style: float = 0.2
    # Multiplies the logit of the raw bo3 price; the entry is calibrated by construction.
    calibration: float = 1.0
    max_iter: int = 1500


@dataclass
class Fit:
    cfg: Config
    player_ids: np.ndarray   # (P,)
    theta: np.ndarray        # (P,)
    var: np.ndarray          # (P,) Laplace diagonal
    group_mu: np.ndarray     # (G,)
    gamma: np.ndarray        # (2,) by source
    iterations: int
    n_maps: np.ndarray       # (P,) weighted map count per player
    cutoff: pd.Timestamp

    def __post_init__(self):
        self._pidx = {int(p): i for i, p in enumerate(self.player_ids)}
        table = regulation_map_win(_GRID[:, None] + self.cfg.sigma_m * _INNER_X[None, :]) @ _INNER_W
        self._map_table = table

    def lineup(self, players, prior_group: int) -> tuple[float, float]:
        mean, var = 0.0, 0.0
        for p in players:
            i = self._pidx.get(int(p))
            if i is None:
                mean += self.group_mu[prior_group]
                var += self.cfg.sigma0 ** 2
            else:
                mean += self.theta[i]
                var += self.var[i]
        return mean / len(players), var / len(players) ** 2

    def price(self, lineup_a, group_a: int, lineup_b, group_b: int) -> float:
        """P(team A wins the bo3) with the lineup uncertainty and series noise integrated out."""
        ma, va = self.lineup(lineup_a, group_a)
        mb, vb = self.lineup(lineup_b, group_b)
        d = (ma - mb) + np.sqrt(va + vb + self.cfg.sigma_s ** 2) * _OUTER_X
        p = np.interp(d, _GRID, self._map_table)
        q = float((p * p * (3 - 2 * p)) @ _OUTER_W)
        q = np.clip(q, 1e-4, 1 - 1e-4)
        return float(1 / (1 + np.exp(-self.cfg.calibration * np.log(q / (1 - q)))))


def _players_and_groups(maps: MapRows) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unique players in first-appearance order, each with the prior group of its first map."""
    flat = np.concatenate([maps.lineup_a, maps.lineup_b], axis=1).ravel()
    groups = np.repeat(maps.group, 2 * LINEUP_SIZE)
    ids, first = np.unique(flat, return_index=True)
    return ids, groups[first], flat, np.searchsorted(ids, flat)


def _window(rows, cutoff: pd.Timestamp, cfg: Config):
    age = (np.datetime64(cutoff) - rows.day) / np.timedelta64(1, "D")
    keep = (age >= 0) & (age < cfg.window_days)
    fields = {k: getattr(rows, k) for k in type(rows).__dataclass_fields__}
    return type(rows)(**{k: v[keep] if isinstance(v, np.ndarray) else v for k, v in fields.items()}), age[keep]


def fit(maps: MapRows, cutoff: pd.Timestamp, cfg: Config, ratings: RatingRows | None = None) -> Fit:
    m, age = _window(maps, cutoff, cfg)
    n_maps = len(m.rounds)
    ids, player_group, flat, pidx = _players_and_groups(m)
    n_p, n_g = len(ids), len(GROUPS)
    if ratings is None:
        ratings = RatingRows(*[np.zeros((0, LINEUP_SIZE) if k == "lineup" else 0, dtype=maps.day.dtype if k == "day" else float)
                               for k in RatingRows.__dataclass_fields__])
    rt, r_age = _window(ratings, cutoff, cfg)
    n_r = len(rt.dev)
    r_rows = np.concatenate([np.arange(n_r), np.repeat(np.arange(n_r), LINEUP_SIZE)])
    r_cols = np.searchsorted(ids, np.concatenate([rt.player, rt.lineup.ravel()]))
    r_val = np.concatenate([np.ones(n_r), np.full(n_r * LINEUP_SIZE, -1.0 / LINEUP_SIZE)])
    R = scipy.sparse.csr_matrix((r_val, (r_rows, r_cols)), shape=(n_r, n_p))
    Rt = R.T.tocsr()
    R2t = R.multiply(R).T.tocsr()
    r_src = rt.source.astype(int)
    w_r = np.where(r_src == FACEIT, cfg.w_faceit, 1.0) * cfg.w_ratings * 0.5 ** (r_age / cfg.halflife_days)
    sigma_r = np.array([rt.dev[r_src == k].std() if (r_src == k).any() else 1.0 for k in (0, 1)])
    inv_var_r = 1 / sigma_r[r_src] ** 2
    rows = np.repeat(np.arange(n_maps), 2 * LINEUP_SIZE)
    sign = np.tile(np.r_[np.full(LINEUP_SIZE, 1.0), np.full(LINEUP_SIZE, -1.0)], n_maps) / LINEUP_SIZE
    M = scipy.sparse.csr_matrix((sign, (rows, pidx)), shape=(n_maps, n_p))
    Mt = M.T.tocsr()
    M2t = M.multiply(M).T.tocsr()
    is_f = (m.source == FACEIT).astype(float)
    w = np.where(m.source == FACEIT, cfg.w_faceit, 1.0) * 0.5 ** (age / cfg.halflife_days)
    r, n = m.rounds_a.astype(float), m.rounds.astype(float)
    G = scipy.sparse.csr_matrix((np.ones(n_p), (player_group, np.arange(n_p))), shape=(n_g, n_p))

    s = np.where(is_f > 0, cfg.scale_faceit, 1.0)

    r_player = np.searchsorted(ids, rt.player)

    def unpack(x):
        return x[:n_p], x[n_p:n_p + n_g], x[n_p + n_g:n_p + n_g + 2], x[n_p + n_g + 2:]

    def objective(x):
        theta, mu, log_gamma, style = unpack(x)
        d = s * (M @ theta)
        pi = 1 / (1 + np.exp(-d))
        ll = w * (r * np.logaddexp(0, -d) + (n - r) * np.logaddexp(0, d))
        resid = theta - mu[player_group]
        gamma = np.exp(log_gamma)[r_src]
        own = R @ theta
        resid_r = rt.dev - gamma * own - style[r_player]
        f = ll.sum() + 0.5 * (resid ** 2).sum() / cfg.sigma0 ** 2 + 0.5 * (mu ** 2).sum() / cfg.sigma_group ** 2 \
            + 0.5 * (w_r * inv_var_r * resid_r ** 2).sum() + 0.5 * (log_gamma ** 2).sum() \
            + 0.5 * (style ** 2).sum() / cfg.sigma_style ** 2
        g_d = -w * (r - n * pi)
        g_r = -w_r * inv_var_r * resid_r
        g_theta = Mt @ (s * g_d) + resid / cfg.sigma0 ** 2 + Rt @ (gamma * g_r)
        g_mu = -(G @ resid) / cfg.sigma0 ** 2 + mu / cfg.sigma_group ** 2
        g_lg = np.bincount(r_src, weights=g_r * own * gamma, minlength=2) + log_gamma
        g_style = np.bincount(r_player, weights=g_r, minlength=n_p) + style / cfg.sigma_style ** 2
        return f, np.concatenate([g_theta, g_mu, g_lg, g_style])

    x0 = np.zeros(2 * n_p + n_g + 2)
    # L-BFGS-B's level-1 BLAS calls on a 130k-vector spawn a thread per host core; inside a 1-vCPU Batch
    # container that oversubscription turned a 10 s fit into 20 minutes.
    with threadpool_limits(limits=1):
        res = scipy.optimize.minimize(objective, x0, jac=True, method="L-BFGS-B",
                                      options={"maxiter": cfg.max_iter, "maxcor": 20, "ftol": 1e-10, "gtol": 1e-5})
    theta, mu, log_gamma, style = unpack(res.x)
    pi = 1 / (1 + np.exp(-(s * (M @ theta))))
    curv = w * n * pi * (1 - pi) * s ** 2
    gamma = np.exp(log_gamma)
    precision = M2t @ curv + R2t @ (w_r * inv_var_r * gamma[r_src] ** 2) + 1 / cfg.sigma0 ** 2
    return Fit(cfg=cfg, player_ids=ids, theta=theta, var=1 / precision, group_mu=mu, gamma=gamma, iterations=int(res.nit),
               n_maps=np.asarray((Mt != 0) @ w).ravel(), cutoff=cutoff)
