"""
Credibility rating: player and team skills from both sources, fitted as one penalised likelihood (a MAP estimate).

    rounds_A ~ Binomial(n, sigmoid(D)),  D = mean(theta[five A]) - mean(theta[five B]) + gamma[A] - gamma[B]
    theta_p ~ N(F_p . beta, sigma0)      class prior: team rank strength, ESEA division, HLTV-seen, debut
    gamma_t ~ N(0, sigma_team)

Each map's log-likelihood is multiplied by credibility[source] * 0.5^(age / half_life). The credibility factor is the
share of a map's nominal binomial information that is real: rounds within a map are correlated (map noise), and an
online FACEIT map says less about a team's HLTV strength than an HLTV map does. With a Gaussian prior this is
Buhlmann-Straub credibility: a player's estimate is his class mean plus Z times his own evidence, with
Z = e / (e + 1/sigma0^2) where e is his credibility-weighted, decayed number of rounds. FACEIT data enters as more
evidence at its own (lower) credibility, and as the class (ESEA division) that sets the prior of a player HLTV has
never seen.

A series is priced pre-veto: every map slot has the same expected gap, the map's own noise is integrated per map and
a series-level noise is shared by the three slots, so P(series) = E[p^2 (3 - 2p)].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.sparse as sp

from gnomepy_research.models.cs2.hltv.bayes_series import UNRANKED
from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e15.data import FACEIT, HLTV, LINEUP_SIZE, N_DIVISIONS

N_REGIONS = 7

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)


@dataclass(frozen=True)
class Config:
    window_days: int = 365
    half_life_days: float = 240.0
    credibility_hltv: float = 0.5
    credibility_faceit: float = 0.25
    sigma0: float = 0.4
    sigma_team: float = 0.3
    debut_days: int = 60
    map_noise: float = 0.37
    series_noise: float = 0.2
    # Price with each rating's own uncertainty (a diagonal Laplace approximation) on top of the series noise: a team
    # whose five has little credible evidence gets a price closer to a coin flip than its point rating implies.
    parameter_risk: bool = True
    # Second measurement per source: each player's map performance relative to his own five's mean, which splits a
    # five's strength among its players (rounds only see the five's mean). Weight is its credibility, like a map's.
    rating_weight: float = 0.5
    # FACEIT per-player stats in the rating channel: off because they make each fit ~10x slower.
    faceit_stats: bool = False
    # Home-region offsets (Europe fixed at zero): regional scenes meet rarely, so their levels are set by the few
    # cross-region maps rather than by each team's class prior. 0 switches them off.
    sigma_region: float = 0.3


@dataclass
class Fit:
    cfg: Config
    pidx: dict
    theta: np.ndarray
    tidx: dict
    gamma: np.ndarray
    beta: np.ndarray
    rank_strength: dict          # team -> latest -log rank before the cutoff
    strength_mu: float
    strength_sd: float
    latest_five: dict            # team -> five of its latest map before the cutoff
    hltv_seen: set = field(default_factory=set)
    theta_var: np.ndarray = None
    gamma_var: np.ndarray = None
    home: dict = field(default_factory=dict)
    rho: np.ndarray = None
    map_table: np.ndarray = None

    def __post_init__(self):
        self.map_table = regulation_map_win(_GRID[:, None] + self.cfg.map_noise * _GH_X[None, :]) @ _GH_W


def _rank_strength(rankings: pd.DataFrame, cutoff: pd.Timestamp, max_age_days: int = 120) -> dict:
    r = rankings[rankings.date >= pd.Timestamp(cutoff) - pd.Timedelta(days=max_age_days)].sort_values("date")
    latest = r.groupby("team_id")["rank"].last()
    return {int(t): -np.log(float(k)) for t, k in latest.items()}


def _features(strength: np.ndarray, div: np.ndarray, seen: np.ndarray, debut: np.ndarray) -> np.ndarray:
    onehot = np.zeros((len(strength), N_DIVISIONS))
    onehot[np.arange(len(strength)), div] = 1.0
    return np.column_stack([strength, onehot[:, 1:], 1.0 - seen, debut])


def _rating_rows(maps: pd.DataFrame, ratings: pd.DataFrame, ia: np.ndarray, ib: np.ndarray, pidx: dict) -> pd.DataFrame:
    """Performance rows of window maps whose player is in one of the map's fives: map row, own side, deviation."""
    key = pd.DataFrame({"match_id": maps.match_id.to_numpy(), "map_name": maps.map_name.to_numpy(), "row": np.arange(len(maps))})
    key = key.drop_duplicates(["match_id", "map_name"])
    r = ratings.merge(key, on=["match_id", "map_name"])
    if r.empty:
        return r.assign(side=[], col=[], dev=[])
    col = r.player.map(pidx)
    r = r[col.notna()].assign(col=col[col.notna()].astype(np.int64))
    in_a = (ia[r.row.to_numpy()] == r.col.to_numpy()[:, None]).any(axis=1)
    in_b = (ib[r.row.to_numpy()] == r.col.to_numpy()[:, None]).any(axis=1)
    r = r.assign(side=np.where(in_a, 0, np.where(in_b, 1, -1)))
    r = r[r.side >= 0]
    g = r.groupby(["row", "side"]).perf
    r = r.assign(dev=r.perf - g.transform("mean"), cnt=g.transform("size"))
    return r[r.cnt == LINEUP_SIZE].reset_index(drop=True)


def fit(maps: pd.DataFrame, rankings: pd.DataFrame, cutoff: pd.Timestamp, cfg: Config, career_first: pd.Series,
        ratings: pd.DataFrame | None = None) -> Fit:
    """`maps` holds the window's maps (data.before); career_first is each player's first map day before the cutoff."""
    cutoff = pd.Timestamp(cutoff)
    strength_of = _rank_strength(rankings, cutoff)
    unranked = -np.log(UNRANKED)

    fa = np.array([list(f) for f in maps.five_a], dtype=np.int64)
    fb = np.array([list(f) for f in maps.five_b], dtype=np.int64)
    players, inv = np.unique(np.concatenate([fa.ravel(), fb.ravel()]), return_inverse=True)
    ia, ib = inv[: fa.size].reshape(fa.shape), inv[fa.size:].reshape(fb.shape)
    teams, tinv = np.unique(np.concatenate([maps.team_a.to_numpy(), maps.team_b.to_numpy()]), return_inverse=True)
    ta, tb = tinv[: len(maps)], tinv[len(maps):]
    n_m, n_p, n_t = len(maps), len(players), len(teams)

    # Each player's class: the rank strength of the first team he played for in the window, the highest ESEA
    # division he played in it, whether HLTV has seen him, and whether he debuted inside the window.
    first_team = np.full(n_p, -1, dtype=np.int64)
    order = np.argsort(maps.known_at.to_numpy(), kind="stable")
    for i in order[::-1]:
        first_team[ia[i]] = maps.team_a.iat[i]
        first_team[ib[i]] = maps.team_b.iat[i]
    strength_raw = np.array([strength_of.get(int(t), unranked) for t in first_team])
    mu, sd = float(strength_raw.mean()), float(strength_raw.std()) or 1.0
    div = np.zeros(n_p, dtype=int)
    seen = np.zeros(n_p)
    src = maps.src.to_numpy()
    mdiv = maps.division.to_numpy()
    for side in (ia, ib):
        np.maximum.at(div, side.ravel(), np.repeat(mdiv, LINEUP_SIZE))
        np.maximum.at(seen, side.ravel(), np.repeat((src == HLTV).astype(float), LINEUP_SIZE))
    grace = career_first.min() + pd.Timedelta(days=cfg.debut_days) if len(career_first) else cutoff
    start = cutoff - pd.Timedelta(days=cfg.window_days)
    first_day = career_first.reindex(players).to_numpy()
    debut = (first_day >= np.datetime64(max(start, grace))).astype(float)
    F = _features((strength_raw - mu) / sd, div, seen, debut)

    rows = np.repeat(np.arange(n_m), LINEUP_SIZE)
    A = sp.csr_matrix((np.full(fa.size, 1.0 / LINEUP_SIZE), (rows, ia.ravel())), shape=(n_m, n_p)) \
        - sp.csr_matrix((np.full(fb.size, 1.0 / LINEUP_SIZE), (rows, ib.ravel())), shape=(n_m, n_p))
    T = sp.csr_matrix((np.ones(n_m), (np.arange(n_m), ta)), shape=(n_m, n_t)) \
        - sp.csr_matrix((np.ones(n_m), (np.arange(n_m), tb)), shape=(n_m, n_t))
    age = (np.datetime64(cutoff) - maps.day.to_numpy()).astype("timedelta64[s]").astype(float) / 86400.0
    w = np.where(src == FACEIT, cfg.credibility_faceit, cfg.credibility_hltv) * 0.5 ** (age / cfg.half_life_days)
    k = maps.rounds_a.to_numpy(float)
    n = maps.rounds.to_numpy(float)
    AT, TT = A.T.tocsr(), T.T.tocsr()
    n_f = F.shape[1]
    inv0, invt = 1.0 / cfg.sigma0 ** 2, 1.0 / cfg.sigma_team ** 2

    home = _home_regions(maps)
    n_reg = N_REGIONS if cfg.sigma_region > 0 else 0
    R = sp.csr_matrix((n_m, 0))
    if n_reg:
        ra = np.array([home.get(int(t), 0) for t in maps.team_a.to_numpy()])
        rb = np.array([home.get(int(t), 0) for t in maps.team_b.to_numpy()])
        R = (sp.csr_matrix((np.ones(n_m), (np.arange(n_m), ra)), shape=(n_m, n_reg))
             - sp.csr_matrix((np.ones(n_m), (np.arange(n_m), rb)), shape=(n_m, n_reg)))[:, 2:].tocsr()
    RT = R.T.tocsr()
    n_rho = R.shape[1]

    rr = _rating_rows(maps, ratings, ia, ib, {int(p): i for i, p in enumerate(players)}) \
        if cfg.rating_weight > 0 and ratings is not None else pd.DataFrame()
    n_r = len(rr)
    if n_r:
        own = np.where(rr.side.to_numpy()[:, None] == 0, ia[rr.row.to_numpy()], ib[rr.row.to_numpy()])
        B = sp.csr_matrix((np.ones(n_r), (np.arange(n_r), rr.col.to_numpy())), shape=(n_r, n_p)) \
            - sp.csr_matrix((np.full(own.size, 1.0 / LINEUP_SIZE), (np.repeat(np.arange(n_r), LINEUP_SIZE), own.ravel())),
                            shape=(n_r, n_p))
        BT = B.T.tocsr()
        r_src = rr.src.to_numpy().astype(int)
        dev = rr.dev.to_numpy(float)
        r_sd = np.array([dev[r_src == s].std() if (r_src == s).any() else 1.0 for s in (HLTV, FACEIT)])
        r_age = age[rr.row.to_numpy()]
        omega = cfg.rating_weight * 0.5 ** (r_age / cfg.half_life_days) / r_sd[r_src] ** 2
    n_g = 2 if n_r else 0

    # Optimised in prior-whitened coordinates, theta = F beta + sigma0 u and gamma = sigma_team z: the same optimum,
    # but a team effect and its five's mean are nearly collinear, which leaves L-BFGS crawling along a narrow valley
    # in the raw coordinates once FACEIT's tens of thousands of thinly observed players are in.
    n_x = n_p + n_t + n_f + n_rho

    def unpack(x):
        u, z, beta, v = x[:n_p], x[n_p:n_p + n_t], x[n_p + n_t:n_p + n_t + n_f], x[n_p + n_t + n_f:n_x]
        return F @ beta + cfg.sigma0 * u, cfg.sigma_team * z, u, z, beta, v

    def objective(x):
        theta, gamma, u, z, beta, v = unpack(x)
        d = A @ theta + T @ gamma + cfg.sigma_region * (R @ v)
        log_p = -np.logaddexp(0.0, -d)
        log_q = -np.logaddexp(0.0, d)
        f = -np.sum(w * (k * log_p + (n - k) * log_q)) + 0.5 * (u @ u + z @ z + beta @ beta + v @ v)
        r = w * (k - n / (1.0 + np.exp(-d)))
        g_theta = -(AT @ r)
        extra = []
        if n_r:
            slope = x[n_x:]
            bt = B @ theta
            e = dev - slope[r_src] * bt
            f += 0.5 * np.sum(omega * e * e) + 0.5 * np.sum((slope - 1.0) ** 2)
            g_theta = g_theta - BT @ (omega * e * slope[r_src])
            extra = [np.array([-np.sum((omega * e * bt)[r_src == s]) for s in (HLTV, FACEIT)]) + (slope - 1.0)]
        g = np.concatenate([cfg.sigma0 * g_theta + u, -cfg.sigma_team * (TT @ r) + z, F.T @ g_theta + beta,
                            -cfg.sigma_region * (RT @ r) + v] + extra)
        return f, g

    x0 = np.concatenate([np.zeros(n_x), np.ones(n_g)])
    res = opt.minimize(objective, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 5000, "maxcor": 20, "ftol": 1e-12, "gtol": 1e-6})
    theta, gamma, _, _, beta, v = unpack(res.x)
    rho = np.concatenate([np.zeros(2), cfg.sigma_region * v]) if n_rho else np.zeros(N_REGIONS)
    x = np.concatenate([theta, gamma, beta, res.x[n_x:]])
    d = A @ x[:n_p] + T @ x[n_p:n_p + n_t] + R @ rho[2:n_reg]
    s_ = 1.0 / (1.0 + np.exp(-d))
    h = w * n * s_ * (1.0 - s_)
    info = A.multiply(A).T @ h + inv0
    if n_r:
        slope = x[n_p + n_t + n_f:]
        info = info + B.multiply(B).T @ (omega * slope[r_src] ** 2)
    theta_var = 1.0 / info
    gamma_var = 1.0 / (T.multiply(T).T @ h + invt)
    latest: dict = {}
    for i in order:
        latest[int(maps.team_a.iat[i])] = maps.five_a.iat[i]
        latest[int(maps.team_b.iat[i])] = maps.five_b.iat[i]
    return Fit(cfg=cfg, pidx={int(p): i for i, p in enumerate(players)}, theta=x[:n_p],
               tidx={int(t): i for i, t in enumerate(teams)}, gamma=x[n_p:n_p + n_t], beta=x[n_p + n_t:n_p + n_t + n_f],
               rank_strength=strength_of, strength_mu=mu, strength_sd=sd, latest_five=latest,
               hltv_seen={int(p) for p, s in zip(players, seen) if s > 0},
               theta_var=np.asarray(theta_var).ravel(), gamma_var=np.asarray(gamma_var).ravel(), home=home, rho=rho)


def _home_regions(maps: pd.DataFrame) -> dict:
    """Each team's most frequent known region among its window maps (ties to the lower code)."""
    rows = pd.DataFrame({"team": np.concatenate([maps.team_a.to_numpy(), maps.team_b.to_numpy()]),
                         "region": np.concatenate([maps.region.to_numpy(), maps.region.to_numpy()])})
    rows = rows[rows.region > 0]
    if rows.empty:
        return {}
    c = rows.groupby(["team", "region"]).size().reset_index(name="n").sort_values(["team", "n", "region"],
                                                                                  ascending=[True, False, True])
    return {int(t): int(r) for t, r in zip(*c.drop_duplicates("team")[["team", "region"]].to_numpy().T)}


def team_rating(f: Fit, team: int) -> tuple[float, float]:
    """Round-logit strength of a team's latest five plus its team effect, and its variance; unseen players take
    their class mean with the prior's variance."""
    unranked = -np.log(UNRANKED)
    s = (f.rank_strength.get(int(team), unranked) - f.strength_mu) / f.strength_sd
    newcomer = float((_features(np.array([s]), np.array([0]), np.array([1.0]), np.array([1.0])) @ f.beta)[0])
    five = f.latest_five.get(int(team)) or (None,) * LINEUP_SIZE
    known = [f.pidx.get(p) for p in five]
    mean = np.mean([f.theta[i] if i is not None else newcomer for i in known])
    var = sum(f.theta_var[i] if i is not None else f.cfg.sigma0 ** 2 for i in known) / LINEUP_SIZE ** 2
    mean += f.rho[f.home.get(int(team), 0)]
    t = f.tidx.get(int(team))
    if t is None:
        return float(mean), float(var + f.cfg.sigma_team ** 2)
    return float(mean + f.gamma[t]), float(var + f.gamma_var[t])


def series_price(f: Fit, team_a: int, team_b: int) -> float:
    mean_a, var_a = team_rating(f, team_a)
    mean_b, var_b = team_rating(f, team_b)
    shared = f.cfg.series_noise ** 2 + (var_a + var_b if f.cfg.parameter_risk else 0.0)
    p = np.interp(mean_a - mean_b + np.sqrt(shared) * _GH_X, _GRID, f.map_table)
    return float(np.clip((p * p * (3 - 2 * p)) @ _GH_W, 1e-4, 1 - 1e-4))
