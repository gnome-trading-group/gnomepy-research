"""
A penalised (MAP) round-level paired-comparison model over HLTV and FACEIT maps, fitted at one cutoff.

    rounds_A ~ Binomial(n, sigmoid(D)), each map's log-likelihood scaled by w_age * phi_source
    D = kappa_source * (mean(theta_A) - mean(theta_B)) + rho[home A] - rho[home B]
    theta_p ~ N(F_p . c, sigma0^2)                      newcomer prior from what is known about the player
    dev_{p,map} ~ N(gamma_source * (theta_p - mean(theta_own five)), sigma_r_source^2)

kappa_HLTV is 1; FACEIT gets its own kappa and phi, because a round gap in an ESEA ladder game is a different
measurement from one in an HLTV-listed match. phi < 1 stands in for the per-map and per-series noise the full Bayes
model samples: rounds within a map are not independent. F_p holds the rank strength of the player's first team in
the window, whether the player has HLTV maps at all, the highest ESEA division played, and the FACEIT skill level,
each with a missingness indicator through its reference level, so a FACEIT-only newcomer is priced from where it
plays rather than from the average HLTV player.

A fit takes seconds (L-BFGS on sparse design matrices), so every week is refitted from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from gnomepy_research.models.cs2.hltv.bayes_series import RankStrength
from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e14.data import DIVISIONS, FACEIT, HLTV, LINEUP, Bundle

N_REGIONS = 7
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)


@dataclass(frozen=True)
class Config:
    window_days: int = 365
    half_life_days: float = 120.0
    phi: tuple[float, float] = (0.5, 0.3)
    sigma0: float = 0.3
    sigma_r: tuple[float, float] = (0.2, 0.25)
    sigma_m: float = 0.37
    sigma_s: float = 0.25
    use_ratings: bool = True
    faceit_priors: bool = True
    # "touch": only FACEIT maps with an HLTV team or a player who has HLTV maps; the rest of the FACEIT graph is
    # more than one step from any HLTV lineup and mostly costs fitting time.
    faceit_scope: str = "touch"
    lineup_source: str = "any"
    maxiter: int = 2000


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass
class Fit:
    cutoff: pd.Timestamp
    cfg: Config
    player_ids: np.ndarray
    theta: np.ndarray
    var: np.ndarray
    coef: np.ndarray
    rho: np.ndarray
    kappa_f: float
    home: dict[int, int]
    lineups: dict[int, list[int]]
    rank: RankStrength
    strength_mu: float
    strength_sd: float
    level_at: dict[int, int]
    div_at: dict[int, int]
    hltv_players: set
    pidx: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self.pidx = {int(p): i for i, p in enumerate(self.player_ids)}
        table = regulation_map_win(_GRID[:, None] + self.cfg.sigma_m * _GH_X[None, :]) @ _GH_W
        self._map_table = table

    def _features(self, player: int, team: int) -> np.ndarray:
        s = (self.rank(team, self.cutoff) - self.strength_mu) / self.strength_sd
        return feature_row(s, int(player) in self.hltv_players, self.div_at.get(int(player), 0),
                           self.level_at.get(int(player), 0), self.cfg.faceit_priors)

    def team_skill(self, team: int) -> tuple[float, float]:
        five = self.lineups.get(int(team))
        if not five:
            f = self._features(-1, team)
            return float(f @ self.coef), self.cfg.sigma0 ** 2 / LINEUP
        mean, var = 0.0, 0.0
        for p in five:
            i = self.pidx.get(int(p))
            if i is None:
                mean += float(self._features(p, team) @ self.coef)
                var += self.cfg.sigma0 ** 2
            else:
                mean += self.theta[i]
                var += self.var[i]
        return mean / LINEUP, var / LINEUP ** 2

    def price(self, team_a: int, team_b: int, bo_type: int = 3) -> float:
        ma, va = self.team_skill(team_a)
        mb, vb = self.team_skill(team_b)
        gap = ma - mb + self.rho[self.home.get(int(team_a), 0)] - self.rho[self.home.get(int(team_b), 0)]
        sd = np.sqrt(va + vb + self.cfg.sigma_s ** 2)
        p = np.interp(gap + sd * _GH_X, _GRID, self._map_table)
        series = p if bo_type == 1 else p * p * (3 - 2 * p)
        return float(series @ _GH_W)


def feature_row(strength: float, has_hltv: bool, div: int, level: int, faceit_priors: bool) -> np.ndarray:
    base = [1.0, strength, float(has_hltv)]
    if not faceit_priors:
        return np.array(base)
    divs = [float(div == d) for d in range(1, len(DIVISIONS) + 1)]
    levels = [float(0 < level <= 7), float(level == 8), float(level == 9), float(level == 0)]
    return np.array(base + divs + levels)


def _home_regions(maps: pd.DataFrame) -> dict[int, int]:
    m = maps[maps.region > 0]
    t = pd.concat([m[["team_a", "region"]].set_axis(["team", "region"], axis=1),
                   m[["team_b", "region"]].set_axis(["team", "region"], axis=1)])
    c = t.groupby(["team", "region"]).size().reset_index(name="n").sort_values(["team", "n", "region"], ascending=[True, False, True])
    return dict(zip(c.drop_duplicates("team").team.astype(np.int64), c.drop_duplicates("team").region.astype(int)))


def _latest_lineups(maps: pd.DataFrame) -> dict[int, list[int]]:
    a = maps[["avail", "team_a", "la"]].set_axis(["avail", "team", "five"], axis=1)
    b = maps[["avail", "team_b", "lb"]].set_axis(["avail", "team", "five"], axis=1)
    t = pd.concat([a, b], ignore_index=True).sort_values("avail", kind="mergesort")
    last = t.drop_duplicates("team", keep="last")
    return {int(k): list(v) for k, v in zip(last.team, last.five)}


def fit(bundle: Bundle, cutoff: pd.Timestamp, cfg: Config = Config()) -> Fit:
    cutoff = pd.Timestamp(cutoff)
    known = bundle.maps[bundle.maps.avail < cutoff]
    win = known[known.day >= cutoff - pd.Timedelta(days=cfg.window_days)]
    if cfg.faceit_scope == "touch" and (win.source == FACEIT).any():
        kh = known[known.source == HLTV]
        hltv_world = np.unique(np.concatenate([np.array(kh.la.tolist()).ravel(), np.array(kh.lb.tolist()).ravel()]))
        f = (win.source == FACEIT).to_numpy()
        fl = np.concatenate([np.array(win.la[f].tolist()), np.array(win.lb[f].tolist())], axis=1)
        touch = np.isin(fl, hltv_world).any(axis=1) | (win.team_a[f].to_numpy() > 0) | (win.team_b[f].to_numpy() > 0)
        keep = np.ones(len(win), bool)
        keep[np.flatnonzero(f)[~touch]] = False
        win = win[keep]
    age = (cutoff - win.day).dt.days.to_numpy(float)
    phi = np.where(win.source.to_numpy() == FACEIT, cfg.phi[1], cfg.phi[0])
    w = phi * 0.5 ** (age / cfg.half_life_days)
    is_f = win.source.to_numpy() == FACEIT

    la = np.array(win.la.tolist(), dtype=np.int64)
    lb = np.array(win.lb.tolist(), dtype=np.int64)
    players = np.unique(np.concatenate([la.ravel(), lb.ravel()]))
    P, M = len(players), len(win)
    ia = np.searchsorted(players, la)
    ib = np.searchsorted(players, lb)
    rows = np.repeat(np.arange(M), LINEUP)
    X = (sp.csr_matrix((np.full(M * LINEUP, 1.0 / LINEUP), (rows, ia.ravel())), shape=(M, P))
         - sp.csr_matrix((np.full(M * LINEUP, 1.0 / LINEUP), (rows, ib.ravel())), shape=(M, P))).tocsr()
    home = _home_regions(win)
    ha = np.array([home.get(int(t), 0) for t in win.team_a], dtype=np.int64)
    hb = np.array([home.get(int(t), 0) for t in win.team_b], dtype=np.int64)
    R = sp.csr_matrix((np.ones(M), (np.arange(M), ha)), shape=(M, N_REGIONS)) \
        - sp.csr_matrix((np.ones(M), (np.arange(M), hb)), shape=(M, N_REGIONS))
    n = win.rounds.to_numpy(float)
    k = win.rounds_a.to_numpy(float)

    rt = bundle.ratings
    pos = pd.Series(np.arange(M), index=win.index)
    rt = rt[rt.map_row.isin(win.index)] if cfg.use_ratings else rt.iloc[:0]
    rmap = pos.reindex(rt.map_row).to_numpy()
    own = np.where(rt.side.to_numpy()[:, None] == 0, ia[rmap], ib[rmap]) if len(rt) else np.zeros((0, LINEUP), np.int64)
    Rn = len(rt)
    G = (sp.csr_matrix((np.ones(Rn), (np.arange(Rn), np.searchsorted(players, rt.player.to_numpy()))), shape=(Rn, P))
         - sp.csr_matrix((np.full(Rn * LINEUP, 1.0 / LINEUP), (np.repeat(np.arange(Rn), LINEUP), own.ravel())), shape=(Rn, P))).tocsr()
    r_src = rt.source.to_numpy() if Rn else np.zeros(0, int)
    r_w = w[rmap] / phi[rmap] if Rn else np.zeros(0)
    r_sig2 = np.where(r_src == FACEIT, cfg.sigma_r[1], cfg.sigma_r[0]) ** 2
    dev = rt.dev.to_numpy(float)

    rank = RankStrength(bundle.ranks[pd.to_datetime(bundle.ranks.date) < cutoff])
    first = pd.concat([win[["avail", "team_a", "la"]].set_axis(["avail", "team", "five"], axis=1),
                       win[["avail", "team_b", "lb"]].set_axis(["avail", "team", "five"], axis=1)]) \
        .sort_values("avail", kind="mergesort").explode("five").drop_duplicates("five")
    debut_team = dict(zip(first.five.astype(np.int64), first.team.astype(np.int64)))
    debut_day = dict(zip(first.five.astype(np.int64), first.avail))
    raw = np.array([rank(debut_team[p], debut_day[p]) for p in players])
    mu, sd = float(raw.mean()), float(raw.std()) or 1.0
    hl = win[win.source == HLTV]
    hltv_players = set(np.concatenate([np.array(hl.la.tolist()).ravel(), np.array(hl.lb.tolist()).ravel()]).tolist()) if len(hl) else set()
    fw = win[win.source == FACEIT]
    div_at: dict[int, int] = {}
    if len(fw):
        d = pd.DataFrame({"p": np.concatenate([np.array(fw.la.tolist()).ravel(), np.array(fw.lb.tolist()).ravel()]),
                          "d": np.tile(np.repeat(fw.division.to_numpy(), LINEUP), 2)})
        d = d.groupby("p").d.max()
        div_at = dict(zip(d.index.astype(np.int64), d.astype(int)))
    lv = bundle.levels[bundle.levels.avail < cutoff].sort_values("avail", kind="mergesort").drop_duplicates("player", keep="last")
    level_at = dict(zip(lv.player.astype(np.int64), lv.level.astype(int)))
    F = np.array([feature_row((raw[i] - mu) / sd, int(p) in hltv_players, div_at.get(int(p), 0), level_at.get(int(p), 0),
                              cfg.faceit_priors) for i, p in enumerate(players)])
    J = F.shape[1]

    s0 = 1.0 / cfg.sigma0 ** 2
    has_f = bool(is_f.any())

    def unpack(x):
        theta = x[:P]
        c = x[P:P + J]
        logk, gh, gf = x[P + J:P + J + 3]
        rho = np.concatenate([[0.0, 0.0], x[P + J + 3:]])
        return theta, c, logk, gh, gf, rho

    def objective(x):
        theta, c, logk, gh, gf, rho = unpack(x)
        kap = np.where(is_f, np.exp(logk), 1.0)
        xt = X @ theta
        D = kap * xt + R @ rho
        nll = float(np.sum(w * (n * _softplus(D) - k * D)))
        gD = w * (n * _sigmoid(D) - k)
        g_theta = X.T @ (kap * gD)
        g_logk = float(np.sum(gD * kap * xt * is_f))
        g_rho = (R.T @ gD)[2:]
        gt = G @ theta
        gam = np.where(r_src == FACEIT, gf, gh)
        res = dev - gam * gt
        nll += float(np.sum(r_w * res ** 2 / (2 * r_sig2)))
        gr = -r_w * res / r_sig2
        g_theta = g_theta + G.T @ (gr * gam)
        g_gh = float(np.sum(gr * gt * (r_src == HLTV)))
        g_gf = float(np.sum(gr * gt * (r_src == FACEIT)))
        resid = theta - F @ c
        nll += 0.5 * s0 * float(resid @ resid) + 0.5 * float(c @ c) + 0.5 * float(rho @ rho) / 0.09 \
            + 0.5 * logk ** 2 / 0.25 + 0.5 * (gh ** 2 + gf ** 2)
        g_theta = g_theta + s0 * resid
        g_c = -s0 * (F.T @ resid) + c
        g_rho = g_rho + rho[2:] / 0.09
        g_logk = g_logk + logk / 0.25 if has_f else 0.0
        grad = np.concatenate([g_theta, g_c, [g_logk, g_gh + gh, g_gf + gf], g_rho])
        return nll, grad

    x0 = np.zeros(P + J + 3 + N_REGIONS - 2)
    x0[P + J + 1] = 1.0
    x0[P + J + 2] = 1.0
    bounds = [(None, None)] * (P + J) + [(-1.5, 1.5) if has_f else (0.0, 0.0), (0.0, None), (0.0, None)] \
        + [(None, None)] * (N_REGIONS - 2)
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": cfg.maxiter, "maxfun": cfg.maxiter * 2, "ftol": 1e-12, "gtol": 1e-7})
    theta, c, logk, gh, gf, rho = unpack(res.x)
    kap = np.where(is_f, np.exp(logk), 1.0)
    D = kap * (X @ theta) + R @ rho
    curv = w * n * _sigmoid(D) * (1 - _sigmoid(D)) * kap ** 2
    h_diag = (X.multiply(X)).T @ curv + s0
    if Rn:
        gam = np.where(r_src == FACEIT, gf, gh)
        h_diag = h_diag + (G.multiply(G)).T @ (r_w * gam ** 2 / r_sig2)
    return Fit(cutoff=cutoff, cfg=cfg, player_ids=players, theta=theta, var=1.0 / h_diag, coef=c, rho=rho,
               kappa_f=float(np.exp(logk)), home=_home_regions(known), lineups=_latest_lineups(known if cfg.lineup_source == "any" else known[known.source == HLTV]), rank=rank,
               strength_mu=mu, strength_sd=sd, level_at=level_at, div_at=div_at, hltv_players=hltv_players)
