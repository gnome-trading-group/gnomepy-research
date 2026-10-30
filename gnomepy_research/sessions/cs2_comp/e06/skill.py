"""Player-skill Kalman filter with an HLTV and a FACEIT measurement channel.

Skill is in round-logit units: a lineup gap D gives each round to team A with probability sigmoid(D). Each player is
Gaussian with variance growing per idle day. A map between two fives updates the ten players jointly (extended Kalman
step on the binomial round count, divided by a beta-binomial inflation). Covariance is kept between teammates only, which
is what lets a fixed five reveal more than its sum once members move, and keeps memory bounded for the tens of
thousands of FACEIT players.

FACEIT maps are a separate measurement of the same skill: their gap is scaled by gamma_f, their rounds carry their own
overdispersion rho_f, idle drift uses tau_f, and a FACEIT-only newcomer starts from a prior set by the competition
division instead of HLTV ranking points.
"""
from __future__ import annotations

import bisect
import collections
import math
from dataclasses import dataclass, field, fields, replace

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e06.events import FACEIT, FIVE, N_DIVISIONS, MapEvent

TIE_TOLERANCE = 1e-9
_X, _W = np.polynomial.hermite_e.hermegauss(21)
_W = _W / _W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)


@dataclass(frozen=True)
class Params:
    tau: float = 0.009319191322975635
    sigma0: float = 0.2536091079975324
    a: float = -0.1081114476105491
    b: float = 0.14696262712175345
    lam: float = 0.0761331162651603
    kappa_max: float = 0.03328732428013525
    c0: float = 7.134081474707468
    delta: float = 0.015162296644758379
    rho: float = 0.017231894894383412
    chem_days: int = 180
    new_player_maps: int = 20
    gamma_f: float = 1.1
    rho_f: float = 0.1
    tau_f: float = 0.006
    sigma0_f: float = 0.3
    a_f: float = -0.6
    div_offsets: tuple = (0.0, 0.17, 0.34, 0.595, 0.17, 0.0)
    sigma_m: float = 0.37
    sigma_s: float = 0.15
    min_games_h: int = 15
    min_games_f: int = 15
    bridge_floor: float = 0.004
    f_precision: float = 1.0
    pi_scale: float = 1.1
    pi_max: float = 0.97
    sub_gap: float = 0.1
    sigma_sub: float = 0.25
    k_h: float = 0.5
    r_h: float = 0.29
    k_f: float = 0.5
    r_f: float = 0.3
    omega: float = 0.05

    def with_values(self, **kv) -> Params:
        return replace(self, **kv)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


NAMES = tuple(f.name for f in fields(Params))


def bo3(p):
    return p * p * (3 - 2 * p)


@dataclass
class Lineup:
    players: tuple
    mus: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    strength: float
    kappa: float
    five_maps: float


class SkillFilter:
    def __init__(self, params: Params, rankings: pd.DataFrame | None):
        self.p = params
        self.mu: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.seen: dict[int, float] = {}
        self.cov: dict[tuple[int, int], float] = {}
        self.games: collections.Counter = collections.Counter()
        self.faceit_games: collections.Counter = collections.Counter()
        self.perf_n: collections.Counter = collections.Counter()
        self.diag: list | None = None
        self.five_days: dict[frozenset, collections.deque] = collections.defaultdict(collections.deque)
        self.rank_limit = math.inf
        self._rank_days: list[float] = []
        self._rank_points: list[dict[int, float]] = []
        if rankings is not None and len(rankings):
            r = rankings.assign(date=pd.to_datetime(rankings["date"]))
            for day, g in r.groupby("date", sort=True):
                self._rank_days.append(pd.Timestamp(day).value / 1e9 / 86400.0)
                self._rank_points.append(dict(zip(g["team_id"].astype(int), g["points"].astype(float))))

    def _points(self, team: int, day: float) -> float:
        i = bisect.bisect_left(self._rank_days, min(day, self.rank_limit)) - 1
        return self._rank_points[i].get(team, 0.0) if i >= 0 else 0.0

    def _prior(self, source: int, team: int, day: float, division: int) -> tuple[float, float]:
        p = self.p
        if source == FACEIT:
            return p.a_f + p.div_offsets[division], p.sigma0_f ** 2
        return p.a + p.b * math.log1p(self._points(team, day)), p.sigma0 ** 2

    def _state(self, player: int, day: float, tau: float, prior: tuple[float, float]) -> tuple[float, float]:
        if player not in self.mu:
            return prior
        idle = max(day - self.seen[player], 0.0)
        return self.mu[player], self.var[player] + tau * tau * idle

    def _five_maps(self, five: tuple, day: float) -> float:
        q = self.five_days.get(frozenset(five))
        if not q:
            return 0.0
        cutoff = day - self.p.chem_days
        return float(sum(1 for d in q if d >= cutoff))

    def _weights(self, mus: np.ndarray) -> np.ndarray:
        top = mus.max()
        best = int(np.argmax(mus >= top - TIE_TOLERANCE))
        w = np.full(len(mus), (1 - self.p.lam) / len(mus))
        w[best] += self.p.lam
        return w

    def lineup(self, players: tuple, day: float, tau: float, priors: list[tuple[float, float]]) -> Lineup:
        states = [self._state(pl, day, tau, pr) for pl, pr in zip(players, priors)]
        mus = np.array([s[0] for s in states])
        var = np.array([s[1] for s in states])
        w = self._weights(mus)
        five_maps = self._five_maps(players, day) if len(players) == FIVE else 0.0
        kappa = self.p.kappa_max * (1 - math.exp(-five_maps / self.p.c0))
        return Lineup(players, mus, var, w, float(w @ mus), kappa, five_maps)

    def _team_cov(self, lu: Lineup) -> np.ndarray:
        c = np.diag(lu.variances)
        pl = lu.players
        for i in range(FIVE):
            for j in range(i + 1, FIVE):
                v = self.cov.get((pl[i], pl[j]) if pl[i] < pl[j] else (pl[j], pl[i]))
                if v is not None:
                    c[i, j] = c[j, i] = v
        return c

    def team_sd2(self, lu: Lineup) -> float:
        return float(lu.weights @ self._team_cov(lu) @ lu.weights)

    def observe(self, ev: MapEvent) -> tuple[float, float] | None:
        """Fold one map in; returns the gap's mean and variance it was predicted with, or None when it was skipped."""
        players = ev.players_a + ev.players_b
        if len(set(players)) != 2 * FIVE:
            return None
        p = self.p
        faceit = ev.source == FACEIT
        tau, gamma, rho = (p.tau_f, p.gamma_f, p.rho_f) if faceit else (p.tau, 1.0, p.rho)
        pa = [self._prior(ev.source, ev.team_a, ev.day, ev.division)] * FIVE
        pb = [self._prior(ev.source, ev.team_b, ev.day, ev.division)] * FIVE
        a = self.lineup(ev.players_a, ev.day, tau, pa)
        b = self.lineup(ev.players_b, ev.day, tau, pb)
        cov = np.zeros((2 * FIVE, 2 * FIVE))
        cov[:FIVE, :FIVE] = self._team_cov(a)
        cov[FIVE:, FIVE:] = self._team_cov(b)
        w = np.concatenate([a.weights, -b.weights]) * gamma
        d = gamma * (a.strength + a.kappa - b.strength - b.kappa)
        s2 = float(w @ cov @ w)
        n = ev.rounds_a + ev.rounds_b
        pi = 1 / (1 + math.exp(-max(-8.0, min(8.0, d))))
        inflation = 1 + (n - 1) * rho
        grad, curv = (ev.rounds_a - n * pi) / inflation, n * pi * (1 - pi) / inflation
        cw = cov @ w
        denom = 1 + curv * s2
        mus = np.concatenate([a.mus, b.mus]) + cw * grad / denom
        cov = cov - np.outer(cw, cw) * curv / denom
        k, r = (p.k_f, p.r_f) if faceit else (p.k_h, p.r_h)
        if k > 0:
            for base, dev in ((0, ev.dev_a), (FIVE, ev.dev_b)):
                if dev is not None:
                    self._perf_update(players[base:base + FIVE], mus, cov, base, np.asarray(dev), k, r, ev.source)
        mu_l, var_l = mus.tolist(), np.maximum(np.diag(cov), 1e-6).tolist()
        for i, pl in enumerate(players):
            self.mu[pl] = mu_l[i]
            self.var[pl] = var_l[i]
            self.seen[pl] = ev.day
            self.games[pl] += 1
            if faceit:
                self.faceit_games[pl] += 1
        for base in (0, FIVE):
            for i in range(FIVE):
                for j in range(i + 1, FIVE):
                    x, y = players[base + i], players[base + j]
                    self.cov[(x, y) if x < y else (y, x)] = float(cov[base + i, base + j])
        cutoff = ev.day - p.chem_days
        for five in (ev.players_a, ev.players_b):
            q = self.five_days[frozenset(five)]
            q.append(ev.day)
            while q and q[0] < cutoff:
                q.popleft()
        return d, s2


    def _perf_update(self, team: tuple, mus: np.ndarray, cov: np.ndarray, base: int, dev: np.ndarray, k: float, r: float,
                     source: int) -> None:
        """Within-team performance deviations measure each player's skill relative to his teammates. A persistent per-player
        offset (role) caps how much repeated measurements can say, so a player's n-th observation is discounted."""
        n = np.array([self.perf_n[pl] for pl in team], dtype=float)
        sl = slice(base, base + FIVE)
        m, pc = mus[sl], cov[sl, sl]
        h = k * (np.eye(FIVE) - 1.0 / FIVE)
        resid = dev - h @ m
        if self.diag is not None:
            self.diag.append((source, resid, h @ m, n))
        om2 = self.p.omega ** 2
        noise = np.diag((r * r + n * om2) ** 2 / (r * r))
        ph = pc @ h.T
        gain = np.linalg.solve(h @ ph + noise, ph.T).T
        mus[sl] = m + gain @ resid
        cov[sl, sl] = pc - gain @ ph.T
        for pl in team:
            self.perf_n[pl] += 1


_TABLES: dict[float, np.ndarray] = {}


def _map_table(sigma_m: float) -> np.ndarray:
    t = _TABLES.get(sigma_m)
    if t is None:
        t = regulation_map_win(_GRID[:, None] + sigma_m * _X[None, :]) @ _W
        _TABLES[sigma_m] = t
    return t


def series_price(mean: float, var: float, p: Params) -> float:
    d = mean + math.sqrt(max(var, 0.0) + p.sigma_s ** 2) * _X
    pm = np.interp(d, _GRID, _map_table(p.sigma_m))
    return float(bo3(pm) @ _W)


def map_price(mean: float, var: float, p: Params) -> float:
    d = mean + math.sqrt(max(var, 0.0)) * _X
    return float(np.interp(d, _GRID, _map_table(p.sigma_m)) @ _W)
