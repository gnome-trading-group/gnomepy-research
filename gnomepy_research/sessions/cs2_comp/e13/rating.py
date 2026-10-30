"""
Player power ratings: a Gaussian filter over every player's round-logit skill, fed by HLTV and FACEIT maps.

This extends the HLTV player-skill filter (models/cs2/hltv/player_skill.py; read its docstring for the lineup, chemistry
and covariance mechanics) in the ways a bookmaker's ratings differ from a fixed model:

- Each source is its own measurement. A FACEIT map's rounds are less informative about the HLTV game than an HLTV map's
  (online, lower stakes, mixed formats), so its likelihood carries its own overdispersion rho_f and a weight w_f.
- Players debuting on FACEIT get their prior from the ESEA division they debut in and their FACEIT skill level, not
  from HLTV ranking points their (usually unranked) team does not have.
- Ratings drift per day idle (tau), so a player returning from a break is uncertain again and the next maps move him
  quickly.
- A player's per-map score relative to his four teammates (HLTV rating; FACEIT kills minus deaths per round) is a
  second, linear measurement of his skill relative to theirs, with a slope and noise per source. Rounds only rate a
  five as a whole; this says who in it carries the team, which is what prices a stand-in or a transfer.
"""
from __future__ import annotations

import bisect
import collections
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import TIE_TOLERANCE

FACEIT = 1


@dataclass(frozen=True)
class RatingConfig:
    tau: float = 0.012
    sigma0: float = 0.2536091079975324
    a: float = -0.1081114476105491
    b: float = 0.14696262712175345
    lam: float = 0.0761331162651603
    kappa_max: float = 0.03328732428013525
    c0: float = 7.134081474707468
    rho: float = 0.017231894894383412
    chem_days: float = 180.0
    rho_f: float = 0.03
    w_f: float = 0.9
    a_f: float = -0.45
    c_div: float = 0.08
    a_unk: float = -0.35
    s_lvl: float = 0.04
    sigma0_f: float = 0.3
    g_h: float = 0.35
    s_h: float = 0.75
    g_f: float = 0.6
    s_f: float = 1.0


@dataclass
class Lineup:
    players: list[int]
    mus: list[float]
    variances: list[float]
    weights: list[float]
    total: float
    five_maps: float = 0.0
    unknown: int = 0
    faceit_only: int = 0
    games: list[int] = field(default_factory=list)


class RatingFilter:
    def __init__(self, cfg: RatingConfig, rankings: pd.DataFrame, faceit_level: dict[int, float]):
        self.cfg = cfg
        self.level = faceit_level
        self.mu: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.seen: dict[int, float] = {}
        self.cov: dict[tuple[int, int], float] = {}
        self.games: collections.Counter = collections.Counter()
        self.hltv_games: collections.Counter = collections.Counter()
        self.five_days: dict[frozenset, collections.deque] = collections.defaultdict(collections.deque)
        self._rank_dates: list[pd.Timestamp] = []
        self._rank_points: list[dict[int, float]] = []
        r = rankings.assign(date=pd.to_datetime(rankings["date"]))
        for day, g in r.groupby("date", sort=True):
            self._rank_dates.append(pd.Timestamp(day))
            self._rank_points.append(dict(zip(g["team_id"].astype(int), g["points"].astype(float))))

    def points(self, team_id: int, day: pd.Timestamp) -> float:
        i = bisect.bisect_left(self._rank_dates, day) - 1
        return self._rank_points[i].get(team_id, 0.0) if i >= 0 else 0.0

    def prior(self, player: int, team_id: int, day: pd.Timestamp, src: int, div: int) -> tuple[float, float]:
        c = self.cfg
        pts = math.log1p(self.points(team_id, day)) if team_id > 0 else 0.0
        if src != FACEIT:
            return c.a + c.b * pts, c.sigma0 ** 2
        base = c.a_f + c.c_div * (div - 2) if div >= 0 else c.a_unk
        lvl = self.level.get(player)
        if lvl is not None:
            base += c.s_lvl * (lvl - 10)
        return base + c.b * pts, c.sigma0_f ** 2

    def state(self, player: int, team_id: int, day: pd.Timestamp, now: float, src: int, div: int) -> tuple[float, float]:
        if player not in self.mu:
            return self.prior(player, team_id, day, src, div)
        idle = max(now - self.seen[player], 0.0)
        return self.mu[player], self.var[player] + self.cfg.tau ** 2 * idle

    def five_maps(self, five, now: float) -> float:
        q = self.five_days.get(frozenset(five))
        if not q:
            return 0.0
        cutoff = now - self.cfg.chem_days
        return float(sum(1 for d in q if d >= cutoff))

    def weights(self, mus: list[float]) -> list[float]:
        n = len(mus)
        top = max(mus)
        best = next(i for i, m in enumerate(mus) if m >= top - TIE_TOLERANCE)
        return [1 / n + self.cfg.lam * ((1.0 if i == best else 0.0) - 1 / n) for i in range(n)]

    def lineup(self, players, team_id: int, day: pd.Timestamp, now: float, src: int = 0, div: int = -1,
               unknown: int = 0) -> Lineup:
        states = [self.state(p, team_id, day, now, src, div) for p in players]
        mus = [s[0] for s in states]
        variances = [s[1] for s in states]
        for _ in range(unknown):
            m, v = self.prior(0, team_id, day, 0, -1)
            mus.append(m)
            variances.append(v)
        w = self.weights(mus)
        fm = self.five_maps(players, now) if len(players) == 5 else 0.0
        kappa = self.cfg.kappa_max * (1 - math.exp(-fm / self.cfg.c0))
        return Lineup(list(players), mus, variances, w, sum(a * b for a, b in zip(w, mus)) + kappa, fm, unknown,
                      sum(1 for p in players if self.hltv_games[p] == 0), [self.games[p] for p in players])

    def block(self, players: list[int], variances: list[float]) -> np.ndarray:
        cov = np.diag(variances)
        n = len(players)
        get = self.cov.get
        for i in range(n):
            pi = players[i]
            for j in range(i + 1, n):
                pj = players[j]
                v = get((pi, pj) if pi < pj else (pj, pi))
                if v is not None:
                    cov[i, j] = cov[j, i] = v
        return cov

    def gap(self, a: Lineup, b: Lineup) -> tuple[float, float, np.ndarray, np.ndarray]:
        players = a.players + b.players
        wa = a.weights[:len(a.players)]
        wb = b.weights[:len(b.players)]
        w = np.array(wa + [-x for x in wb])
        cov = self.block(players, a.variances[:len(a.players)] + b.variances[:len(b.players)])
        extra = sum(x * x * v for x, v in zip(a.weights[len(a.players):], a.variances[len(a.players):])) \
            + sum(x * x * v for x, v in zip(b.weights[len(b.players):], b.variances[len(b.players):]))
        return a.total - b.total, float(w @ cov @ w) + extra, cov, w

    def observe_dev(self, players, devs, src: int) -> None:
        """Kalman update of a five on its players' scores relative to the five's mean (call after observe)."""
        c = self.cfg
        g, s = (c.g_f, c.s_f) if src == FACEIT else (c.g_h, c.s_h)
        if g <= 0:
            return
        players = list(players)
        cov = self.block(players, [self.var[p] for p in players])
        mu = np.array([self.mu[p] for p in players])
        h = g * (np.eye(5) - 0.2)
        ph = cov @ h.T
        gain = ph @ np.linalg.inv(h @ ph + s * s * np.eye(5))
        mu = mu + gain @ (np.asarray(devs, float) - h @ mu)
        cov = cov - gain @ ph.T
        for i, p in enumerate(players):
            self.mu[p] = float(mu[i])
            self.var[p] = max(1e-6, float(cov[i, i]))
            for j in range(i + 1, 5):
                q = players[j]
                self.cov[(p, q) if p < q else (q, p)] = float(cov[i, j])

    def observe(self, t: pd.Timestamp, now: float, src: int, div: int, team_a: int, team_b: int, pa, pb,
                rounds_a: int, rounds_b: int) -> float:
        """Update on one map; returns the pre-map round-logit gap."""
        c = self.cfg
        a = self.lineup(pa, team_a, t, now, src, div)
        b = self.lineup(pb, team_b, t, now, src, div)
        d, s2, cov, w = self.gap(a, b)
        n = rounds_a + rounds_b
        pi = 1 / (1 + math.exp(-d))
        rho, wt = (c.rho_f, c.w_f) if src == FACEIT else (c.rho, 1.0)
        inflation = (1 + (n - 1) * rho) / wt
        grad, curv = (rounds_a - n * pi) / inflation, n * pi * (1 - pi) / inflation
        cw = cov @ w
        denom = 1 + curv * s2
        mus = np.array(a.mus + b.mus) + cw * (grad / denom)
        cov = cov - np.outer(cw, cw) * (curv / denom)
        players = a.players + b.players
        hltv = src != FACEIT
        for i, p in enumerate(players):
            self.mu[p] = float(mus[i])
            self.var[p] = max(1e-6, float(cov[i, i]))
            self.seen[p] = now
            self.games[p] += 1
            if hltv:
                self.hltv_games[p] += 1
            for j in range(i + 1, 10):
                q = players[j]
                self.cov[(p, q) if p < q else (q, p)] = float(cov[i, j])
        cutoff = now - c.chem_days
        for five in (pa, pb):
            dq = self.five_days[frozenset(five)]
            dq.append(now)
            while dq and dq[0] < cutoff:
                dq.popleft()
        return d
