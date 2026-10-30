"""Opponent-adjusted player skill over the joint HLTV + FACEIT graph, as one extended Kalman filter.

Every map, from either source, is one observation of the gap between two fives, and every player (an HLTV id, or a
FACEIT-only key) is one Gaussian node. Skill is in round-logit units: a gap D gives each round to team A with
probability sigmoid(D). A FACEIT map is a noisier and rescaled measurement of the same latent skill (`faceit_scale`,
`faceit_rho`), so players linked across sources bridge the two tiers and carry their level into FACEIT-only
opponents. A FACEIT-only player is his competition class's offset (one shared node per class, learned through the linked
players he meets) plus his own deviation, which is how an ESEA Open player is placed below an ESEA Main one before any
result. HLTV and FACEIT maps also carry a per-player performance channel: the team's mean rating gap measures the skill
gap, and each player's rating relative to his teammates measures who in the five is better.

Compared with models/cs2/hltv/player_skill.py the covariance is kept inside each five only: nodes number in the
hundreds of thousands once every FACEIT player is a node, and opponents' covariances are second order."""
from __future__ import annotations

import bisect
import collections
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import map_win_prob

LINEUP_SIZE = 5
UNKNOWN_BASE = 10 ** 9


@dataclass(frozen=True)
class Params:
    tau: float = 0.009319191322975635
    sigma0: float = 0.2536091079975324
    a: float = -0.1081114476105491
    b: float = 0.14696262712175345
    lam: float = 0.0761331162651603
    kappa_max: float = 0.03328732428013525
    c0: float = 7.134081474707468
    rho: float = 0.017231894894383412
    chem_days: int = 180
    faceit_scale: float = 1.0
    faceit_rho: float = 0.03
    faceit_tau: float = 0.009319191322975635
    faceit_sigma0: float = 0.25
    class_mu0: float = -0.35
    class_sigma: float = 0.4
    class_tau: float = 0.003
    use_faceit: bool = True
    newcomer_anchor: float = 0.0
    newcomer_shift: float = 0.0
    newcomer_sigma: float = 0.2536091079975324
    hltv_perf_gamma: float = 0.0
    hltv_perf_sigma: float = 0.2
    faceit_perf_gamma: float = 0.0
    faceit_perf_sigma: float = 0.2
    hltv_dev_gamma: float = 0.0
    hltv_dev_sigma: float = 0.3
    faceit_dev_gamma: float = 0.0
    faceit_dev_sigma: float = 0.3
    lineup_var: float = 0.0
    tie_tolerance: float = 1e-9


class RankPoints:
    def __init__(self, rankings: pd.DataFrame):
        self.dates: list[pd.Timestamp] = []
        self.points: list[dict[int, float]] = []
        if len(rankings):
            r = rankings.assign(date=pd.to_datetime(rankings["date"]))
            for day, g in r.groupby("date", sort=True):
                self.dates.append(pd.Timestamp(day))
                self.points.append(dict(zip(g["team_id"].astype(int), g["points"].astype(float))))

    def __call__(self, team_id: int, day: pd.Timestamp) -> float:
        i = bisect.bisect_left(self.dates, day) - 1
        return self.points[i].get(team_id, 0.0) if i >= 0 else 0.0


@dataclass
class Lineup:
    players: tuple[int, ...]
    mus: list[float]
    effective: list[float]
    variances: list[float]
    block: np.ndarray
    weights: np.ndarray
    strength: float
    five_maps: float


@dataclass
class Gap:
    mean: float
    var: float


class Filter:
    def __init__(self, params: Params, rankings: RankPoints):
        self.p = params
        self.ranks = rankings
        self.mu: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.seen: dict[int, float] = {}
        self.games: collections.Counter = collections.Counter()
        self.cov: dict[tuple[int, int], float] = {}
        self.five_days: dict[frozenset, collections.deque] = collections.defaultdict(collections.deque)
        self.home: dict[int, str] = {}
        self.class_mu: dict[str, float] = {}
        self.class_var: dict[str, float] = {}
        self.class_seen: dict[str, float] = {}
        self.faceit_games: collections.Counter = collections.Counter()
        self.rank_cap = math.inf
        self.anchor: dict[int, float] = {}

    def _class_state(self, comp: str, day: float) -> tuple[float, float]:
        if comp not in self.class_mu:
            return self.p.class_mu0, self.p.class_sigma ** 2
        idle = max(day - self.class_seen[comp], 0.0)
        return self.class_mu[comp], self.class_var[comp] + self.p.class_tau ** 2 * idle

    def _state(self, player: int, team: int, day: float, src: str, comp: str) -> tuple[float, float, float]:
        """State mean, effective skill and variance. A FACEIT-only player lives under the competition class he debuted in: his
        skill is the class offset plus his own deviation, so the class level, learned through the linked players he plays
        against, is his prior."""
        if player not in self.mu and src == "faceit" and player < 0:
            self.home.setdefault(player, comp)
        home = self.home.get(player)
        if home is not None:
            offset = self._class_state(home, day)[0]
            if player not in self.mu:
                return 0.0, offset, self.p.faceit_sigma0 ** 2
            idle = max(day - self.seen[player], 0.0)
            return self.mu[player], self.mu[player] + offset, self.var[player] + self.p.faceit_tau ** 2 * idle
        if player not in self.mu:
            pts = self.ranks(team, pd.Timestamp("2023-01-01") + pd.Timedelta(days=min(day, self.rank_cap))) if src == "hltv" else 0.0
            m = self.p.a + self.p.b * math.log1p(pts)
            if player >= UNKNOWN_BASE:
                anchor = self.anchor.get((player - UNKNOWN_BASE) // 5, m)
                m = (1 - self.p.newcomer_anchor) * m + self.p.newcomer_anchor * anchor + self.p.newcomer_shift
                return m, m, self.p.newcomer_sigma ** 2
            return m, m, self.p.sigma0 ** 2
        idle = max(day - self.seen[player], 0.0)
        return self.mu[player], self.mu[player], self.var[player] + self.p.tau ** 2 * idle

    def _five_maps(self, five: tuple[int, ...], day: float) -> float:
        q = self.five_days.get(frozenset(five))
        if not q:
            return 0.0
        cutoff = day - self.p.chem_days
        return float(sum(1 for d in q if d >= cutoff))

    def _weights(self, mus: list[float]) -> np.ndarray:
        n = len(mus)
        top = max(mus)
        best = next(i for i, m in enumerate(mus) if m >= top - self.p.tie_tolerance)
        w = np.full(n, (1 - self.p.lam) / n)
        w[best] += self.p.lam
        return w

    def lineup(self, players: tuple[int, ...], team: int, day: float, src: str, comp: str) -> Lineup:
        states = [self._state(p, team, day, src, comp) for p in players]
        mus = [s[0] for s in states]
        effective = [s[1] for s in states]
        variances = [s[2] for s in states]
        block = np.diag(variances)
        n = len(players)
        for i in range(n):
            for j in range(i + 1, n):
                c = self.cov.get((players[i], players[j]) if players[i] < players[j] else (players[j], players[i]))
                if c is not None:
                    block[i, j] = block[j, i] = c
        w = self._weights(effective)
        five_maps = self._five_maps(players, day) if n == LINEUP_SIZE else 0.0
        kappa = self.p.kappa_max * (1 - math.exp(-five_maps / self.p.c0))
        strength = float(w @ np.array(effective)) + kappa
        return Lineup(players, mus, effective, variances, block, w, strength, five_maps)

    def _perf_updates(self, mus: np.ndarray, cov: np.ndarray, w: np.ndarray, kap: float, sa, sb, src: str) -> None:
        g_team = self.p.hltv_perf_gamma if src == "hltv" else self.p.faceit_perf_gamma
        s_team = self.p.hltv_perf_sigma if src == "hltv" else self.p.faceit_perf_sigma
        g_dev = self.p.hltv_dev_gamma if src == "hltv" else self.p.faceit_dev_gamma
        s_dev = self.p.hltv_dev_sigma if src == "hltv" else self.p.faceit_dev_sigma
        xa, xb = np.asarray(sa, dtype=float), np.asarray(sb, dtype=float)
        va, vb = np.isfinite(xa), np.isfinite(xb)
        if g_team > 0 and va.sum() >= 4 and vb.sum() >= 4:
            y = xa[va].mean() - xb[vb].mean()
            cw = cov @ w
            var_d = float(w @ cw)
            s = g_team ** 2 * var_d + s_team ** 2
            innov = y - g_team * float(w @ mus + kap)
            mus += g_team * cw * innov / s
            cov -= np.outer(cw, cw) * g_team ** 2 / s
        if g_dev > 0:
            for off, x, v in ((0, xa, va), (len(xa), xb, vb)):
                idx = np.flatnonzero(v)
                if len(idx) < 3:
                    continue
                n = len(idx)
                dev = x[idx] - x[idx].mean()
                h = np.eye(n) - 1.0 / n
                state = mus[off + idx]
                sub = cov[np.ix_(off + idx, off + idx)]
                hh = g_dev * h
                s = hh @ sub @ hh.T + s_dev ** 2 * np.eye(n)
                gain = sub @ hh.T @ np.linalg.inv(s)
                mus[off + idx] += gain @ (dev - hh @ state)
                cov[np.ix_(off + idx, off + idx)] = sub - gain @ hh @ sub

    def observe(self, day: float, team_a: int, team_b: int, pa: tuple[int, ...], pb: tuple[int, ...], ra: int, rb: int,
                src: str, comp: str, sa=None, sb=None) -> None:
        if len(set(pa) | set(pb)) != len(pa) + len(pb):
            return
        a = self.lineup(pa, team_a, day, src, comp)
        b = self.lineup(pb, team_b, day, src, comp)
        scale, rho = (1.0, self.p.rho) if src == "hltv" else (self.p.faceit_scale, self.p.faceit_rho)
        na, nb = len(pa), len(pb)
        jac: dict[str, float] = {}
        for lu, sign in ((a, 1.0), (b, -1.0)):
            for p, wi in zip(lu.players, lu.weights):
                if p in self.home:
                    jac[self.home[p]] = jac.get(self.home[p], 0.0) + sign * wi * scale
        classes = list(jac)
        nc = len(classes)
        w = np.concatenate([a.weights, -b.weights]) * scale
        w = np.concatenate([w, [jac[c] for c in classes]])
        cov = np.zeros((na + nb + nc, na + nb + nc))
        cov[:na, :na], cov[na:na + nb, na:na + nb] = a.block, b.block
        class_states = [self._class_state(c, day) for c in classes]
        for j, (_, v) in enumerate(class_states):
            cov[na + nb + j, na + nb + j] = v
        d = scale * (a.strength - b.strength)
        n = ra + rb
        pi = 1 / (1 + math.exp(-d))
        inflation = 1 + (n - 1) * rho
        grad, curv = (ra - n * pi) / inflation, n * pi * (1 - pi) / inflation
        cw = cov @ w
        denom = 1 + curv * float(w @ cw)
        mus = np.array(a.mus + b.mus + [s[0] for s in class_states]) + cw * grad / denom
        cov = cov - np.outer(cw, cw) * curv / denom
        if sa is not None:
            self._perf_updates(mus, cov, w, scale * (a.strength - float(a.weights @ np.array(a.effective)) - (b.strength - float(b.weights @ np.array(b.effective)))), sa, sb, src)
        for j, c in enumerate(classes):
            self.class_mu[c] = float(mus[na + nb + j])
            self.class_var[c] = max(1e-6, float(cov[na + nb + j, na + nb + j]))
            self.class_seen[c] = day
        players = pa + pb
        for i, p in enumerate(players):
            self.mu[p] = float(mus[i])
            self.var[p] = max(1e-6, float(cov[i, i]))
            self.seen[p] = max(self.seen.get(p, day), day)
            self.games[p] += 1
            if src == "faceit":
                self.faceit_games[p] += 1
        for off, five in ((0, pa), (na, pb)):
            for i in range(len(five)):
                for j in range(i + 1, len(five)):
                    key = (five[i], five[j]) if five[i] < five[j] else (five[j], five[i])
                    self.cov[key] = float(cov[off + i, off + j])
        cutoff = day - self.p.chem_days
        for five in (pa, pb):
            if len(five) == LINEUP_SIZE:
                q = self.five_days[frozenset(five)]
                q.append(day)
                while q and q[0] < cutoff:
                    q.popleft()

    def gap(self, a: Lineup, b: Lineup) -> Gap:
        wa, wb = a.weights, b.weights
        var = float(wa @ a.block @ wa + wb @ b.block @ wb) + 2 * self.p.lineup_var
        return Gap(a.strength - b.strength, var)

    def series_price(self, team_a: int, team_b: int, la: tuple[int, ...], lb: tuple[int, ...], day: float) -> tuple[float, Gap]:
        a = self.lineup(la, team_a, day, "hltv", "hltv_online")
        b = self.lineup(lb, team_b, day, "hltv", "hltv_online")
        g = self.gap(a, b)
        pm = map_win_prob(g.mean, g.var)
        return pm * pm * (3 - 2 * pm), g
