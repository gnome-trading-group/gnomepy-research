"""
Player-skill filter over HLTV and FACEIT maps.

Skill is in round-logit units: the lineup gap D gives each round to team A with probability sigmoid(D). Each player is
Gaussian; variance grows tau^2 per idle day. A map updates the ten players with an extended Kalman step on the
binomial rounds. Players of one five only ever reveal their sum, so covariance between teammates is kept (and between
no one else): without it a fixed five's variance would never fall.

FACEIT is its own measurement. Its rounds count for less than an HLTV map's (rho_faceit inflates their variance), its
gap is scaled by scale_faceit, and a FACEIT-only player starts from a prior set by FACEIT skill level.

Individual performance is a second measurement per player-map: the stat-profile rating minus the team's mean on the
map, rating_dev ~ N(gamma_src * (skill - mean skill of the five), sigma_r_src^2). Rounds identify a five's mean; these
deviations are what separate the players in it, which is what matters when the five changes.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e07.data import FACEIT, LINEUP, HLTV, Streams

RECENT_MAPS = 20
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)
_CENTER = np.eye(LINEUP) - 1.0 / LINEUP
_Q = np.linalg.svd(_CENTER)[0][:, : LINEUP - 1]


@dataclass(frozen=True)
class FilterConfig:
    tau: float = 0.009319191322975635
    sigma0: float = 0.2536091079975324
    a: float = -0.1081114476105491
    b: float = 0.14696262712175345
    rho_hltv: float = 0.017231894894383412
    rho_faceit: float = 0.03
    scale_faceit: float = 1.0
    level_slope: float = 0.0
    faceit_prior: float = -0.1
    gamma_hltv: float = 0.0
    sigma_r_hltv: float = 0.25
    gamma_faceit: float = 0.0
    sigma_r_faceit: float = 0.3
    sigma_map: float = 0.37
    sigma_series: float = 0.15
    use_faceit: bool = True
    use_hltv: bool = True
    selection_var: float = 1.0
    recency: float = 0.9
    newcomer_shift: float = 0.0
    rank_prior_faceit: bool = True


def _bo3(p: np.ndarray) -> np.ndarray:
    return p * p * (3 - 2 * p)


class Filter:
    def __init__(self, cfg: FilterConfig, streams: Streams, rank_points):
        self.cfg = cfg
        self.s = streams
        n = streams.n_players
        self.mu = np.zeros(n)
        self.var = np.zeros(n)
        self.seen = np.full(n, np.nan)
        self.games = np.zeros(n, dtype=np.int64)
        self.faceit_games = np.zeros(n, dtype=np.int64)
        self.cov: dict[tuple[int, int], float] = {}
        self.team_times: dict[int, list[float]] = {}
        self.team_fives: dict[int, list[np.ndarray]] = {}
        self.faceit_fives: dict[int, list[tuple[float, np.ndarray]]] = {}
        self.rank_points = rank_points
        self.last_five: dict[int, tuple[np.ndarray, float]] = {}
        self.pos = 0
        noise = cfg.sigma_map
        self._map_table = regulation_map_win(_GRID[:, None] + noise * _GH_X[None, :]) @ _GH_W

    def _prior(self, team: int, t: float, level: float, src: int) -> float:
        c = self.cfg
        if src == FACEIT and (team < 0 or not c.rank_prior_faceit):
            lv = 0.0 if not np.isfinite(level) else (level - 5.0) / 2.0
            return c.faceit_prior + c.level_slope * lv
        return c.a + c.b * math.log1p(self.rank_points(team, t)) if team >= 0 else c.a

    def _enter(self, ids: np.ndarray, team: int, t: float, levels: np.ndarray, src: int) -> tuple[np.ndarray, np.ndarray]:
        tau2 = self.cfg.tau ** 2
        for j, p in enumerate(ids):
            if np.isnan(self.seen[p]):
                self.mu[p] = self._prior(team, t, levels[j], src)
                self.var[p] = self.cfg.sigma0 ** 2
                self.seen[p] = t
        idle = np.maximum(t - self.seen[ids], 0.0)
        self.var[ids] += tau2 * idle
        self.seen[ids] = t
        return self.mu[ids].copy(), self.var[ids].copy()

    def _block(self, ids: np.ndarray, var: np.ndarray) -> np.ndarray:
        P = np.diag(var)
        for i in range(LINEUP):
            for j in range(i + 1, LINEUP):
                p, q = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                c = self.cov.get((p, q))
                if c is not None:
                    P[i, j] = P[j, i] = c
        return P

    def _store(self, ids: np.ndarray, mu: np.ndarray, P: np.ndarray) -> None:
        self.mu[ids] = mu
        self.var[ids] = np.maximum(np.diag(P), 1e-6)
        for i in range(LINEUP):
            for j in range(i + 1, LINEUP):
                p, q = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                self.cov[(p, q)] = float(P[i, j])

    def observe(self, m: int) -> None:
        s, c = self.s, self.cfg
        src = int(s.src[m])
        if (src == FACEIT and not c.use_faceit) or (src == HLTV and not c.use_hltv):
            return
        t = float(s.t[m])
        ia, ib = s.a[m], s.b[m]
        if len(set(ia.tolist()) | set(ib.tolist())) < 2 * LINEUP:
            return
        mu_a, var_a = self._enter(ia, int(s.team_a[m]), t, s.level_a[m], src)
        mu_b, var_b = self._enter(ib, int(s.team_b[m]), t, s.level_b[m], src)
        Pa, Pb = self._block(ia, var_a), self._block(ib, var_b)
        scale = c.scale_faceit if src == FACEIT else 1.0
        rho = c.rho_faceit if src == FACEIT else c.rho_hltv
        w = np.full(LINEUP, 1.0 / LINEUP)
        d = scale * float(w @ mu_a - w @ mu_b)
        va, vb = Pa @ w, Pb @ w
        s2 = scale ** 2 * float(w @ va + w @ vb)
        ra, rb = float(s.rounds_a[m]), float(s.rounds_b[m])
        n = ra + rb
        pi = 1.0 / (1.0 + math.exp(-d))
        infl = 1.0 + (n - 1.0) * rho
        grad, curv = (ra - n * pi) / infl, n * pi * (1 - pi) / infl
        denom = 1.0 + curv * s2
        k = grad * scale / denom
        mu_a = mu_a + va * k
        mu_b = mu_b - vb * k
        f = curv * scale ** 2 / denom
        Pa = Pa - np.outer(va, va) * f
        Pb = Pb - np.outer(vb, vb) * f
        gamma = c.gamma_faceit if src == FACEIT else c.gamma_hltv
        if gamma > 0.0:
            sig = c.sigma_r_faceit if src == FACEIT else c.sigma_r_hltv
            for ids, dev, mu, P in ((ia, s.dev_a[m], mu_a, Pa), (ib, s.dev_b[m], mu_b, Pb)):
                if np.isfinite(dev).all():
                    yq = _Q.T @ dev
                    PQ = P @ _Q
                    S = gamma ** 2 * (_Q.T @ PQ) + sig ** 2 * np.eye(LINEUP - 1)
                    K = gamma * np.linalg.solve(S, PQ.T).T
                    mu += K @ (yq - gamma * (_Q.T @ mu))
                    P -= gamma * K @ PQ.T
        self._store(ia, mu_a, Pa)
        self._store(ib, mu_b, Pb)
        self.games[ia] += 1
        self.games[ib] += 1
        if src == FACEIT:
            self.faceit_games[ia] += 1
            self.faceit_games[ib] += 1
        if src == FACEIT:
            for team, ids in ((int(s.team_a[m]), ia), (int(s.team_b[m]), ib)):
                if team >= 0:
                    fives = self.faceit_fives.setdefault(team, [])
                    fives.append((t, ids.copy()))
                    if len(fives) > 2 * RECENT_MAPS:
                        del fives[: len(fives) - RECENT_MAPS]
        if src == HLTV:
            ta, tb = int(s.team_a[m]), int(s.team_b[m])
            if ta >= 0:
                self.last_five[ta] = (ia.copy(), t)
                self.team_times.setdefault(ta, []).append(t)
                self._remember(ta, ia)
            if tb >= 0:
                self.last_five[tb] = (ib.copy(), t)
                self.team_times.setdefault(tb, []).append(t)
                self._remember(tb, ib)

    def _remember(self, team: int, ids: np.ndarray) -> None:
        fives = self.team_fives.setdefault(team, [])
        fives.append(ids.copy())
        if len(fives) > 2 * RECENT_MAPS:
            del fives[: len(fives) - RECENT_MAPS]

    def recent_fives(self, team: int) -> list[np.ndarray]:
        """The fives of the team's last maps, most recent first."""
        return self.team_fives.get(team, [])[-RECENT_MAPS:][::-1]

    def recent_faceit_fives(self, team: int, t: float, days: float) -> list[np.ndarray]:
        """Fives the team (as inferred from FACEIT) fielded in the last `days`, most recent first."""
        return [ids for ft, ids in self.faceit_fives.get(team, [])[::-1] if t - days <= ft <= t][:RECENT_MAPS]

    def advance(self, before_day: float) -> None:
        t = self.s.t
        while self.pos < len(t) and t[self.pos] < before_day:
            self.observe(self.pos)
            self.pos += 1

    def ids_state(self, ids: np.ndarray, t: float) -> tuple[float, float]:
        """Mean and variance of the mean skill of a five."""
        idle = np.maximum(t - self.seen[ids], 0.0)
        var = self.var[ids] + self.cfg.tau ** 2 * idle
        P = self._block(ids, var)
        w = np.full(LINEUP, 1.0 / LINEUP)
        return float(w @ self.mu[ids]), float(w @ P @ w)

    def players_state(self, ids: list, team: int, t: float, day: float) -> tuple[float, float]:
        """Mean and variance of the mean skill of a lineup whose slots may be unknown (None) or never seen: those take the prior."""
        c = self.cfg
        known = [i for i in ids if i is not None and not np.isnan(self.seen[i])]
        n_fill = LINEUP - len(known)
        mean = sum(self.mu[i] for i in known)
        var = 0.0
        if known:
            k = np.array(known)
            idle = np.maximum(t - self.seen[k], 0.0)
            P = np.zeros((len(k), len(k)))
            for a in range(len(k)):
                P[a, a] = self.var[k[a]] + c.tau ** 2 * idle[a]
                for b in range(a + 1, len(k)):
                    lo, hi = (k[a], k[b]) if k[a] < k[b] else (k[b], k[a])
                    P[a, b] = P[b, a] = self.cov.get((lo, hi), 0.0)
            var = float(P.sum())
        mean += n_fill * self._prior(team, day, np.nan, HLTV)
        var += n_fill * c.sigma0 ** 2
        return mean / LINEUP, var / LINEUP ** 2

    def weighted_state(self, slots: list[tuple[int | None, float]], team: int, t: float, day: float) -> tuple[float, float]:
        """Mean and variance of a lineup given as (player or None, probability he plays); the probabilities sum to 5."""
        c = self.cfg
        prior = self._prior(team, day, np.nan, HLTV) - c.newcomer_shift
        known = [(i, w / LINEUP) for i, w in slots if i is not None and not np.isnan(self.seen[i])]
        unknown_w = sum(w / LINEUP for i, w in slots if i is None or np.isnan(self.seen[i]))
        mean = prior * unknown_w
        var = unknown_w ** 2 * c.sigma0 ** 2
        if known:
            k = np.array([i for i, _ in known])
            w = np.array([x for _, x in known])
            idle = np.maximum(t - self.seen[k], 0.0)
            P = np.diag(self.var[k] + c.tau ** 2 * idle)
            for a in range(len(k)):
                for b in range(a + 1, len(k)):
                    lo, hi = (k[a], k[b]) if k[a] < k[b] else (k[b], k[a])
                    cv = self.cov.get((lo, hi))
                    if cv is not None:
                        P[a, b] = P[b, a] = cv
            mean += float(w @ self.mu[k])
            var += float(w @ P @ w)
            total = float(w.sum() + unknown_w)
            centre = (mean / total) if total > 0 else 0.0
            probs = np.clip(w * LINEUP, 0.0, 1.0)
            var += c.selection_var * float(((probs * (1 - probs)) * (self.mu[k] - centre) ** 2).sum()) / LINEUP ** 2
        return mean, var

    def lineup_state(self, team: int, t: float, day: float, ids: np.ndarray | None = None) -> tuple[float, float]:
        """Mean and variance of the team's mean skill from a five (default: its last known five), or the rank prior."""
        if ids is None:
            known = self.last_five.get(team)
            ids = None if known is None else known[0]
        if ids is None:
            return self._prior(team, day, np.nan, HLTV), self.cfg.sigma0 ** 2 / LINEUP
        return self.ids_state(ids, t)

    def team_maps(self, team: int, t: float, days: float) -> int:
        times = self.team_times.get(team)
        return 0 if not times else len(times) - bisect.bisect_left(times, t - days)

    def profile_weighted(self, slots: list[tuple[int | None, float]], t: float) -> tuple[float, float]:
        """Like profile, for a lineup given as (player or None, probability he plays): the probability-weighted sum of
        (skill - prior centre) / 5 over players with a map in this filter, and the probability mass of those players."""
        total, mass = 0.0, 0.0
        for i, w in slots:
            if i is not None and self.faceit_games[i] > 0:
                total += w * (self.mu[i] - self.cfg.faceit_prior)
                mass += w
        return total / LINEUP, mass / LINEUP

    def profile(self, ids: np.ndarray, t: float) -> tuple[float, float, int]:
        """Sum over the players with a map in this filter of (skill - prior centre) / 5, its variance, and how many."""
        covered = self.faceit_games[ids] > 0
        if not covered.any():
            return 0.0, 0.0, 0
        c = self.cfg
        idle = np.maximum(t - self.seen[ids[covered]], 0.0)
        var = self.var[ids[covered]] + c.tau ** 2 * idle
        return float((self.mu[ids[covered]] - c.faceit_prior).sum() / LINEUP), float(var.sum() / LINEUP ** 2), int(covered.sum())

    def price(self, team_a: int, team_b: int, t: float, cutoff_day: float,
              ids_a: np.ndarray | None = None, ids_b: np.ndarray | None = None,
              lineups: tuple[list, list] | None = None,
              weighted: tuple[list, list] | None = None) -> tuple[float, float, float]:
        if weighted is not None:
            ma, va = self.weighted_state(weighted[0], team_a, t, cutoff_day)
            mb, vb = self.weighted_state(weighted[1], team_b, t, cutoff_day)
        elif lineups is not None:
            ma, va = self.players_state(lineups[0], team_a, t, cutoff_day)
            mb, vb = self.players_state(lineups[1], team_b, t, cutoff_day)
        else:
            ma, va = self.lineup_state(team_a, t, cutoff_day, ids_a)
            mb, vb = self.lineup_state(team_b, t, cutoff_day, ids_b)
        mean, var = ma - mb, va + vb + self.cfg.sigma_series ** 2
        d = mean + math.sqrt(var) * _GH_X
        p = np.interp(d, _GRID, self._map_table)
        return float((_bo3(p) @ _GH_W)), mean, var
