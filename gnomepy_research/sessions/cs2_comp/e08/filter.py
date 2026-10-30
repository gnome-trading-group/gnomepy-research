"""
Player skill as a Gaussian state, updated map by map from HLTV and FACEIT maps alike.

Skill is in round-logit units. A map between fives A and B gives A each round with probability
sigmoid(kappa_src * (S_A - S_B)), where S is the weighted mean of the five's skills (extra weight on the best
player). The rounds are overdispersed (beta-binomial inflation rho_src). FACEIT rounds are a second noisy measurement
of the same skill: they have their own overdispersion and scale kappa_f, and the unlinked players they bring in (most
of the FACEIT population) get a wide prior well below the HLTV newcomer prior, so that linked players are not
credited for beating weaker opposition. Teammates' errors are kept correlated (a fixed five only reveals its sum),
but only within a team, so every map touches two 5x5 blocks.

Each map also carries a within-team performance measurement: the player's stat score minus his team's mean on that
map, a noisy view of skill minus teammates' mean skill. Its persistent part (role, style) is absorbed by a per-player
style term estimated from the filter's own residuals, so a role that always rates high does not read as skill.

A player seen for the first time starts from a prior set by what is known at that moment: the team's HLTV ranking
points on HLTV, the FACEIT skill level and competition tier on FACEIT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gnomepy_research.sessions.cs2_comp.e08.data import FACEIT, FIVE, HLTV

NS_PER_DAY = 86_400e9


@dataclass(frozen=True)
class Params:
    tau2: float = 0.00012
    lam: float = 0.0761331162651603
    a_h: float = -0.1081114476105491
    b_h: float = 0.14696262712175345
    sigma0_h: float = 0.2536091079975324
    a_f: float = -1.0
    b_f: float = 0.10
    comp_off: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sigma0_f: float = 1.0
    kappa_f: float = 1.0
    rho_h: float = 0.017231894894383412
    rho_f: float = 0.02
    gamma_h: float = 0.2
    sig_r_h: float = 0.5
    gamma_f: float = 0.4
    sig_r_f: float = 0.5
    style_shrink: float = 20.0
    f_floor: float = 0.0
    use_faceit: bool = True
    hybrid_min: int = 1


MAX_LOGIT = 30.0
MAX_STEP = 1.0
MAX_SKILL = 3.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-min(max(x, -MAX_LOGIT), MAX_LOGIT)))


class SkillFilter:
    def __init__(self, p: Params):
        self.p = p
        self.tau2 = p.tau2
        self.track_from = -(2 ** 62)
        self.mu: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.seen: dict[int, int] = {}
        self.cov: dict[tuple[int, int], float] = {}
        self.style_sum: dict[int, float] = {}
        self.style_n: dict[int, float] = {}
        self.nll = np.zeros(2)
        self.nmaps = np.zeros(2)

    def _prior(self, feat: float, src: int, comp: int) -> tuple[float, float]:
        p = self.p
        if src == HLTV:
            return p.a_h + p.b_h * feat, p.sigma0_h ** 2
        return p.a_f + p.b_f * (feat - 5.0) + p.comp_off[comp], p.sigma0_f ** 2

    def block(self, players, feats, src: int, comp: int, t: int) -> tuple[np.ndarray, np.ndarray]:
        mu = np.empty(FIVE)
        cov = np.zeros((FIVE, FIVE))
        for i, pl in enumerate(players):
            m = self.mu.get(pl)
            if m is None:
                mu[i], cov[i, i] = self._prior(feats[i], src, comp)
            else:
                mu[i] = m
                cov[i, i] = self.var[pl] + self.tau2 * max((t - self.seen[pl]) / NS_PER_DAY, 0.0)
        cv = self.cov
        for i in range(FIVE):
            pi = players[i]
            for j in range(i + 1, FIVE):
                pj = players[j]
                c = cv.get((pi, pj) if pi < pj else (pj, pi))
                if c is not None:
                    cov[i, j] = cov[j, i] = c
        return mu, cov

    def weights(self, mu: np.ndarray) -> np.ndarray:
        w = np.full(FIVE, (1 - self.p.lam) / FIVE)
        w[int(np.argmax(mu))] += self.p.lam
        return w

    def style(self, players) -> np.ndarray:
        s, n, k = self.style_sum, self.style_n, self.p.style_shrink
        return np.array([s.get(pl, 0.0) / (n.get(pl, 0.0) + k) for pl in players])

    def _rating_update(self, mu: np.ndarray, cov: np.ndarray, dev: np.ndarray, style: np.ndarray, gamma: float,
                       sig_r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = gamma * (np.eye(FIVE) - 1.0 / FIVE)
        pred = h @ mu + style
        resid = dev - pred
        ch = cov @ h.T
        s = h @ ch + sig_r ** 2 * np.eye(FIVE)
        k = np.linalg.solve(s, ch.T).T
        return mu + k @ resid, cov - k @ ch.T, resid

    def update(self, t: int, src: int, comp: int, pa, pb, fa, fb, ra: int, n: int, da, db) -> None:
        p = self.p
        mu_a, ca = self.block(pa, fa, src, comp, t)
        mu_b, cb = self.block(pb, fb, src, comp, t)
        wa, wb = self.weights(mu_a), self.weights(mu_b)
        cwa, cwb = ca @ wa, cb @ wb
        s2 = float(wa @ cwa + wb @ cwb)
        kappa = 1.0 if src == HLTV else p.kappa_f
        rho = p.rho_h if src == HLTV else p.rho_f
        d = kappa * float(wa @ mu_a - wb @ mu_b)
        if t >= self.track_from:
            self.observe_loss(src, ra, n, d)
        pi = _sigmoid(d)
        infl = 1.0 + (n - 1) * rho
        grad, curv = (ra - n * pi) / infl, n * pi * (1 - pi) / infl
        den = 1.0 + kappa * kappa * curv * s2
        step_a, step_b = kappa * cwa * grad / den, kappa * cwb * grad / den
        shrink = min(1.0, MAX_STEP / max(float(np.abs(step_a).max()), float(np.abs(step_b).max()), 1e-12))
        mu_a = mu_a + shrink * step_a
        mu_b = mu_b - shrink * step_b
        scale = kappa * kappa * curv / den
        ca = ca - scale * np.outer(cwa, cwa)
        cb = cb - scale * np.outer(cwb, cwb)

        gamma = p.gamma_h if src == HLTV else p.gamma_f
        sig_r = p.sig_r_h if src == HLTV else p.sig_r_f
        if gamma > 0.0:
            for side, (players, dev) in enumerate(((pa, da), (pb, db))):
                if np.isnan(dev[0]):
                    continue
                mu, cov = (mu_a, ca) if side == 0 else (mu_b, cb)
                style = self.style(players)
                mu, cov, resid = self._rating_update(mu, cov, np.asarray(dev), style, gamma, sig_r)
                for pl, r, st in zip(players, resid, style):
                    self.style_sum[pl] = self.style_sum.get(pl, 0.0) + float(r) + float(st)
                    self.style_n[pl] = self.style_n.get(pl, 0.0) + 1.0
                if side == 0:
                    mu_a, ca = mu, cov
                else:
                    mu_b, cb = mu, cov

        if src == FACEIT and p.f_floor > 0.0:
            ca = ca + np.diag(np.maximum(p.f_floor ** 2 - np.diag(ca), 0.0))
            cb = cb + np.diag(np.maximum(p.f_floor ** 2 - np.diag(cb), 0.0))

        for players, mu, cov in ((pa, mu_a, ca), (pb, mu_b, cb)):
            for i, pl in enumerate(players):
                self.mu[pl] = min(max(float(mu[i]), -MAX_SKILL), MAX_SKILL)
                self.var[pl] = max(1e-6, float(cov[i, i]))
                self.seen[pl] = t
                for j in range(i + 1, FIVE):
                    q = players[j]
                    self.cov[(pl, q) if pl < q else (q, pl)] = float(cov[i, j])

    def observe_loss(self, src: int, ra: int, n: int, d: float) -> None:
        pi = min(max(_sigmoid(d), 1e-9), 1 - 1e-9)
        self.nll[src] -= (ra * math.log(pi) + (n - ra) * math.log(1 - pi)) / n
        self.nmaps[src] += 1
