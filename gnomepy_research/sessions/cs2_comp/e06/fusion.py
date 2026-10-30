"""The two filters side by side, and the bridge that turns FACEIT skill into evidence about HLTV skill.

The HLTV filter sees only HLTV maps and the FACEIT filter only FACEIT maps, so each source keeps its own scale, noise and
newcomer prior and an amateur-heavy FACEIT graph cannot drag HLTV skills around. Players present in both (linked through
hltv_faceit_map) identify a regression of HLTV skill on FACEIT skill, refitted every Monday from the filters' own states;
for a player the FACEIT state, mapped through the regression with its residual variance added, is a second Gaussian
measurement fused with the HLTV state (or the HLTV newcomer prior) by precision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e06.events import FACEIT, FIVE, MapEvent
from gnomepy_research.sessions.cs2_comp.e06.skill import TIE_TOLERANCE, Params, SkillFilter, map_price, series_price

WEEK = 7 * 86400.0
MONDAY_OFFSET = 4 * 86400.0


def week_start(t: float) -> float:
    """The Monday 00:00 UTC at or before t (1970-01-01 was a Thursday)."""
    return math.floor((t - MONDAY_OFFSET) / WEEK) * WEEK + MONDAY_OFFSET


@dataclass(frozen=True)
class Bridge:
    alpha: float
    beta: float
    resid_var: float
    n: int


class Combined:
    def __init__(self, params: Params, rankings: pd.DataFrame | None, use_faceit: bool, joint: bool = True):
        self.p = params
        self.joint = joint
        self.h = SkillFilter(params, rankings)
        self.f = SkillFilter(params, None) if use_faceit and not joint else None
        self.bridge: Bridge | None = None
        self._week: float | None = None

    def advance(self, t: float) -> None:
        """Refit the bridge at the first call in each week, from the states as they stood at that Monday."""
        w = week_start(t)
        if w != self._week:
            self._week = w
            self.bridge = self._fit_bridge() if self.f is not None else None

    def observe(self, ev: MapEvent) -> None:
        self.advance(ev.t)
        (self.f if ev.source == FACEIT and not self.joint else self.h).observe(ev)

    def _fit_bridge(self) -> Bridge | None:
        p, h, f = self.p, self.h, self.f
        xs, ys, vh = [], [], []
        for pl, gf in f.games.items():
            if gf >= p.min_games_f and h.games.get(pl, 0) >= p.min_games_h:
                xs.append(f.mu[pl])
                ys.append(h.mu[pl])
                vh.append(h.var[pl])
        if len(xs) < 40:
            return None
        x, y = np.array(xs), np.array(ys)
        beta = float(np.cov(x, y, bias=True)[0, 1] / max(x.var(), 1e-9))
        alpha = float(y.mean() - beta * x.mean())
        resid = float(np.mean((y - alpha - beta * x) ** 2) - np.mean(vh))
        return Bridge(alpha, beta, max(resid, p.bridge_floor), len(xs))

    def state(self, player: int, day: float, prior: tuple[float, float]) -> tuple[float, float, float]:
        """Fused (mean, variance) of a player's HLTV-scale skill, and the factor by which fusion shrank the variance."""
        p = self.p
        mh, vh = self.h._state(player, day, p.tau, prior)
        b = self.bridge
        if b is None or player not in self.f.mu:
            return mh, vh, 1.0
        mf, vf = self.f._state(player, day, p.tau_f, (0.0, 1.0))
        ef, vef = b.alpha + b.beta * mf, b.beta ** 2 * vf + b.resid_var
        ph, pf = 1.0 / vh, p.f_precision / vef
        var = 1.0 / (ph + pf)
        return (mh * ph + ef * pf) * var, var, var / vh

    def _slot_probability(self, freq: float, cap: float | None) -> float:
        return min(self.p.pi_max if cap is None else cap, self.p.pi_scale * freq)

    def team_moments(self, team: int, day: float, expected: list[tuple[int, float]], extras: list[int],
                     cap: float | None = None) -> tuple[float, float]:
        """Mean and variance of a team's strength when each expected player plays with his own probability and a missing
        one is replaced by a bench player or an unknown stand-in."""
        p, h = self.p, self.h
        prior = h._prior(0, team, day, 0)
        pad = FIVE - len(expected)
        players = [pl for pl, _ in expected]
        pis = np.array([self._slot_probability(fr, cap) for _, fr in expected] + [0.0] * pad)
        states = [self.state(pl, day, prior) for pl in players] + [(prior[0], prior[1], 1.0)] * pad
        mus = np.array([s[0] for s in states])
        var = np.array([s[1] for s in states])
        shrink = np.sqrt(np.array([s[2] for s in states]))
        bench = [self.state(pl, day, prior) for pl in extras[:3]]
        if bench:
            mu_sub = float(np.mean([s[0] for s in bench]))
            var_sub = float(np.mean([s[1] for s in bench])) + p.sigma_sub ** 2 / 4
        else:
            mu_sub, var_sub = float(mus.mean()) - p.sub_gap, p.sigma_sub ** 2
        m = pis * mus + (1 - pis) * mu_sub
        v = pis * var + (1 - pis) * var_sub + pis * (1 - pis) * (mus - mu_sub) ** 2
        top = m.max()
        best = int(np.argmax(m >= top - TIE_TOLERANCE))
        w = np.full(FIVE, (1 - p.lam) / FIVE)
        w[best] += p.lam
        c = np.zeros((FIVE, FIVE))
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                a, b = players[i], players[j]
                cv = h.cov.get((a, b) if a < b else (b, a))
                if cv is not None:
                    c[i, j] = c[j, i] = cv * pis[i] * pis[j] * shrink[i] * shrink[j]
        five_maps = h._five_maps(tuple(players), day) if len(players) == FIVE else 0.0
        kappa = p.kappa_max * (1 - math.exp(-five_maps / p.c0)) * float(np.prod(pis))
        return float(w @ m) + kappa, float(w @ c @ w + (w * w) @ v)

    def price(self, day: float, team_a: int, expected_a, extras_a, team_b: int, expected_b, extras_b,
              cap: float | None = None) -> dict:
        mean_a, var_a = self.team_moments(team_a, day, expected_a, extras_a, cap)
        mean_b, var_b = self.team_moments(team_b, day, expected_b, extras_b, cap)
        mean, var = mean_a - mean_b, var_a + var_b
        return {"d": mean, "var": var, "q": series_price(mean, var, self.p), "p_map": map_price(mean, var, self.p)}

    def price_known(self, day: float, team_a: int, lineup_a: tuple, team_b: int, lineup_b: tuple) -> dict:
        """The same price for fives known for certain; a diagnostic of what lineup uncertainty costs."""
        return self.price(day, team_a, [(pl, 1.0) for pl in lineup_a], [], team_b, [(pl, 1.0) for pl in lineup_b], [], cap=1.0)

    def faceit_depth(self, players) -> int:
        counts = self.h.faceit_games if self.joint else (self.f.games if self.f is not None else {})
        return sum(1 for pl in players if counts.get(pl, 0) > 0)
