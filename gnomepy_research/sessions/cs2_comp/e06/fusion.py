"""Prices a series from the filter state when the two fives are only partly known.

Each side's five comes from the roster inference as expected players with appearance frequencies. A slot is filled by its
expected player with a probability from that frequency, and otherwise by a bench player or an unknown stand-in; the team's
mean and variance follow from that mixture, with the covariance between teammates the filter kept.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e06.events import FIVE
from gnomepy_research.sessions.cs2_comp.e06.skill import TIE_TOLERANCE, Params, SkillFilter, map_price, series_price


class Combined:
    def __init__(self, params: Params, rankings: pd.DataFrame | None):
        self.p = params
        self.h = SkillFilter(params, rankings)

    def observe(self, ev) -> None:
        self.h.observe(ev)

    def _slot_probability(self, freq: float, cap: float | None) -> float:
        return min(self.p.pi_max if cap is None else cap, self.p.pi_scale * freq)

    def team_moments(self, team: int, day: float, expected: list[tuple[int, float]], extras: list[int],
                     cap: float | None = None) -> tuple[float, float]:
        p, h = self.p, self.h
        prior = h._prior(0, team, day, 0)
        pad = FIVE - len(expected)
        players = [pl for pl, _ in expected]
        pis = np.array([self._slot_probability(fr, cap) for _, fr in expected] + [0.0] * pad)
        states = [h._state(pl, day, p.tau, prior) for pl in players] + [prior] * pad
        mus = np.array([s[0] for s in states])
        var = np.array([s[1] for s in states])
        bench = [h._state(pl, day, p.tau, prior) for pl in extras[:3]]
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
                    c[i, j] = c[j, i] = cv * pis[i] * pis[j]
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
        return sum(1 for pl in players if self.h.faceit_games.get(pl, 0) > 0)
