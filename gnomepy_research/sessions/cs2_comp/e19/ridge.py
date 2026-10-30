"""
Trailing-window ridge fit of player and team round-logit strengths, refit every Monday: a smoother next to the
filter channels.

The filter is a forward pass: a player's early maps are read against opponents whose skills were still priors and
never revisited. Refitting the trailing window every Monday reads every map in it with every other map's
information, which is what the NUTS model does on its 12-month window, at a fraction of the cost. Each map is a
weighted binomial in the lineup gap (sum of player strengths plus a team effect); the weight decays with age and
discounts FACEIT maps and within-map round correlation. Strengths are deviations from each player's debut prior,
so a FACEIT-only division sits at its own level even when it is not connected to HLTV play in the window.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.sparse

from gnomepy_research.sessions.cs2_comp.e19.filter import FACEIT, FilterConfig

WEEK = pd.Timedelta(days=7)
# Debut priors are on the filter's lineup scale (a weighted mean of five); a lineup here is a sum of five.
FIVE = 5.0


class RidgeRater:
    def __init__(self, stream: pd.DataFrame, cfg: FilterConfig, sources: tuple[str, ...],
                 prior: Callable[[int | None, pd.Timestamp, str], tuple[float, float]]):
        self.cfg = cfg
        self.prior = prior
        s = stream[stream.source.isin(sources) & (stream.players_a.map(len) == 5) & (stream.players_b.map(len) == 5)]
        s = s.reset_index(drop=True)
        self.eff = s.eff.to_numpy()
        self.player_col: dict[int, int] = {}
        self.team_col: dict[int, int] = {}
        self.player_prior: dict[int, float] = {}
        rows, cols, vals = [], [], []
        offset = np.zeros(len(s))
        if cfg.ridge_players:
            for i, r in enumerate(s.itertuples(index=False)):
                day = pd.Timestamp(r.day)
                for sign, five, hltv_team in ((1.0, r.players_a, r.hltv_a), (-1.0, r.players_b, r.hltv_b)):
                    for p in five:
                        if p not in self.player_col:
                            self.player_col[p] = len(self.player_col)
                            self.player_prior[p] = prior(hltv_team, day, r.div)[0] / FIVE
                        rows.append(i)
                        cols.append(self.player_col[p])
                        vals.append(sign)
                        offset[i] += sign * self.player_prior[p]
        n_players = len(self.player_col)
        if cfg.ridge_team:
            for i, r in enumerate(s.itertuples(index=False)):
                for sign, key in ((1.0, r.key_a), (-1.0, r.key_b)):
                    if key not in self.team_col:
                        self.team_col[key] = n_players + len(self.team_col)
                    rows.append(i)
                    cols.append(self.team_col[key])
                    vals.append(sign)
        self.n_params = n_players + len(self.team_col)
        self.X = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(s), self.n_params))
        self.offset = offset
        self.rounds_a = s.rounds_a.to_numpy(float)
        self.n = (s.rounds_a + s.rounds_b).to_numpy(float)
        src_w = np.where(s.source.to_numpy() == FACEIT, cfg.ridge_faceit_weight, 1.0)
        self.base_w = src_w / (1 + (self.n - 1) * cfg.rho)
        self.lam = np.full(self.n_params, cfg.ridge_lam_player)
        self.lam[n_players:] = cfg.ridge_lam_team
        self.theta = np.zeros(self.n_params)
        self.fitted_at: pd.Timestamp | None = None

    def fit(self, at: pd.Timestamp) -> int:
        """Refit over the maps dated in [at - window, at); returns the number of maps used."""
        lo = int(np.searchsorted(self.eff, (at - self.cfg.ridge_window_weeks * WEEK).to_datetime64(), side="left"))
        hi = int(np.searchsorted(self.eff, at.to_datetime64(), side="left"))
        self.fitted_at = at
        if hi - lo < 50:
            return hi - lo
        X = self.X[lo:hi]
        active = np.unique(X.indices)
        Xa = X[:, active].tocsr()
        age_weeks = (at - pd.DatetimeIndex(self.eff[lo:hi])).days.to_numpy() / 7.0
        w = self.base_w[lo:hi] * 0.5 ** (age_weeks / self.cfg.ridge_half_life_weeks)
        ra, n, off = self.rounds_a[lo:hi], self.n[lo:hi], self.offset[lo:hi]
        lam = self.lam[active]
        XaT = Xa.T.tocsr()

        def objective(theta):
            eta = Xa @ theta + off
            nll = -(w * (ra * eta - n * np.logaddexp(0.0, eta))).sum() + 0.5 * (lam * theta * theta).sum()
            p = 1 / (1 + np.exp(-eta))
            grad = -(XaT @ (w * (ra - n * p))) + lam * theta
            return nll, grad

        res = scipy.optimize.minimize(objective, self.theta[active], jac=True, method="L-BFGS-B",
                                      options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7})
        self.theta[active] = res.x
        return hi - lo

    def strength(self, five: tuple[int, ...], key: int, hltv_team: int | None, at: pd.Timestamp) -> float:
        total = 0.0
        for p in five if self.cfg.ridge_players else ():
            if p in self.player_col:
                total += self.player_prior[p] + self.theta[self.player_col[p]]
            else:
                total += self.prior(hltv_team, at, "other")[0] / FIVE
        if self.cfg.ridge_team and key in self.team_col:
            total += self.theta[self.team_col[key]]
        return total
