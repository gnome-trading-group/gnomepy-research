"""Map-pool series model: team strength plus shrunk team-by-map effects, mixed over the maps a bo3 could be played on."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

MAPS = ["de_ancient", "de_mirage", "de_nuke", "de_dust2", "de_inferno", "de_anubis", "de_overpass"]
WINDOW_DAYS = 540
HALF_LIFE_DAYS = 180.0
SHARE_DAYS = 150
SHARE_PRIOR = 3.0


def _maps_before(data_dir, cutoff) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", cutoff)
    h = h[~showmatch_mask(h) & h.map_name.isin(MAPS) & h.team_a_won.notna()].copy()
    h["t"] = pd.to_datetime(h.match_time.fillna(h.match_date)).dt.tz_localize(None) if h.match_time.dt.tz is None \
        else h.match_time.dt.tz_convert(None)
    h = h[h.t >= cutoff - pd.Timedelta(days=WINDOW_DAYS)]
    h["age"] = (cutoff - h.t).dt.total_seconds() / 86400
    return h


class MapPoolFit:
    def __init__(self, data_dir, cutoff, lam_team=1.0, lam_map=8.0):
        self.cutoff = pd.Timestamp(cutoff)
        h = _maps_before(data_dir, self.cutoff)
        teams = pd.unique(np.concatenate([h.team_a_id.values, h.team_b_id.values]))
        self.tix = {t: i for i, t in enumerate(teams)}
        self.nt = len(teams)
        self.mix = {m: i for i, m in enumerate(MAPS)}
        ia = h.team_a_id.map(self.tix).values
        ib = h.team_b_id.map(self.tix).values
        im = h.map_name.map(self.mix).values
        n = len(h)
        r = np.arange(n)
        ones = np.ones(n)
        nm = len(MAPS)
        Xt = sp.csr_matrix((np.r_[ones, -ones], (np.r_[r, r], np.r_[ia, ib])), shape=(n, self.nt))
        Xm = sp.csr_matrix((np.r_[ones, -ones], (np.r_[r, r], np.r_[ia * nm + im, ib * nm + im])), shape=(n, self.nt * nm))
        lan = h.is_lan.fillna(0).values.astype(float)
        X = sp.hstack([Xt, Xm]).tocsr()
        y = h.team_a_won.values.astype(float)
        w = 0.5 ** (h.age.values / HALF_LIFE_DAYS)
        pen = np.r_[np.full(self.nt, lam_team), np.full(self.nt * nm, lam_map)]
        d = X.shape[1]

        def f(b):
            z = X @ b[:d] + b[d]
            ll = np.sum(w * (np.logaddexp(0, z) - y * z))
            p = 1 / (1 + np.exp(-z))
            g = X.T @ (w * (p - y))
            g0 = np.sum(w * (p - y))
            return ll + 0.5 * np.sum(pen * b[:d] ** 2), np.r_[g + pen * b[:d], g0]

        res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 300})
        self.b = res.x[:d]
        self.home = res.x[d]
        self.team = self.b[:self.nt]
        self.tmap = self.b[self.nt:].reshape(self.nt, nm)

        recent = h[h.age <= SHARE_DAYS]
        counts = np.zeros((self.nt, nm))
        for col in ("team_a_id", "team_b_id"):
            np.add.at(counts, (recent[col].map(self.tix).values, recent.map_name.map(self.mix).values), 1)
        glob = counts.sum(0) / counts.sum()
        self.share = (counts + SHARE_PRIOR * nm * glob) / (counts.sum(1, keepdims=True) + SHARE_PRIOR * nm)
        self.glob = glob

    def map_logits(self, a, b):
        za = self.team[self.tix[a]] if a in self.tix else 0.0
        zb = self.team[self.tix[b]] if b in self.tix else 0.0
        ma = self.tmap[self.tix[a]] if a in self.tix else np.zeros(len(MAPS))
        mb = self.tmap[self.tix[b]] if b in self.tix else np.zeros(len(MAPS))
        return za - zb + ma - mb + self.home

    def _team_only(self, a, b):
        za = self.team[self.tix[a]] if a in self.tix else 0.0
        zb = self.team[self.tix[b]] if b in self.tix else 0.0
        return np.full(len(MAPS), za - zb + self.home)

    def map_weights(self, a, b, power=1.0):
        sa = self.share[self.tix[a]] if a in self.tix else self.glob
        sb = self.share[self.tix[b]] if b in self.tix else self.glob
        w = (sa * sb) ** (power / 2)
        return w / w.sum()

    def series_prob(self, a, b, power=1.0, use_map=True):
        z = self.map_logits(a, b)
        if not use_map:
            z = self._team_only(a, b)
        p = 1 / (1 + np.exp(-z))
        w = self.map_weights(a, b, power)
        total, norm = 0.0, 0.0
        for i, j, k in combinations(range(len(MAPS)), 3):
            wt = w[i] * w[j] * w[k]
            pi, pj, pk = p[i], p[j], p[k]
            s = pi * pj + pi * (1 - pj) * pk + (1 - pi) * pj * pk
            total += wt * s
            norm += wt
        return total / norm
