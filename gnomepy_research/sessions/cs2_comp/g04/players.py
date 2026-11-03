"""Player-additive map model: team strength is the sum of the announced five's skills plus a shrunk team residual."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g04.model import _maps_before


STATS = ["rating", "adr", "kast", "round_swing_pct", "openr"]


COHESION = False


class PlayerFit:
    def __init__(self, data_dir, cutoff, lam_player=2.0, lam_team=2.0, half_life=300.0, soft=0.0, rating_prior=20.0, use_rating=True):
        self.cutoff = pd.Timestamp(cutoff)
        h = _maps_before(data_dir, self.cutoff)
        h = h[h.team_a_player_ids.map(lambda x: x is not None and len(x) == 5) & h.team_b_player_ids.map(lambda x: x is not None and len(x) == 5)]
        players = pd.unique(np.concatenate(list(h.team_a_player_ids) + list(h.team_b_player_ids)))
        self.pix = {p: i for i, p in enumerate(players)}
        teams = pd.unique(np.concatenate([h.team_a_id.values, h.team_b_id.values]))
        self.tix = {t: i for i, t in enumerate(teams)}
        npl, nt, n = len(players), len(teams), len(h)
        rows, cols, vals = [], [], []
        for r, (pa, pb) in enumerate(zip(h.team_a_player_ids, h.team_b_player_ids)):
            for p in pa:
                rows.append(r); cols.append(self.pix[p]); vals.append(1.0)
            for p in pb:
                rows.append(r); cols.append(self.pix[p]); vals.append(-1.0)
        Xp = sp.csr_matrix((vals, (rows, cols)), shape=(n, npl))
        self.stats = list(STATS) if use_rating else []
        k = len(self.stats)
        self.rz = np.zeros((npl, k))
        if k:
            ps = load(data_dir, "hltv/cs2_player_map_stats", self.cutoff, ["match_date", "side", "player_id", "rating", "adr", "kast", "round_swing_pct", "ek", "ed"])
            ps = ps[ps.side == "all"]
            ps = ps[pd.to_datetime(ps.match_date) >= self.cutoff - pd.Timedelta(days=540)].copy()
            ps["openr"] = ps.ek / (ps.ek + ps.ed).clip(lower=1)
            age = (self.cutoff - pd.to_datetime(ps.match_date)).dt.total_seconds() / 86400
            wt = 0.5 ** (age / half_life)
            ids = np.array(players)
            for j, c in enumerate(self.stats):
                if c == "logn" or c.startswith("f_"):
                    continue
                v = ps[c]
                ok = v.notna()
                mu = 1.0 if c == "rating" else np.average(v[ok], weights=wt[ok])
                sd = np.sqrt(np.average((v[ok] - mu) ** 2, weights=wt[ok]))
                d_ = pd.DataFrame({"p": ps.player_id[ok], "w": wt[ok], "wr": wt[ok] * (v[ok] - mu)})
                g = d_.groupby("p").agg(wr=("wr", "sum"), w=("w", "sum"))
                m = (g.wr / (g.w + rating_prior)) / sd
                self.rz[:, j] = np.array([m.get(p, 0.0) for p in ids])
            fst = [c for c in self.stats if c.startswith("f_") and c != "f_n"]
            if fst:
                fm = load(data_dir, "faceit/faceit_matches", self.cutoff, ["match_id", "finished_at"])
                fs = load(data_dir, "faceit/faceit_player_map_stats", self.cutoff, ["match_id", "hltv_id", "adr", "k_r_ratio"])
                fs = fs[fs.hltv_id.notna()].merge(fm, on="match_id")
                fage = (self.cutoff - pd.to_datetime(fs.finished_at, unit="s")).dt.total_seconds() / 86400
                fs = fs[fage <= 540].assign(w=lambda d: 0.5 ** (fage[d.index] / half_life))
                for c in fst:
                    col = c[2:]
                    v = fs[col]
                    ok = v.notna()
                    mu = np.average(v[ok], weights=fs.w[ok])
                    sd = np.sqrt(np.average((v[ok] - mu) ** 2, weights=fs.w[ok]))
                    gg = pd.DataFrame({"p": fs.hltv_id[ok].astype(int), "w": fs.w[ok], "wr": fs.w[ok] * (v[ok] - mu)}).groupby("p").sum()
                    m = (gg.wr / (gg.w + rating_prior)) / sd
                    self.rz[:, self.stats.index(c)] = np.array([m.get(p, 0.0) for p in ids])
                    if c == fst[0] and "f_n" in self.stats:
                        self.rz[:, self.stats.index("f_n")] = np.log1p(np.array([gg.w.get(p, 0.0) for p in ids]))
            if "logn" in self.stats:
                cnt = ps.assign(w=wt).groupby("player_id").w.sum()
                self.rz[:, self.stats.index("logn")] = np.log1p(np.array([cnt.get(p, 0.0) for p in ids]))
        self.use_rating = use_rating
        r = np.arange(n)
        Xt = sp.csr_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[r, r], np.r_[h.team_a_id.map(self.tix), h.team_b_id.map(self.tix)])), shape=(n, nt))
        wt_all = 0.5 ** (h.age.values / half_life)
        self.together = {}
        for pa, pb, wi in zip(h.team_a_player_ids, h.team_b_player_ids, wt_all):
            for five in (pa, pb):
                key = frozenset(five)
                self.together[key] = self.together.get(key, 0.0) + wi
        ca = np.array([np.log1p(max(self.together[frozenset(pa)] - wi, 0.0)) for pa, wi in zip(h.team_a_player_ids, wt_all)])
        cb = np.array([np.log1p(max(self.together[frozenset(pb)] - wi, 0.0)) for pb, wi in zip(h.team_b_player_ids, wt_all)])
        coh = (ca - cb)[:, None] if COHESION else np.zeros((n, 0))
        kc = coh.shape[1]
        Xr = sp.csr_matrix(np.hstack([np.asarray(Xp @ self.rz), coh]))
        X = sp.hstack([Xp, Xt, Xr]).tocsr()
        y = h.team_a_won.values.astype(float)
        share = (h.team_a_score / (h.team_a_score + h.team_b_score)).fillna(0.5).values
        y = (1 - soft) * y + soft * np.clip(share * 1.0, 0, 1)
        w = 0.5 ** (h.age.values / half_life)
        pen = np.r_[np.full(npl, lam_player), np.full(nt, lam_team), np.zeros(k + kc)]
        d = X.shape[1]

        def f(b):
            z = X @ b[:d] + b[d]
            p = 1 / (1 + np.exp(-z))
            g = X.T @ (w * (p - y)) + pen * b[:d]
            return np.sum(w * (np.logaddexp(0, z) - y * z)) + 0.5 * np.sum(pen * b[:d] ** 2), np.r_[g, np.sum(w * (p - y))]

        res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 300})
        self.player = res.x[:npl]
        self.team = res.x[npl:d - k - kc]
        self.beta = res.x[d - k - kc:d - kc]
        self.beta_coh = res.x[d - kc:d]
        self.home = res.x[d]
        z = X @ res.x[:d] + res.x[d]
        won = h.team_a_won.values.astype(float)
        a = 1.0
        for _ in range(30):
            p = 1 / (1 + np.exp(-a * z))
            a += np.sum(w * (won - p) * z) / (np.sum(w * p * (1 - p) * z * z) + 1e-9)
        self.scale = a

    def _coh(self, five):
        return np.log1p(self.together.get(frozenset(five), 0.0))

    def strength(self, team, lineup):
        s = sum(self.player[self.pix[p]] + self.rz[self.pix[p]] @ self.beta if p in self.pix else 0.0 for p in lineup)
        if COHESION:
            s += self.beta_coh[0] * self._coh(lineup)
        return s + (self.team[self.tix[team]] if team in self.tix else 0.0)

    def map_logit(self, a, la, b, lb):
        return self.scale * (self.strength(a, la) - self.strength(b, lb) + self.home)
