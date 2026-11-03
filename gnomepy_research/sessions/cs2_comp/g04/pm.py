"""Player-additive round-share model on HLTV and FACEIT maps.

A team's strength is the sum of its five players' skills plus a small HLTV-only team residual. Players are keyed by HLTV id
where one exists and by FACEIT id otherwise, so one person carries one skill across both sources. Shrunk HLTV rating and
experience counts enter as covariates with source-specific coefficients."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g04.model import MAPS, _maps_before


def _keys(hltv_ids, faceit_ids):
    out = []
    for h, f in zip(hltv_ids, faceit_ids):
        out.append(int(h) if h == h else f)
    return out


def _faceit_maps(data_dir, cutoff, window_days):
    fm = load(data_dir, "faceit/faceit_match_history", cutoff)
    fm = fm[fm.team_a_player_ids.map(len).eq(5) & fm.team_b_player_ids.map(len).eq(5)
            & fm.team_a_faceit_ids.map(len).eq(5) & fm.team_b_faceit_ids.map(len).eq(5)]
    ex = load(data_dir, "faceit/faceit_exclusions", cutoff)
    whole = set(ex[ex.map_name.isna()].match_id)
    part = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
    fm = fm[~fm.match_id.isin(whole) & ~pd.Series(list(zip(fm.match_id, fm.map_name)), index=fm.index).isin(part)]
    link = load(data_dir, "faceit/faceit_hltv_series_link", cutoff)
    fm = fm[~fm.match_id.isin(set(link.faceit_match_id))]
    fm = fm[fm.team_a_won.notna()].copy()
    t = pd.to_datetime(fm.match_time)
    fm["t"] = t.dt.tz_convert(None) if t.dt.tz is not None else t
    fm = fm[fm.t >= cutoff - pd.Timedelta(days=window_days)]
    fm["age"] = (cutoff - fm.t).dt.total_seconds() / 86400
    fm["pa"] = [_keys(a, b) for a, b in zip(fm.team_a_player_ids, fm.team_a_faceit_ids)]
    fm["pb"] = [_keys(a, b) for a, b in zip(fm.team_b_player_ids, fm.team_b_faceit_ids)]
    return fm


class Fit:
    def __init__(self, data_dir, cutoff, lam_player=16.0, lam_team=0.5, half_life=1200.0, rating_prior=20.0,
                 faceit_weight=0.5, use_faceit=True, window_days=540, lam_map=None):
        self.cutoff = cutoff = pd.Timestamp(cutoff)
        h = _maps_before(data_dir, cutoff)
        h = h[h.team_a_player_ids.map(lambda x: x is not None and len(x) == 5)
              & h.team_b_player_ids.map(lambda x: x is not None and len(x) == 5)]
        pa = [[int(p) for p in x] for x in h.team_a_player_ids]
        pb = [[int(p) for p in x] for x in h.team_b_player_ids]
        age = h.age.values
        a_score, b_score = h.team_a_score.values, h.team_b_score.values
        won = h.team_a_won.values.astype(float)
        src = np.zeros(len(h))
        ta, tb = h.team_a_id.values, h.team_b_id.values
        mp = list(h.map_name)
        if use_faceit:
            fm = _faceit_maps(data_dir, cutoff, window_days)
            pa += list(fm.pa)
            pb += list(fm.pb)
            age = np.r_[age, fm.age.values]
            a_score = np.r_[a_score, fm.team_a_score.values]
            b_score = np.r_[b_score, fm.team_b_score.values]
            won = np.r_[won, fm.team_a_won.values.astype(float)]
            src = np.r_[src, np.ones(len(fm))]
            mp += list(fm.map_name)
            ta = np.r_[ta, np.full(len(fm), np.nan)]
            tb = np.r_[tb, np.full(len(fm), np.nan)]
        n = len(pa)
        players = pd.unique(np.array([p for l in pa + pb for p in l], dtype=object))
        self.pix = {p: i for i, p in enumerate(players)}
        npl = len(players)
        rows = np.repeat(np.arange(n), 5)
        ca = np.array([self.pix[p] for l in pa for p in l])
        cb = np.array([self.pix[p] for l in pb for p in l])
        Xp = sp.csr_matrix((np.r_[np.ones(5 * n), -np.ones(5 * n)], (np.r_[rows, rows], np.r_[ca, cb])), shape=(n, npl))
        wt = 0.5 ** (age / half_life)

        ids = list(players)
        is_h = np.array([isinstance(p, int) for p in ids])
        ps = load(data_dir, "hltv/cs2_player_map_stats", cutoff, ["match_date", "side", "player_id", "rating"])
        ps = ps[(ps.side == "all") & ps.rating.notna()]
        ps = ps[pd.to_datetime(ps.match_date) >= cutoff - pd.Timedelta(days=window_days)]
        pw = 0.5 ** ((cutoff - pd.to_datetime(ps.match_date)).dt.total_seconds().values / 86400 / half_life)
        g = pd.DataFrame({"p": ps.player_id.values, "w": pw, "wr": pw * (ps.rating.values - 1.0)}).groupby("p").sum()
        rating = np.array([g.wr.get(p, 0.0) / (g.w.get(p, 0.0) + rating_prior) / 0.1 if h_ else 0.0 for p, h_ in zip(ids, is_h)])
        cnt = np.zeros((npl, 2))
        for s in (0, 1):
            m = src == s
            for lists in (pa, pb):
                np.add.at(cnt[:, s], [self.pix[p] for l, k in zip(lists, m) if k for p in l], np.repeat(wt[m], 5))
        self.cov = np.column_stack([rating, np.log1p(cnt[:, 0]), np.log1p(cnt[:, 1])])
        covsum = np.asarray(Xp @ self.cov)
        blocks = [covsum * (src == 0)[:, None], covsum[:, 1:] * (src == 1)[:, None]] if use_faceit else [covsum * (src == 0)[:, None]]
        k = sum(b.shape[1] for b in blocks)
        self.use_faceit = use_faceit

        teams = pd.unique(np.concatenate([ta[src == 0], tb[src == 0]]))
        self.tix = {t: i for i, t in enumerate(teams)}
        nt = len(teams)
        hr = np.where(src == 0)[0]
        Xt = sp.csr_matrix((np.r_[np.ones(len(hr)), -np.ones(len(hr))],
                            (np.r_[hr, hr], np.r_[[self.tix[t] for t in ta[hr]], [self.tix[t] for t in tb[hr]]])), shape=(n, nt))
        mi = np.array([MAPS.index(m) if m in MAPS else -1 for m in mp])
        okm = np.repeat(mi >= 0, 5)
        nm = len(MAPS)
        if lam_map is not None:
            Xm = sp.csr_matrix((np.r_[np.ones(5 * n), -np.ones(5 * n)][np.r_[okm, okm]],
                                (np.r_[rows, rows][np.r_[okm, okm]], np.r_[ca * nm + np.repeat(mi, 5), cb * nm + np.repeat(mi, 5)][np.r_[okm, okm]])),
                               shape=(n, npl * nm))
        else:
            Xm = sp.csr_matrix((n, 0))
        X = sp.hstack([Xp, Xt, sp.csr_matrix(np.hstack(blocks)), sp.csr_matrix(src[:, None]), Xm]).tocsr()
        share = np.where(a_score + b_score > 0, a_score / np.maximum(a_score + b_score, 1), 0.5)
        w = wt * np.where(src == 1, faceit_weight, 1.0)
        pen = np.r_[np.full(npl, lam_player), np.full(nt, lam_team), np.zeros(k + 1), np.full(Xm.shape[1], lam_map or 0.0)]
        d = X.shape[1]

        def f(b):
            z = X @ b[:d] + b[d]
            p = 1 / (1 + np.exp(-z))
            gr = X.T @ (w * (p - share)) + pen * b[:d]
            return np.sum(w * (np.logaddexp(0, z) - share * z)) + 0.5 * np.sum(pen * b[:d] ** 2), np.r_[gr, np.sum(w * (p - share))]

        res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 400})
        b = res.x
        self.player, self.team = b[:npl], b[npl:npl + nt]
        self.beta = np.zeros((3,))
        self.beta[:] = b[npl + nt:npl + nt + 3]
        self.home = b[d]
        self.pmap = b[npl + nt + k + 1:d].reshape(npl, nm) if lam_map is not None else None
        z = X @ b[:d] + b[d]
        hm = src == 0
        a = 1.0
        for _ in range(30):
            p = 1 / (1 + np.exp(-a * z[hm]))
            a += np.sum(w[hm] * (won[hm] - p) * z[hm]) / (np.sum(w[hm] * p * (1 - p) * z[hm] ** 2) + 1e-9)
        self.scale = a

    def strength(self, team, lineup):
        s = 0.0
        for p in lineup:
            i = self.pix.get(int(p))
            if i is not None:
                s += self.player[i] + self.cov[i] @ self.beta
        return s + (self.team[self.tix[team]] if team in self.tix else 0.0)

    def map_effects(self, lineup):
        return sum(self.pmap[self.pix[int(p)]] for p in lineup if int(p) in self.pix) if self.pmap is not None else 0.0

    def experience(self, lineup):
        vals = []
        for p in lineup:
            i = self.pix.get(int(p))
            vals.append(0.0 if i is None else float(self.cov[i, 1] + self.cov[i, 2]))
        return min(vals)

    def map_logit(self, a, la, b, lb):
        return self.scale * (self.strength(a, la) - self.strength(b, lb) + self.home)
