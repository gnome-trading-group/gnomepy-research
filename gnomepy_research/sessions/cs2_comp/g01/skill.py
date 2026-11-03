"""Opponent-adjusted player skill from ridge regressions on round margin and on HLTV player rating."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg

WINDOW_DAYS = 540
ANCHOR = {"rating": 1.0}
STAT_TARGETS = ["rating", "swing"]
FEATURES = ["margin", *STAT_TARGETS, "min_n", "exp"]


STATS_COLUMNS = ["mapstatsid", "is_team_a", "player_id", "side", "rating", "adr", "kast", "ek", "ed", "round_swing_pct"]


def prepare_stats(stats: pd.DataFrame) -> pd.DataFrame:
    st = stats[stats.side == "all"].copy()
    st["entry"] = st.ek - st.ed
    st["swing"] = st.round_swing_pct
    return st


def _ridge(X: sp.csr_matrix, y: np.ndarray, w: np.ndarray, lam: float, prior: np.ndarray | None = None) -> np.ndarray:
    n = X.shape[1]
    prior = np.zeros(n) if prior is None else prior
    Xw = X.T.multiply(w).tocsr()
    A = (Xw @ X + lam * sp.identity(n)).tocsr()
    b = Xw @ y + lam * prior
    sol, _ = cg(A, b, x0=prior, rtol=1e-6, maxiter=500)
    return sol


class SkillFit:
    def __init__(self, maps: pd.DataFrame, cutoff: pd.Timestamp, stats: pd.DataFrame | None = None,
                 halflife: float = 400.0, lam: float = 10.0, lam_rating: float = 10.0, gamma: float = 0.3,
                 anchored: bool = True):
        self.anchored = anchored
        m = maps[(maps.t < cutoff) & (maps.t >= cutoff - pd.Timedelta(days=WINDOW_DAYS))]
        age = (cutoff - m.t).dt.total_seconds().values / 86400.0
        w = 0.5 ** (age / halflife)
        ids = pd.unique(np.concatenate([np.concatenate(m.team_a_player_ids.values), np.concatenate(m.team_b_player_ids.values)]))
        self.index = {p: i for i, p in enumerate(ids)}
        n = len(m)
        rows = np.repeat(np.arange(n), 5)
        ca = np.fromiter((self.index[p] for l in m.team_a_player_ids for p in l), int, count=5 * n)
        cb = np.fromiter((self.index[p] for l in m.team_b_player_ids for p in l), int, count=5 * n)
        shape = (n, len(ids))
        X = (sp.csr_matrix((np.ones(5 * n), (rows, ca)), shape=shape) - sp.csr_matrix((np.ones(5 * n), (rows, cb)), shape=shape)).tocsr()
        tot = (m.team_a_score + m.team_b_score).values
        y = (m.team_a_score.values - m.team_b_score.values) / np.maximum(tot, 1)
        self.margin_skill = _ridge(X, y, w, lam)
        self.n_maps = np.asarray(sp.csr_matrix((np.repeat(w, 5), (rows, ca)), shape=shape).sum(0)).ravel() + \
            np.asarray(sp.csr_matrix((np.repeat(w, 5), (rows, cb)), shape=shape).sum(0)).ravel()

        self.stat_skill: dict[str, np.ndarray] = {}
        if stats is not None:
            self._fit_stats(m, stats, w, lam_rating, gamma)

    def _fit_stats(self, m, stats, w_map, lam, gamma):
        wmap = pd.Series(w_map, index=m.mapstatsid.values)
        opp = {}
        for r in m.itertuples(index=False):
            opp[(r.mapstatsid, True)] = r.team_b_player_ids
            opp[(r.mapstatsid, False)] = r.team_a_player_ids
        st = stats[stats.mapstatsid.isin(wmap.index) & stats.player_id.isin(self.index) & stats.rating.notna()]
        rows, cols, vals = [], [], []
        for i, r in enumerate(st.itertuples(index=False)):
            rows.append(i)
            cols.append(self.index[r.player_id])
            vals.append(1.0)
            for p in opp[(r.mapstatsid, bool(r.is_team_a))]:
                if p in self.index:
                    rows.append(i)
                    cols.append(self.index[p])
                    vals.append(-gamma / 5.0)
        X = sp.csr_matrix((vals, (rows, cols)), shape=(len(st), len(self.index)))
        w = wmap.loc[st.mapstatsid.values].values
        for name in STAT_TARGETS:
            y = st[name].values.astype(float)
            ok = np.isfinite(y)
            mu, sd = np.average(y[ok], weights=w[ok]), np.sqrt(np.average((y[ok] - np.average(y[ok], weights=w[ok])) ** 2, weights=w[ok]))
            yy = np.where(ok, (y - (ANCHOR.get(name, mu) if self.anchored else mu)) / sd, 0.0)
            self.stat_skill[name] = _ridge(X, yy, w * ok, lam)

    def lineup(self, ids) -> tuple[float, float]:
        s = [self.margin_skill[self.index[p]] if p in self.index else 0.0 for p in ids]
        n = [self.n_maps[self.index[p]] if p in self.index else 0.0 for p in ids]
        return float(np.sum(s)), float(np.mean(n)) if n else 0.0

    def exp_lineup(self, ids) -> float:
        return float(sum(np.log1p(self.n_maps[self.index[p]]) for p in ids if p in self.index))

    def stat_lineup(self, name, ids) -> float:
        v = self.stat_skill[name]
        return float(sum(v[self.index[p]] for p in ids if p in self.index))

    def diff(self, pa, pb) -> list[float]:
        sa, na = self.lineup(pa)
        sb, nb = self.lineup(pb)
        return [sa - sb] + [self.stat_lineup(k, pa) - self.stat_lineup(k, pb) for k in STAT_TARGETS] + [np.log1p(min(na, nb)), self.exp_lineup(pa) - self.exp_lineup(pb)]
