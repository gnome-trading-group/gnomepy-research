"""Opponent-adjusted player skill from ridge regressions on round margin and on HLTV player rating."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg

ANCHOR = {"rating": 1.0}
STAT_TARGETS = ["rating", "swing"]
FEATURES = ["margin", *STAT_TARGETS, "min_n", "exp", "team", "margin_short"]


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


def prepare_faceit(fm: pd.DataFrame, excl: pd.DataFrame, finished: pd.DataFrame) -> pd.DataFrame:
    whole = set(excl.loc[excl.map_name.isna(), "match_id"])
    per_map = set(zip(excl.match_id, excl.map_name))
    keep = ~fm.match_id.isin(whole) & ~pd.Series(list(zip(fm.match_id, fm.map_name)), index=fm.index).isin(per_map)
    f = fm[keep & (fm.team_a_players == 5) & (fm.team_b_players == 5)].copy()
    f = f.dropna(subset=["team_a_score", "team_b_score"])

    def keys(hltv, fid):
        return [int(h) if h == h else str(x) for h, x in zip(hltv, fid)]

    f["team_a_player_ids"] = [keys(h, x) for h, x in zip(f.team_a_player_ids, f.team_a_faceit_ids)]
    f["team_b_player_ids"] = [keys(h, x) for h, x in zip(f.team_b_player_ids, f.team_b_faceit_ids)]
    f = f[f.team_a_player_ids.map(len).eq(5) & f.team_b_player_ids.map(len).eq(5)]
    f["t"] = pd.to_datetime(f.match_time.fillna(f.match_date), utc=True).dt.tz_convert(None)
    f["known"] = f.match_id.map(finished.set_index("match_id").finished_at.map(lambda x: pd.Timestamp(x, unit="s")))
    f = f.dropna(subset=["known"])
    return f[["known", "t", "team_a_player_ids", "team_b_player_ids", "team_a_score", "team_b_score"]].reset_index(drop=True)


def _window(maps: pd.DataFrame, cutoff: pd.Timestamp, halflife: float, window_days: float) -> tuple[pd.DataFrame, np.ndarray]:
    m = maps[(maps.t < cutoff) & (maps.t >= cutoff - pd.Timedelta(days=window_days))]
    return m, 0.5 ** ((cutoff - m.t).dt.total_seconds().values / 86400.0 / halflife)


class SkillFit:
    def __init__(self, maps: pd.DataFrame, cutoff: pd.Timestamp, stats: pd.DataFrame | None = None,
                 faceit: pd.DataFrame | None = None, faceit_weight: float = 2.0,
                 halflife: float = 800.0, lam: float = 5.0, lam_rating: float = 3.0, gamma: float = 1.0,
                 anchored: bool = True, window_days: float = 900.0, team_lam: float | None = None):
        self.anchored = anchored
        m, w_h = _window(maps, cutoff, halflife, window_days)
        cols = ["team_a_player_ids", "team_b_player_ids", "team_a_score", "team_b_score"]
        if faceit is not None:
            f, w_f = _window(faceit[faceit.known < cutoff], cutoff, halflife, window_days)
            allm = pd.concat([m[cols], f[cols]], ignore_index=True)
            w = np.concatenate([w_h, faceit_weight * w_f])
        else:
            allm, w = m[cols], w_h
        self.index: dict = {}
        for col in (allm.team_a_player_ids, allm.team_b_player_ids):
            for lineup in col:
                for p in lineup:
                    self.index.setdefault(p, len(self.index))
        ids = self.index
        n = len(allm)
        rows = np.repeat(np.arange(n), 5)
        ca = np.fromiter((self.index[p] for l in allm.team_a_player_ids for p in l), int, count=5 * n)
        cb = np.fromiter((self.index[p] for l in allm.team_b_player_ids for p in l), int, count=5 * n)
        shape = (n, len(ids))
        X = (sp.csr_matrix((np.ones(5 * n), (rows, ca)), shape=shape) - sp.csr_matrix((np.ones(5 * n), (rows, cb)), shape=shape)).tocsr()
        tot = (allm.team_a_score + allm.team_b_score).values
        y = (allm.team_a_score.values - allm.team_b_score.values) / np.maximum(tot, 1)
        self.team_index: dict = {}
        self.team_effect = np.zeros(0)
        if team_lam is None:
            self.margin_skill = _ridge(X, y, w, lam)
        else:
            for t in np.concatenate([m.team_a_id.values, m.team_b_id.values]):
                self.team_index.setdefault(t, len(self.team_index))
            nh = len(m)
            ta = np.fromiter((self.team_index[t] for t in m.team_a_id.values), int, count=nh)
            tb = np.fromiter((self.team_index[t] for t in m.team_b_id.values), int, count=nh)
            tshape = (n, len(self.team_index))
            XT = sp.csr_matrix((np.ones(nh), (np.arange(nh), ta)), shape=tshape) - sp.csr_matrix((np.ones(nh), (np.arange(nh), tb)), shape=tshape)
            scale = np.sqrt(lam / team_lam)
            sol = _ridge(sp.hstack([X, scale * XT]).tocsr(), y, w, lam)
            self.margin_skill = sol[:len(ids)]
            self.team_effect = scale * sol[len(ids):]
        self.n_maps = np.asarray(sp.csr_matrix((np.repeat(w, 5), (rows, ca)), shape=shape).sum(0)).ravel() + \
            np.asarray(sp.csr_matrix((np.repeat(w, 5), (rows, cb)), shape=shape).sum(0)).ravel()

        self.stat_skill: dict[str, np.ndarray] = {}
        if stats is not None:
            self._fit_stats(m, stats, w_h, lam_rating, gamma)

    def _design(self, own_keys, opp_lists, gamma):
        lookup = np.vectorize(lambda p: self.index.get(p, -1), otypes=[int])
        opp = lookup(np.array([list(o) for o in opp_lists], dtype=object))
        n = len(own_keys)
        own = np.fromiter((self.index.get(p, -1) for p in own_keys), int, count=n)
        rows = np.concatenate([np.arange(n), np.repeat(np.arange(n), 5)])
        cols = np.concatenate([own, opp.ravel()])
        vals = np.concatenate([np.ones(n), np.full(5 * n, -gamma / 5.0)])
        ok = cols >= 0
        return sp.csr_matrix((vals[ok], (rows[ok], cols[ok])), shape=(n, len(self.index)))

    def _solve_targets(self, X, st, w, names, lam):
        for name in names:
            y = st[name].values.astype(float)
            ok = np.isfinite(y)
            mu = np.average(y[ok], weights=w[ok])
            sd = np.sqrt(np.average((y[ok] - mu) ** 2, weights=w[ok]))
            yy = np.where(ok, (y - (ANCHOR.get(name, mu) if self.anchored else mu)) / sd, 0.0)
            self.stat_skill[name] = _ridge(X, yy, w * ok, lam)

    def _fit_stats(self, m, stats, w_map, lam, gamma):
        row_of = pd.Series(np.arange(len(m)), index=m.mapstatsid.values)
        st = stats[stats.mapstatsid.isin(row_of.index) & stats.player_id.isin(self.index) & stats.rating.notna()]
        r = row_of.loc[st.mapstatsid.values].values
        opp_a = np.stack(m.team_b_player_ids.values)[r]
        opp_b = np.stack(m.team_a_player_ids.values)[r]
        opp = np.where(st.is_team_a.values.astype(bool)[:, None], opp_a, opp_b)
        X = self._design(st.player_id.values, list(opp), gamma)
        self._solve_targets(X, st, w_map[r], STAT_TARGETS, lam)

    def lineup(self, ids) -> tuple[float, float]:
        s = [self.margin_skill[self.index[p]] if p in self.index else 0.0 for p in ids]
        n = [self.n_maps[self.index[p]] if p in self.index else 0.0 for p in ids]
        return float(np.sum(s)), float(np.mean(n)) if n else 0.0

    def exp_lineup(self, ids) -> float:
        return float(sum(np.log1p(self.n_maps[self.index[p]]) for p in ids if p in self.index))

    def stat_lineup(self, name, ids) -> float:
        v = self.stat_skill[name]
        return float(sum(v[self.index[p]] for p in ids if p in self.index))

    def team_diff(self, ta, tb) -> float:
        if not self.team_index:
            return 0.0
        return float(sum(sg * self.team_effect[self.team_index[t]] for t, sg in ((ta, 1), (tb, -1)) if t in self.team_index))

    def diff(self, pa, pb, ta=None, tb=None) -> list[float]:
        sa, na = self.lineup(pa)
        sb, nb = self.lineup(pb)
        return [sa - sb] + [self.stat_lineup(k, pa) - self.stat_lineup(k, pb) for k in STAT_TARGETS] + [np.log1p(min(na, nb)), self.exp_lineup(pa) - self.exp_lineup(pb), self.team_diff(ta, tb)]


class MultiFit:
    """The main fit plus an optional short-halflife fit whose margin skill tracks recent form."""

    def __init__(self, maps, cutoff, stats, faceit=None, short_halflife: float | None = None, short_lam: float = 5.0, **kw):
        self.main = SkillFit(maps, cutoff, stats, faceit=faceit, **kw)
        self.short = None
        if short_halflife is not None:
            self.short = SkillFit(maps, cutoff, None, faceit=faceit, halflife=short_halflife, lam=short_lam,
                                  **{k: v for k, v in kw.items() if k in ("faceit_weight", "window_days")})

    def diff(self, pa, pb, ta=None, tb=None) -> list[float]:
        d = self.main.diff(pa, pb, ta, tb)
        short = 0.0 if self.short is None else self.short.lineup(pa)[0] - self.short.lineup(pb)[0]
        return d + [short]
