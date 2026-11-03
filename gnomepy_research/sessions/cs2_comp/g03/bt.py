"""Weekly ridge Bradley-Terry player skill on HLTV maps, optionally with FACEIT maps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from gnomepy_research.sessions.cs2_comp import bundle

ROUND_NORM = 8.0


class Skill:
    """Ridge skill plus a learned slope on per-player covariates (HLTV rating, FACEIT form), so thin-history players inherit a prior."""

    def __init__(self, ridge: dict, beta: np.ndarray, covs: list[dict], scale: float):
        self.ridge, self.beta, self.covs, self.scale = ridge, beta, covs, scale

    def get(self, p, default=0.0):
        v = self.ridge.get(p, 0.0)
        for b, c in zip(self.beta, self.covs):
            v += b * self.scale * c.get(p, 0.0)
        return v

    def __contains__(self, p):
        return p in self.ridge


EMPTY = Skill({}, np.zeros(0), [], 0.0)
MIN_LINKED = 3
FIRST_FIT = pd.Timestamp("2024-07-01")


@dataclass(frozen=True)
class Params:
    window_days: int = 900
    half_life_days: float = 800.0
    c_ridge: float = 0.1
    faceit_weight: float = 1.0
    rounds: bool = True
    team_scale: float = 0.0
    cov_scale: float = 6.0


def _ids(x):
    return [int(v) for v in x if v is not None and v == v]


def faceit_maps(data_dir, cutoff, hist: pd.DataFrame) -> pd.DataFrame:
    h = bundle.load(data_dir, "faceit/faceit_match_history", cutoff,
                    ["match_id", "map_name", "match_time", "team_a_player_ids", "team_b_player_ids", "team_a_won", "team_a_score", "team_b_score"])
    ex = bundle.load(data_dir, "faceit/faceit_exclusions", cutoff)
    link = bundle.load(data_dir, "faceit/faceit_hltv_series_link", cutoff, ["faceit_match_id", "hltv_match_id"])
    twin_t = link.hltv_match_id.map(hist.groupby("match_id").t.min())
    twin = dict(zip(link.faceit_match_id, twin_t))
    fin = bundle.load(data_dir, "faceit/faceit_matches", cutoff, ["match_id", "finished_at"]).set_index("match_id").finished_at
    whole = set(ex[ex.map_name.isna()].match_id)
    bad = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
    h = h[~h.match_id.isin(whole)]
    h = h[[(a, b) not in bad for a, b in zip(h.match_id, h.map_name)]]
    h = h.assign(ta=-1, tb=-1, sa=h.team_a_score, sb=h.team_b_score, a=h.team_a_player_ids.map(_ids), b=h.team_b_player_ids.map(_ids))
    h = h[(h.a.map(len) >= MIN_LINKED) & (h.b.map(len) >= MIN_LINKED) & (h.a.map(len) <= 5) & (h.b.map(len) <= 5)]
    t = pd.to_datetime(h.match_time)
    t = t.dt.tz_convert(None) if t.dt.tz is not None else t
    return pd.DataFrame({"t": t.to_numpy(), "a": h.a.to_numpy(), "b": h.b.to_numpy(), "y": h.team_a_won.astype(float).to_numpy(),
                         "known": pd.to_datetime(h.match_id.map(fin), unit="s").to_numpy(),
                         "twin_t": pd.to_datetime(h.match_id.map(twin)).to_numpy(),
                         "ta": h.ta.to_numpy(), "tb": h.tb.to_numpy(), "sa": h.sa.to_numpy(float), "sb": h.sb.to_numpy(float)})


def hltv_maps(h: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"t": h.t.to_numpy(), "a": h.team_a_player_ids.map(_ids).to_numpy(),
                         "b": h.team_b_player_ids.map(_ids).to_numpy(), "y": h.team_a_won.astype(float).to_numpy(),
                         "known": h.t.to_numpy(), "twin_t": pd.NaT,
                         "ta": h.team_a_id.to_numpy(), "tb": h.team_b_id.to_numpy(),
                         "sa": h.team_a_score.to_numpy(float), "sb": h.team_b_score.to_numpy(float)})


def fit_week(maps: pd.DataFrame, w: pd.Timestamp, faceit_flag: np.ndarray, P: Params = Params(), covs: list[dict] | None = None):
    win = (maps.known < w) & (maps.t >= w - pd.Timedelta(days=P.window_days)) & ~(maps.twin_t < w)
    m = maps[win.to_numpy()]
    fl = faceit_flag[win.to_numpy()]
    if len(m) < 200:
        return EMPTY
    age = (w - m.t).dt.total_seconds().to_numpy() / 86400.0
    wt = 0.5 ** (age / P.half_life_days) * np.where(fl, P.faceit_weight, 1.0)
    players = sorted({p for xs in m.a for p in xs} | {p for xs in m.b for p in xs})
    idx = {p: i for i, p in enumerate(players)}
    teams = sorted(set(m.ta) | set(m.tb)) if P.team_scale else []
    tidx = {t: len(players) + i for i, t in enumerate(teams)}
    rows, cols, vals = [], [], []
    for r, (a, b) in enumerate(zip(m.a, m.b)):
        for p in a:
            rows.append(r), cols.append(idx[p]), vals.append(1.0)
        for p in b:
            rows.append(r), cols.append(idx[p]), vals.append(-1.0)
    if P.team_scale:
        for r, (a, b) in enumerate(zip(m.ta, m.tb)):
            if a in tidx:
                rows.append(r), cols.append(tidx[a]), vals.append(P.team_scale)
            if b in tidx:
                rows.append(r), cols.append(tidx[b]), vals.append(-P.team_scale)
    ncov = len(covs) if (covs and P.cov_scale) else 0
    if ncov:
        base = len(players) + len(teams)
        for k, c in enumerate(covs):
            for r, (a, b) in enumerate(zip(m.a, m.b)):
                v = sum(c.get(p, 0.0) for p in a) - sum(c.get(p, 0.0) for p in b)
                if v:
                    rows.append(r), cols.append(base + k), vals.append(v * P.cov_scale)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(m), len(players) + len(teams) + ncov))
    y = m.y.to_numpy()
    if P.rounds:
        ok = (m.sa + m.sb > 0).to_numpy()
        X, wt, y, ra, rb = X[ok], wt[ok], y[ok], m.sa.to_numpy()[ok], m.sb.to_numpy()[ok]
        X = sp.vstack([X, X]).tocsr()
        y = np.r_[np.ones(len(ra)), np.zeros(len(ra))]
        wt = np.r_[wt * ra, wt * rb] / ROUND_NORM
    clf = LogisticRegression(C=P.c_ridge, fit_intercept=False, max_iter=1000)
    clf.fit(X, y, sample_weight=wt)
    coef = clf.coef_[0]
    return Skill(dict(zip(players, coef[:len(players)])), coef[len(players) + len(teams):], covs or [], P.cov_scale if ncov else 0.0)


def weekly_skills(data_dir, cutoff, hist: pd.DataFrame, weeks, use_faceit: bool, P: Params = Params(), covs=None):
    maps = hltv_maps(hist)
    flag = np.zeros(len(maps), bool)
    if use_faceit:
        fm = faceit_maps(data_dir, cutoff, hist)
        maps = pd.concat([maps, fm], ignore_index=True)
        flag = np.r_[flag, np.ones(len(fm), bool)]
    return {w: fit_week(maps, w, flag, P, covs[w] if covs else None) if w >= FIRST_FIT else EMPTY for w in weeks}
