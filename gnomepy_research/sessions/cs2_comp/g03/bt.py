"""Weekly ridge Bradley-Terry player skill on HLTV maps, optionally with FACEIT maps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from gnomepy_research.sessions.cs2_comp import bundle

ROUND_NORM = 8.0
MIN_LINKED = 3
FIRST_FIT = pd.Timestamp("2024-07-01")


@dataclass(frozen=True)
class Params:
    window_days: int = 900
    half_life_days: float = 800.0
    c_ridge: float = 0.2
    faceit_weight: float = 1.0
    rounds: bool = True
    team_scale: float = 0.0


def _ids(x):
    return [int(v) for v in x if v is not None and v == v]


def faceit_maps(data_dir, cutoff) -> pd.DataFrame:
    h = bundle.load(data_dir, "faceit/faceit_match_history", cutoff,
                    ["match_id", "map_name", "match_time", "team_a_player_ids", "team_b_player_ids", "team_a_won", "team_a_score", "team_b_score"])
    ex = bundle.load(data_dir, "faceit/faceit_exclusions", cutoff)
    link = bundle.load(data_dir, "faceit/faceit_hltv_series_link", cutoff, ["faceit_match_id", "hltv_match_id"])
    whole = set(ex[ex.map_name.isna()].match_id) | set(link.faceit_match_id)
    bad = set(zip(ex[ex.map_name.notna()].match_id, ex[ex.map_name.notna()].map_name))
    h = h[~h.match_id.isin(whole)]
    h = h[[(a, b) not in bad for a, b in zip(h.match_id, h.map_name)]]
    h = h.assign(ta=-1, tb=-1, sa=h.team_a_score, sb=h.team_b_score, a=h.team_a_player_ids.map(_ids), b=h.team_b_player_ids.map(_ids))
    h = h[(h.a.map(len) >= MIN_LINKED) & (h.b.map(len) >= MIN_LINKED) & (h.a.map(len) <= 5) & (h.b.map(len) <= 5)]
    t = pd.to_datetime(h.match_time)
    t = t.dt.tz_convert(None) if t.dt.tz is not None else t
    return pd.DataFrame({"t": t.to_numpy(), "a": h.a.to_numpy(), "b": h.b.to_numpy(), "y": h.team_a_won.astype(float).to_numpy(),
                         "ta": h.ta.to_numpy(), "tb": h.tb.to_numpy(), "sa": h.sa.to_numpy(float), "sb": h.sb.to_numpy(float)})


def hltv_maps(h: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"t": h.t.to_numpy(), "a": h.team_a_player_ids.map(_ids).to_numpy(),
                         "b": h.team_b_player_ids.map(_ids).to_numpy(), "y": h.team_a_won.astype(float).to_numpy(),
                         "ta": h.team_a_id.to_numpy(), "tb": h.team_b_id.to_numpy(),
                         "sa": h.team_a_score.to_numpy(float), "sb": h.team_b_score.to_numpy(float)})


def fit_week(maps: pd.DataFrame, w: pd.Timestamp, faceit_flag: np.ndarray, P: Params = Params()):
    win = (maps.t < w) & (maps.t >= w - pd.Timedelta(days=P.window_days))
    m = maps[win.to_numpy()]
    fl = faceit_flag[win.to_numpy()]
    if len(m) < 200:
        return {}
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
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(m), len(players) + len(teams)))
    y = m.y.to_numpy()
    if P.rounds:
        ok = (m.sa + m.sb > 0).to_numpy()
        X, wt, y, ra, rb = X[ok], wt[ok], y[ok], m.sa.to_numpy()[ok], m.sb.to_numpy()[ok]
        X = sp.vstack([X, X]).tocsr()
        y = np.r_[np.ones(len(ra)), np.zeros(len(ra))]
        wt = np.r_[wt * ra, wt * rb] / ROUND_NORM
    clf = LogisticRegression(C=P.c_ridge, fit_intercept=False, max_iter=300)
    clf.fit(X, y, sample_weight=wt)
    return dict(zip(players, clf.coef_[0][:len(players)]))


def weekly_skills(data_dir, cutoff, hist: pd.DataFrame, weeks, use_faceit: bool, P: Params = Params()):
    maps = hltv_maps(hist)
    flag = np.zeros(len(maps), bool)
    if use_faceit:
        fm = faceit_maps(data_dir, cutoff)
        maps = pd.concat([maps, fm], ignore_index=True)
        flag = np.r_[flag, np.ones(len(fm), bool)]
    return {w: fit_week(maps, w, flag, P) if w >= FIRST_FIT else {} for w in weeks}
