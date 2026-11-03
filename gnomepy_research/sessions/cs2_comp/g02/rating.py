"""Player-level ridge-logistic ratings over HLTV and FACEIT rounds, priced through a bo3 series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp import bundle

UNRANKED = 150.0
ROUND_INFLATION = 8.0
FACEIT_OFFSET = 10**9


def _ids(x) -> list[int]:
    return [] if x is None or isinstance(x, float) else [int(v) for v in x]


def rank_feature(rank: pd.Series) -> pd.Series:
    return -np.log(rank.fillna(UNRANKED).clip(upper=UNRANKED).astype(float))


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def hltv_maps(data_dir, cutoff: pd.Timestamp, window_days: int) -> pd.DataFrame:
    h = bundle.load(data_dir, "hltv/cs2_match_history", cutoff)
    h = h[~showmatch_mask(h)].dropna(subset=["team_a_won"])
    day = _naive(h.match_time.fillna(h.match_date))
    keep = day >= cutoff - pd.Timedelta(days=window_days)
    h, day = h[keep], day[keep]
    return pd.DataFrame({"a": h.team_a_player_ids.map(_ids), "b": h.team_b_player_ids.map(_ids), "day": day,
                         "ra": h.team_a_score.astype(float), "rb": h.team_b_score.astype(float), "src": 0,
                         "ka": rank_feature(h.team_a_rank), "kb": rank_feature(h.team_b_rank)})


def faceit_maps(data_dir, cutoff: pd.Timestamp, window_days: int) -> pd.DataFrame:
    f = bundle.load(data_dir, "faceit/faceit_match_history", cutoff)
    link = bundle.load(data_dir, "faceit/faceit_hltv_series_link", cutoff)
    excl = bundle.load(data_dir, "faceit/faceit_exclusions", cutoff)
    f = f[~f.match_id.isin(link.faceit_match_id) & ~f.match_id.isin(excl[excl.map_name.isna()].match_id)]
    both = excl.dropna(subset=["map_name"])
    f = f[~f.set_index(["match_id", "map_name"]).index.isin(both.set_index(["match_id", "map_name"]).index)]
    day = _naive(f.match_time.fillna(f.match_date))
    keep = (day >= cutoff - pd.Timedelta(days=window_days)) & f.team_a_score.notna() & f.team_b_score.notna()
    f, day = f[keep], day[keep]
    m = bundle.load(data_dir, "faceit/hltv_faceit_map")
    to_hltv = dict(zip(m.faceit_id, m.hltv_id.astype(int)))
    uuids: dict[str, int] = {}

    def key(u: str) -> int:
        return to_hltv.get(u) or uuids.setdefault(u, FACEIT_OFFSET + len(uuids))

    def side(col) -> list[list[int]]:
        return [[key(u) for u in us] if us is not None and len(us) == 5 else [] for us in col]

    a, b = side(f.team_a_faceit_ids), side(f.team_b_faceit_ids)
    out = pd.DataFrame({"a": a, "b": b, "day": day.to_numpy(), "ra": f.team_a_score.astype(float).to_numpy(),
                        "rb": f.team_b_score.astype(float).to_numpy(), "src": 1, "ka": 0.0, "kb": 0.0})
    return out


def fit_ratings(maps: pd.DataFrame, cutoff: pd.Timestamp, c: float, half_life: float, faceit_weight: float,
                rank_scale: float = 1.0, skill: dict[int, float] | None = None):
    """Returns (player ratings, per-rank-point coefficient)."""
    maps = maps[(maps.a.map(len) == 5) & (maps.b.map(len) == 5)]
    players = sorted({p for col in (maps.a, maps.b) for ps in col for p in ps})
    idx = {p: i for i, p in enumerate(players)}
    rows, cols, vals = [], [], []
    for r, (a, b) in enumerate(zip(maps.a, maps.b)):
        for p in a:
            rows.append(r), cols.append(idx[p]), vals.append(1.0)
        for p in b:
            rows.append(r), cols.append(idx[p]), vals.append(-1.0)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(maps), len(players)))
    kd = sp.csr_matrix((maps.ka - maps.kb).to_numpy()[:, None] * rank_scale)
    extra = [kd]
    if skill is not None:
        gap = np.array([sum(skill.get(p, 0.0) for p in a) - sum(skill.get(p, 0.0) for p in b) for a, b in zip(maps.a, maps.b)])
        extra.append(sp.csr_matrix(gap[:, None] * rank_scale))
    X = sp.hstack([X] + extra).tocsr()
    age = (cutoff - maps.day).dt.total_seconds().to_numpy() / 86400
    decay = 0.5 ** (age / half_life) * np.where(maps.src.to_numpy() == 1, faceit_weight, 1.0)
    Xs = sp.vstack([X, X])
    ys = np.concatenate([np.ones(len(maps)), np.zeros(len(maps))])
    ws = np.concatenate([maps.ra.to_numpy(), maps.rb.to_numpy()]) * np.concatenate([decay, decay]) / ROUND_INFLATION
    clf = LogisticRegression(C=c, fit_intercept=False, solver="lbfgs", max_iter=300)
    clf.fit(Xs, ys, sample_weight=ws)
    coef = clf.coef_[0]
    if skill is not None:
        return dict(zip(players, coef[:-2])), coef[-2] * rank_scale, coef[-1] * rank_scale
    return dict(zip(players, coef[:-1])), coef[-1] * rank_scale


def bo3(p: np.ndarray) -> np.ndarray:
    return p * p * (3 - 2 * p)


def price(ratings: dict[int, float], rank_coef: float, a: list[int], b: list[int], rank_diff: float,
          scale: float, skill_term: float = 0.0) -> float:
    d = sum(ratings.get(p, 0.0) for p in a) - sum(ratings.get(p, 0.0) for p in b) + rank_coef * rank_diff + skill_term
    return float(bo3(1 / (1 + np.exp(-scale * d))))


def current_ranks(data_dir, cutoff: pd.Timestamp) -> dict[int, float]:
    r = bundle.load(data_dir, "hltv/cs2_team_rankings", cutoff)
    r = r[r.date == r.date.max()]
    return dict(zip(r.team_id.astype(int), r["rank"].astype(float)))


def rank_diff(ranks: dict[int, float], ta: int, tb: int) -> float:
    f = lambda t: -np.log(min(ranks.get(t, UNRANKED), UNRANKED))
    return f(ta) - f(tb)
