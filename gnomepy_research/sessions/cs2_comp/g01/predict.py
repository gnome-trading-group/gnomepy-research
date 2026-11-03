"""g01: Elo ratings (team + player, round margin, roster-change reset) calibrated by logistic regression, then pooled
with the kickoff market price through a ridge-to-identity logistic fitted on past series that have a market."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.g01 import ratings, skill

STAGE1_FROM = pd.Timestamp("2025-10-06")
LAM = 5.0
ELO_PARAMS = {"k_player": 40.0, "margin": 0.9}
EPS = 1e-4


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def kickoff_prices(paths: pd.DataFrame) -> pd.Series:
    d = paths.copy()
    d["t"] = pd.to_datetime(d.t, utc=True).dt.tz_convert(None)
    d["kickoff"] = pd.to_datetime(d.kickoff, utc=True).dt.tz_convert(None)
    d = d[d.t <= d.kickoff].sort_values("t")
    last = d.groupby("match_id").tail(1).set_index("match_id")
    fresh = (last.kickoff - last.t) <= pd.Timedelta(minutes=30)
    return last.p_a.where(fresh)


_FITS: dict[pd.Timestamp, skill.SkillFit] = {}


def skill_features(maps: pd.DataFrame, stats: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(queries), 3))
    for week, idx in queries.groupby("week").indices.items():
        if week not in _FITS:
            _FITS[week] = skill.SkillFit(maps, week, stats)
        fit = _FITS[week]
        for i in idx:
            out[i] = fit.diff(list(queries.pa.iloc[i]), list(queries.pb.iloc[i]))
    return out


def stage1_matrix(f: np.ndarray, sk: np.ndarray, lan: np.ndarray) -> np.ndarray:
    return np.column_stack([sk[:, :2], f[:, :2], lan])


def fit_pool(lm: np.ndarray, lp: np.ndarray, y: np.ndarray, lam: float = LAM) -> np.ndarray:
    X = np.column_stack([lm, lp, np.ones_like(lm)])
    prior = np.array([1.0, 0.0, 0.0])
    pen = np.array([1.0, 1.0, 0.0]) * lam

    def nll(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(pen * (b - prior) ** 2)

    def grad(b):
        return X.T @ (1 / (1 + np.exp(-(X @ b))) - y) + pen * (b - prior)

    return minimize(nll, prior, jac=grad, method="L-BFGS-B").x


def predict_week(data_dir: str, week: pd.Timestamp, targets: pd.DataFrame) -> pd.DataFrame:
    hist = bundle.load(data_dir, "hltv/cs2_match_history", before=week)
    maps = ratings.prepare_maps(hist)
    ser = ratings.series_table(maps)
    ser = ser[(ser.bo == 3) & ser.y.notna() & (ser.t >= STAGE1_FROM)].copy()
    ser["week"] = ratings.monday(ser.t)
    tq = pd.DataFrame({"week": week, "team_a_id": targets.team_a_id.values, "team_b_id": targets.team_b_id.values,
                       "pa": targets.team_a_lineup_ids.map(list).values, "pb": targets.team_b_lineup_ids.map(list).values,
                       "lan": targets.is_lan.fillna(0).astype(float).values})
    tq["pa"] = [a if len(a) == 5 else [] for a in tq.pa]
    tq["pb"] = [b if len(b) == 5 else [] for b in tq.pb]
    hq = ser[["week", "team_a_id", "team_b_id", "pa", "pb"]].copy()
    hq["lan"] = ser.lan.fillna(0).astype(float).values
    allq = pd.concat([hq, tq], ignore_index=True)
    feats = ratings.snapshot_features(maps, allq, ELO_PARAMS)
    stats = bundle.load(data_dir, "hltv/cs2_player_map_stats", before=week,
                        columns=["match_date", "mapstatsid", "is_team_a", "player_id", "side", "rating"])
    stats = stats[stats.side == "all"]
    sk = skill_features(maps, stats, allq[["week", "pa", "pb"]].reset_index(drop=True))
    X = stage1_matrix(feats, sk, allq.lan.values)
    nh = len(hq)
    Xh, Xt = X[:nh], X[nh:]

    train = (ser.t >= STAGE1_FROM).values
    clf = LogisticRegression(C=1.0, max_iter=500).fit(np.nan_to_num(Xh[train]), ser.y.values[train])
    p_h = clf.predict_proba(np.nan_to_num(Xh))[:, 1]
    p_t = clf.predict_proba(np.nan_to_num(Xt))[:, 1]

    m_hist = kickoff_prices(bundle.load(data_dir, "market/series_paths", before=week))
    m_tgt = kickoff_prices(bundle.target_paths(data_dir, targets.match_id))
    m0_h = ser.match_id.map(m_hist).values.astype(float)
    ok = ~np.isnan(m0_h)
    coef = fit_pool(logit(m0_h[ok]), logit(p_h[ok]), ser.y.values[ok])
    m0_t = targets.match_id.map(m_tgt).values.astype(float)
    q = p_t.copy()
    have = ~np.isnan(m0_t)
    z = coef[0] * logit(m0_t[have]) + coef[1] * logit(p_t[have]) + coef[2]
    q[have] = 1 / (1 + np.exp(-z))
    return pd.DataFrame({"match_id": targets.match_id.values, "q": np.clip(q, EPS, 1 - EPS), "fit_cutoff": week,
                         "p_rating": p_t, "m0": m0_t})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", type=int, nargs="*")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    t = pd.read_parquet(a.targets)
    t["week_cutoff"] = pd.to_datetime(t.week_cutoff)
    weeks = sorted(t.week_cutoff.unique())
    if a.weeks:
        weeks = [weeks[i] for i in a.weeks]
    out = [predict_week(a.data, pd.Timestamp(w), t[t.week_cutoff == w]) for w in weeks]
    res = pd.concat(out, ignore_index=True)
    if not a.debug:
        res = res[["match_id", "q", "fit_cutoff"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
