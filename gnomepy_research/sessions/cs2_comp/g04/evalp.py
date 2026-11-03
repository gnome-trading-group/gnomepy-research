"""Own pool-vs-market diagnostic (the harness is rationed)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

D = "/Users/mprey/gnome-research-work/cs2/groundup/comp/data"


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(X, y):
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return w


def ll(p, y):
    p = np.clip(p, 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def truth_and_market():
    t = pd.read_parquet(f"{D}/targets_tune.parquet")
    h = pd.read_parquet(f"{D}/hltv/cs2_match_history.parquet", columns=["match_id", "team_a_won", "team_a_id"])
    h = h[h.match_id.isin(t.match_id)]
    s = h.groupby("match_id").team_a_won.agg(["sum", "count"])
    s["y"] = (s["sum"] * 2 > s["count"]).astype(float)
    p = pd.read_parquet(f"{D}/market/target_paths.parquet")
    p = p[(p.t <= p.kickoff) & (p.t >= p.kickoff - pd.Timedelta(minutes=30))].sort_values("t").groupby("match_id").p_a.last()
    t = t.merge(s[["y"]], left_on="match_id", right_index=True).merge(p.rename("m0"), left_on="match_id", right_index=True)
    t["month"] = pd.to_datetime(t.match_time).dt.strftime("%Y-%m")
    return t


def score(path, t=None):
    t = truth_and_market() if t is None else t
    q = pd.read_parquet(path).set_index("match_id").q
    d = t[t.match_id.isin(q.index)].copy()
    d["q"] = q.loc[d.match_id].values
    y, m = d.y.values, d.month.values
    X = np.column_stack([logit(d.m0), logit(d.q), np.ones(len(d))])
    out = np.empty(len(d))
    for mo in np.unique(m):
        tr = m != mo
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fit(X[tr], y[tr])))
    base = ll(d.m0.values, y)
    res = {"n": len(d), "market": base.mean(), "q-mkt": (ll(d.q.values, y) - base).mean(), "pool-mkt": (ll(out, y) - base).mean()}
    diff = ll(out, y) - base
    rng = np.random.default_rng(0)
    bs = [diff[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    res["ci"] = np.percentile(bs, [2.5, 97.5]).round(4).tolist()
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, score(p))
