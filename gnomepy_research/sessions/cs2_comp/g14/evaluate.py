"""
g14's own dev scorer, mirroring harness/score.py with outcomes read from the bundle's HLTV history and m0 from the
target paths: pool(market, q) - market, q - market, and with an e12 reference file, stack - baseline pool.

    python -m gnomepy_research.sessions.cs2_comp.g14.evaluate <predictions.parquet> [--data <bundle>] [--e12 <parquet>]
        [--col q]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.g14.data import logit, path_features
from gnomepy_research.sessions.cs2_comp.bundle import target_paths

DATA = Path("/Users/mprey/gnome-research-work/cs2/groundup/comp/data")


def ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(X, y):
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return w


def pool(m0, q, y, months, extra=None):
    X = np.column_stack([logit(m0), logit(q)] + ([] if extra is None else [logit(extra)]) + [np.ones(len(y))])
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fit(X[tr], y[tr])))
    return out


def ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "n": int(len(x))}


def fmt(c):
    star = "*" if c["lo"] > 0 or c["hi"] < 0 else " "
    return f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}]{star} n={c['n']}"


def truth(data_dir: Path) -> pd.DataFrame:
    t = pd.read_parquet(data_dir / "targets_tune.parquet")
    h = pd.read_parquet(data_dir / "hltv/cs2_match_history.parquet")
    h = h[~showmatch_mask(h)]
    s = h.groupby("match_id").agg(wins=("team_a_won", "sum"), maps=("team_a_won", "size"))
    s = s[s.wins * 2 != s.maps]
    s["y"] = (s.wins * 2 > s.maps).astype(int)
    mk = path_features(target_paths(data_dir, t.match_id))
    d = t[["match_id", "match_date", "is_lan"]].merge(s[["y"]], left_on="match_id", right_index=True).merge(mk[["match_id", "m0"]], on="match_id", how="left")
    d["m0"] = d.m0.fillna(0.5)
    d["month"] = pd.to_datetime(d.match_date).dt.strftime("%Y-%m")
    return d


def score(pred: pd.DataFrame, data_dir: Path = DATA, e12: pd.DataFrame | None = None, col: str = "q", quiet: bool = False) -> dict:
    d = truth(data_dir).merge(pred[["match_id", col]].rename(columns={col: "q"}), on="match_id", how="left")
    missing = int(d.q.isna().sum())
    d["q"] = d.q.fillna(d.m0)
    y, months = d.y.to_numpy(float), d.month.to_numpy()
    d["l_m"] = ll(d.m0, y)
    d["l_q"] = ll(d.q, y)
    d["l_pool"] = ll(pool(d.m0, d.q, y, months), y)
    out = {"n": len(d), "missing": missing, "market_ll": float(d.l_m.mean()), "q_ll": float(d.l_q.mean()), "pool_ll": float(d.l_pool.mean()),
           "q_minus_market": ci(d.l_q - d.l_m), "pool_minus_market": ci(d.l_pool - d.l_m)}
    if e12 is not None:
        d = d.merge(e12[["match_id", "q"]].rename(columns={"q": "q_e12"}), on="match_id", how="left")
        d["q_e12"] = d.q_e12.fillna(d.m0)
        d["l_base_pool"] = ll(pool(d.m0, d.q_e12, y, months), y)
        d["l_stack"] = ll(pool(d.m0, d.q, y, months, d.q_e12), y)
        out["pool_minus_baseline_pool"] = ci(d.l_pool - d.l_base_pool)
        out["stack_minus_baseline_pool"] = ci(d.l_stack - d.l_base_pool)
        out["baseline_pool_minus_market"] = ci(d.l_base_pool - d.l_m)
    by_month = d.groupby("month").apply(lambda g: pd.Series({"market": g.l_m.mean(), "q": g.l_q.mean(), "pool": g.l_pool.mean(), "n": len(g)}))
    out["by_month"] = by_month
    if not quiet:
        print(f"n={out['n']} missing={missing} market {out['market_ll']:.4f} q {out['q_ll']:.4f} pool {out['pool_ll']:.4f}")
        print(f"q - market      {fmt(out['q_minus_market'])}")
        print(f"pool - market   {fmt(out['pool_minus_market'])}")
        if e12 is not None:
            print(f"pool - basepool {fmt(out['pool_minus_baseline_pool'])}")
            print(f"stack - basepool{fmt(out['stack_minus_baseline_pool'])}")
            print(f"basepool - mkt  {fmt(out['baseline_pool_minus_market'])}")
        print(by_month.round(4).to_string())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--e12")
    ap.add_argument("--col", default="q")
    a = ap.parse_args()
    e12 = pd.read_parquet(a.e12) if a.e12 else None
    score(pd.read_parquet(a.predictions), Path(a.data), e12, a.col)


if __name__ == "__main__":
    main()
