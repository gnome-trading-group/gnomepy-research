"""
Does the pre-kickoff price path know more than the kickoff price? A development check, not part of the predictions.

    python -m gnomepy_research.sessions.cs2_comp.g06.market_paths --data <bundle> --e12 <e12_hist parquet>

For every bo3 series with a market path it prints month-held-out log loss differences against the kickoff price alone
(m0, the last print within 30 minutes of kickoff) of logistic pools that add the path's time-weighted mean logit over
the last two hours, with and without e12's price. Late moves overshoot and partly revert, so the two-hour average beats
m0 on its own; once e12's price is pooled the average adds nothing, because the moves that revert are the ones away
from the skill price.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.g06.history import logit, market_series_targets, path_features


def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = expit(X @ w)
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return w


def held_out_loss(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = np.column_stack([d[c].to_numpy(float) for c in cols] + [np.ones(len(d))])
    y = d.y.to_numpy(float)
    months = d.kickoff.dt.strftime("%Y-%m").to_numpy()
    p = np.empty(len(d))
    for m in np.unique(months):
        tr = months != m
        p[~tr] = expit(X[~tr] @ _fit(X[tr], y[tr]))
    p = np.clip(p, 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--e12", required=True, help="output of g06.e12_hist")
    a = ap.parse_args()
    s = market_series_targets(a.data, pd.Timestamp("2100-01-01"))
    paths = pd.read_parquet(f"{a.data}/market/series_paths.parquet")
    d = s.merge(path_features(paths), on="match_id").merge(pd.read_parquet(a.e12)[["match_id", "e12f"]], on="match_id")
    d = d.assign(z0=logit(d.m0.fillna(d.p_last)), ze=logit(d.e12f))
    base = held_out_loss(d, ["z0"])
    for cols in (["tw_120"], ["z0", "tw_120"], ["z0", "ze"], ["z0", "ze", "tw_120"]):
        diff = held_out_loss(d, cols) - base
        print(f"{' + '.join(cols):24s} {diff.mean():+.4f} (se {diff.std() / np.sqrt(len(diff)):.4f}, n={len(diff)})")


if __name__ == "__main__":
    main()
