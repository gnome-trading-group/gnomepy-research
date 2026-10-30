"""
Development walk-forward for e12 and the fit of predict.py's logistic layer.

For each weekly cutoff from --first to --last, the same rows predict.py builds, for the HLTV bo3 series of the
following week, with their outcome. Then the layer's weights for both variants (with and without FACEIT), their
month-held-out log loss, and the FACEIT variant's gain split by how many of the less-linked five's players have a
FACEIT id.

    python -m gnomepy_research.sessions.cs2_comp.e12.dev --data <bundle> --out <dir> --first 2025-10-06 --last 2026-03-23

Only cutoffs before the first tune week (2026-03-30) are used for the weights in predict.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.e12.predict import design, rows


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1]), X.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def held_out(X: np.ndarray, y: np.ndarray, months: np.ndarray) -> np.ndarray:
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fit_logistic(X[tr], y[tr])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cutoffs = list(pd.date_range(a.first, a.last, freq="7D"))
    inputs = Inputs(a.data, cutoffs[-1] + pd.Timedelta(days=7), faceit=True)
    s = inputs.series[inputs.series.bo_type == 3]
    targets = pd.concat([s[(s.time >= c) & (s.time < c + pd.Timedelta(days=7))].assign(week_cutoff=c) for c in cutoffs],
                        ignore_index=True)[["match_id", "team_a", "team_b", "week_cutoff", "y"]]
    h, f = rows(inputs, targets, True, a.data, verbose=True)
    h.to_parquet(out / "dev_hltv.parquet", index=False)
    f.to_parquet(out / "dev_faceit.parquet", index=False)
    y = h.y.to_numpy(float)
    months = pd.to_datetime(h.week_cutoff).dt.to_period("M").astype(str).to_numpy()
    result, losses = {}, {}
    for name, frame in (("hltv", design(h, None)), ("faceit", design(h, f))):
        X = frame.to_numpy()
        losses[name] = log_loss(held_out(X, y, months), y)
        result[name] = {"weights": dict(zip(frame.columns, map(float, fit_logistic(X, y)))),
                        "held_out_log_loss": float(losses[name].mean())}
    diff = losses["faceit"] - losses["hltv"]
    boot = diff[np.random.default_rng(0).integers(0, len(diff), (2000, len(diff)))].mean(1)
    result["faceit_minus_hltv"] = {"mean": float(diff.mean()), "lo": float(np.percentile(boot, 2.5)),
                                   "hi": float(np.percentile(boot, 97.5)), "n": int(len(diff))}
    cov = np.minimum(f.linked_a, f.linked_b).to_numpy()
    result["by_linked"] = {k: {"mean": float(diff[m].mean()), "n": int(m.sum())} for k, m in
                           (("0", cov == 0), ("1-3", (cov >= 1) & (cov <= 3)), ("4-5", cov >= 4)) if m.any()}
    (out / "layer.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
