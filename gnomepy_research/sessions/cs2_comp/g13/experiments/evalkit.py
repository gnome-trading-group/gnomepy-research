"""The harness pool, log loss and bootstrap interval, so dev scores here mean what the harness means (no private data)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


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
    X = np.column_stack([logit(m0), logit(q)] + ([] if extra is None else [logit(e) for e in extra]) + [np.ones(len(y))])
    out = np.empty(len(y))
    for m in np.unique(months):
        tr = months != m
        out[~tr] = 1 / (1 + np.exp(-X[~tr] @ fit(X[tr], y[tr])))
    return out


def ci(x, n_boot=2000):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "n": int(len(x))}


def fmt(c):
    star = "*" if c["lo"] > 0 or c["hi"] < 0 else " "
    return f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}]{star} n={c['n']}"
