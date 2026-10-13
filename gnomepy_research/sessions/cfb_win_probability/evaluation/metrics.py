"""Probability scoring: log loss, Brier, calibration error with documented binning, reliability
tables, and game-level paired bootstrap comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6
# Equal-width probability bins for reliability tables and ECE.
N_BINS = 10


def log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    out = df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean"))
    out.index = [f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}" for b in out.index]
    return out


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error: bin-size-weighted |mean_p - observed| over equal-width bins."""
    r = reliability(y, p, n_bins)
    return float((r["n"] * (r["mean_p"] - r["observed"]).abs()).sum() / r["n"].sum())


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b * logit(p) by Newton's method; b = 1, a = 0 is perfect."""
    p = np.clip(p, EPS, 1 - EPS)
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - mu)
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[1]), float(w[0])


def summarize(y: np.ndarray, p: np.ndarray) -> dict:
    slope, intercept = calibration_slope(y, p)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p).mean()),
        "brier": float(brier(y, p).mean()),
        "ece": ece(y, p),
        "accuracy": float(((p > 0.5) == (y == 1)).mean()),
        "calib_slope": slope,
        "calib_intercept": intercept,
    }


def paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Log-loss difference a - b (negative = a better), resampling games."""
    rng = np.random.default_rng(seed)
    d = log_loss(y, p_a) - log_loss(y, p_b)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return {"diff": float(d.mean()), "se": float(boots.std()), "lo95": float(np.quantile(boots, 0.025)),
            "hi95": float(np.quantile(boots, 0.975)), "p_a_worse": float((boots > 0).mean())}
