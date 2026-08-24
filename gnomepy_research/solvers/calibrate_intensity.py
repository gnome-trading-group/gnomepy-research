"""
Calibrate the Feil-Nendel intensity model to historical Polymarket order flow.

The parametric intensity functions (A0, A1, ξ, ν, k0, k1, κ) control how
aggressively market participants take liquidity at different spreads. Starting
from the paper's defaults is reasonable, but calibrating to actual Polymarket
data produces more accurate fill probabilities and therefore better quotes.

Calibration approach:
  1. Estimate empirical fill rates at various (t_norm, p, spread) combinations
     from historical L2 snapshots: for each snapshot, check if a hypothetical
     limit order at a given spread would have been executed in the next N seconds.
  2. Fit (A0, A1, ξ, ν, k0, k1, κ) via Levenberg-Marquardt least squares to
     minimize the residual between Λ_b(t,p,π) and the empirical fill rate.

Input data format (DataFrame with columns):
    t_norm        float   normalized time since listing ∈ [0,1]
    mid           float   mid-price ∈ (0,1)
    spread_from_mid float  quoted spread from mid (positive = bid below mid)
    fill_count    int     observed fills within window_ns at this spread
    exposure_ns   int64   nanoseconds of market exposure at this spread level

Usage::

    from gnomepy_research.solvers.calibrate_intensity import calibrate
    from gnomepy_research.solvers.prediction_market_hjb import HJBParams

    result = calibrate(df)
    params = HJBParams(**result)
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from gnomepy_research.solvers.prediction_market_hjb import HJBParams


def _lambda_b_vectorized(t_norm: np.ndarray, p: np.ndarray, spread: np.ndarray,
                         A0: float, A1: float, xi: float, nu: float,
                         k0: float, k1: float, kappa: float) -> np.ndarray:
    """Bid intensity evaluated element-wise for calibration (all inputs arrays)."""
    pi = np.clip(p - spread, 1e-4, 1.0 - 1e-4)
    k = k0 + k1 * t_norm**kappa
    ramp = (np.exp(xi * t_norm) - 1.0) / (np.exp(xi) - 1.0 + 1e-12)
    A = (A0 + A1 * ramp) * np.sqrt(np.clip(p * (1.0 - p), 0.0, None))
    denom = np.clip(pi + p, 1e-12, None)
    ratio = np.clip(2.0 * pi / denom, 0.0, None)
    B = ratio**nu * np.exp(-k * spread)
    return A * B


def _residuals(x: np.ndarray, t_norm: np.ndarray, p: np.ndarray,
               spread: np.ndarray, empirical_rate: np.ndarray) -> np.ndarray:
    A0, A1, xi, nu, k0, k1, kappa = x
    predicted = _lambda_b_vectorized(t_norm, p, spread, A0, A1, xi, nu, k0, k1, kappa)
    return predicted - empirical_rate


def calibrate(df, params: HJBParams | None = None, max_iter: int = 500) -> dict:
    """Fit intensity parameters to empirical fill rate data.

    Args:
        df: DataFrame with columns [t_norm, mid, spread_from_mid, fill_count,
            exposure_ns].
        params: starting-point HJBParams (defaults to paper values).
        max_iter: Levenberg-Marquardt iteration limit.

    Returns:
        Dict of calibrated intensity parameters — pass as **kwargs to HJBParams.
    """
    if params is None:
        params = HJBParams()

    t_norm = df['t_norm'].to_numpy(dtype=float)
    p = df['mid'].to_numpy(dtype=float)
    spread = df['spread_from_mid'].to_numpy(dtype=float)
    exposure_s = df['exposure_ns'].to_numpy(dtype=float) / 1e9
    empirical_rate = df['fill_count'].to_numpy(dtype=float) / np.clip(exposure_s, 1e-9, None)

    mask = (
        (t_norm > 0) & (t_norm < 1)
        & (p > 0.05) & (p < 0.95)
        & (spread > 0) & (spread < 0.5)
        & (exposure_s > 0.1)
    )
    t_norm, p, spread, empirical_rate = (
        t_norm[mask], p[mask], spread[mask], empirical_rate[mask]
    )

    x = np.array([params.A0, params.A1, params.xi, params.nu,
                  params.k0, params.k1, params.kappa])
    bounds_lo = np.array([1.0,  0.0,  0.1, 0.1, 1.0,  0.0, 0.5])
    bounds_hi = np.array([500., 500., 5.0, 5.0, 200., 200., 5.0])
    x = np.clip(x, bounds_lo, bounds_hi)

    lam = 1e-3
    eps = 1e-5

    for _ in range(max_iter):
        r = _residuals(x, t_norm, p, spread, empirical_rate)
        loss = float(np.dot(r, r))

        J = np.zeros((len(r), len(x)))
        for j in range(len(x)):
            xp = x.copy(); xp[j] += eps
            xm = x.copy(); xm[j] -= eps
            J[:, j] = (
                _residuals(xp, t_norm, p, spread, empirical_rate)
                - _residuals(xm, t_norm, p, spread, empirical_rate)
            ) / (2 * eps)

        JtJ = J.T @ J
        Jtr = J.T @ r
        diag_reg = lam * (np.diag(JtJ) + 1e-8)
        step = np.linalg.solve(JtJ + np.diag(diag_reg), -Jtr)

        x_new = np.clip(x + step, bounds_lo, bounds_hi)
        r_new = _residuals(x_new, t_norm, p, spread, empirical_rate)
        loss_new = float(np.dot(r_new, r_new))

        if loss_new < loss:
            x = x_new
            lam = max(lam / 3, 1e-7)
            if np.linalg.norm(step) < 1e-9:
                break
        else:
            lam = min(lam * 3, 1e3)

    return {
        'A0': float(x[0]),
        'A1': float(x[1]),
        'xi': float(x[2]),
        'nu': float(x[3]),
        'k0': float(x[4]),
        'k1': float(x[5]),
        'kappa': float(x[6]),
    }


def print_params(params: HJBParams) -> None:
    for k, v in asdict(params).items():
        print(f"  {k}: {v}")
