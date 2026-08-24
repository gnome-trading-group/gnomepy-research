"""
Offline HJB PDE solver for the Feil-Nendel prediction market maker model.

Implements the finite difference scheme from §4.2 of "Optimal Market Making
in Prediction Markets" (Feil & Nendel, 2026). Solves for the reduced value
function V(t,p,q) on a 3D grid using:
  - IMEX: implicit Euler for the diffusion term, explicit Hamiltonians
  - Centered differences in the price dimension (Neumann BCs via ghost cells)
  - scipy.linalg.solve_banded for the per-inventory-level tridiagonal systems

Run as a script to produce the default value function table:
    poetry run python -m gnomepy_research.solvers.prediction_market_hjb
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class HJBParams:
    # Volatility: σ(t,x) = σ0 + σ1*(t/T)^η + σ2/(1+x²), x = logit(p)
    sigma0: float = 0.6
    sigma1: float = 0.4
    sigma2: float = 0.2
    eta: float = 3.0
    # Activity: A(t,p) = (A0 + A1*(exp(ξt/T)-1)/(exp(ξ)-1)) * sqrt(p(1-p))
    A0: float = 100.0
    A1: float = 150.0
    xi: float = 2.0
    # Intensity shape: k(t) = k0 + k1*(t/T)^κ, boundary decay exponent ν
    nu: float = 1.0
    k0: float = 35.0
    k1: float = 50.0
    kappa: float = 1.5
    # Risk aversion
    gamma: float = 4e-3
    gamma_T: float = 1e-3
    # Inventory: levels {-Q, -Q+Δ, ..., Q}
    Q: float = 100.0
    delta_q: float = 10.0
    # Normalized time horizon (always 1.0; rescale real time outside the model)
    T: float = 1.0
    # Grid resolution (n_t≥500 required for temporal stability of IMEX scheme)
    n_t: int = 500
    n_p: int = 100
    p_min: float = 0.02
    p_max: float = 0.98
    # Unused — kept for backwards compatibility with saved .npz metadata
    max_fp_iters: int = 30
    fp_tol: float = 1e-8


# ---------------------------------------------------------------------------
# Intensity model
# ---------------------------------------------------------------------------

def _sigma(t: float, logit_p: np.ndarray, params: HJBParams) -> np.ndarray:
    tn = t / params.T
    return params.sigma0 + params.sigma1 * tn**params.eta + params.sigma2 / (1.0 + logit_p**2)


def _k(t: float, params: HJBParams) -> float:
    return params.k0 + params.k1 * (t / params.T) ** params.kappa


def _activity(t: float, p: np.ndarray, params: HJBParams) -> np.ndarray:
    tn = t / params.T
    ramp = (np.exp(params.xi * tn) - 1.0) / (np.exp(params.xi) - 1.0)
    return (params.A0 + params.A1 * ramp) * np.sqrt(np.clip(p * (1.0 - p), 0.0, None))


def _lambda_b(t: float, p: np.ndarray, pi: np.ndarray, params: HJBParams) -> np.ndarray:
    """Bid intensity Λ_b = A(t,p) · (2π/(π+p))^ν · exp(-k(t)·(p-π))."""
    k = _k(t, params)
    A = _activity(t, p, params)
    denom = np.clip(pi + p, 1e-12, None)
    ratio = np.clip(2.0 * pi / denom, 0.0, None)
    return A * ratio ** params.nu * np.exp(-k * (p - pi))


def _lambda_a(t: float, p: np.ndarray, pi: np.ndarray, params: HJBParams) -> np.ndarray:
    """Ask intensity Λ_a = A(t,p) · (2(1-π)/(2-π-p))^ν · exp(-k(t)·(π-p))."""
    k = _k(t, params)
    A = _activity(t, p, params)
    denom = np.clip(2.0 - pi - p, 1e-12, None)
    ratio = np.clip(2.0 * (1.0 - pi) / denom, 0.0, None)
    return A * ratio ** params.nu * np.exp(-k * (pi - p))


# ---------------------------------------------------------------------------
# Optimal quote computation via vectorized bisection (Proposition 3.2)
# ---------------------------------------------------------------------------

def _u_b(pi: np.ndarray, p: np.ndarray, k: float, nu: float) -> np.ndarray:
    """u_b(π) = p - π - Λ_b/(∂_π Λ_b). Strictly decreasing in π."""
    g = nu * p / np.clip(pi * (pi + p), 1e-12, None) + k
    return p - pi - 1.0 / np.clip(g, 1e-12, None)


def _u_a(pi: np.ndarray, p: np.ndarray, k: float, nu: float) -> np.ndarray:
    """u_a(π) = π - p - Λ_a/(|∂_π Λ_a|). Strictly increasing in π."""
    h = nu * (2.0 - p - pi) / np.clip((1.0 - pi) * (2.0 - pi - p), 1e-12, None) + k
    return pi - p - 1.0 / np.clip(h, 1e-12, None)


def _bisect_pi_b(z: np.ndarray, p: np.ndarray, k: float, nu: float,
                 n_iter: int = 60) -> np.ndarray:
    """Vectorized bisection for π_b* s.t. u_b(π) = z (u_b strictly decreasing).

    π_b is constrained to (0, p] — market maker never bids above mid-price.
    Above-mid bids create exponential blowup in the intensity and are
    economically irrational (buying above fair value).
    """
    eps = 1e-5
    lo = np.full_like(z, eps)
    hi = np.clip(p, eps, 1.0 - eps)  # upper bound is the mid-price

    u_lo = _u_b(hi, p, k, nu)  # u_b at π=p (most negative in feasible range)
    u_hi = _u_b(lo, p, k, nu)  # u_b at π≈0 (most positive ≈ p)

    in_range = (z > u_lo) & (z < u_hi)
    # z ≤ u_lo → bid at max (π = p); z ≥ u_hi → bid near 0 (very passive)
    pi = np.where(z <= u_lo, hi, np.where(z >= u_hi, lo, lo))

    if np.any(in_range):
        a_f, b_f = lo[in_range], hi[in_range]
        p_f, z_f = p[in_range], z[in_range]
        for _ in range(n_iter):
            m = (a_f + b_f) / 2
            # u_b decreasing: u_b(m) > z → π* > m → raise lower bound
            a_f = np.where(_u_b(m, p_f, k, nu) > z_f, m, a_f)
            b_f = np.where(_u_b(m, p_f, k, nu) > z_f, b_f, m)
        pi[in_range] = (a_f + b_f) / 2

    return pi


def _bisect_pi_a(z: np.ndarray, p: np.ndarray, k: float, nu: float,
                 n_iter: int = 60) -> np.ndarray:
    """Vectorized bisection for π_a* s.t. u_a(π) = z (u_a strictly increasing).

    π_a is constrained to [p, 1) — market maker never asks below mid-price.
    Below-mid asks create exponential blowup in the intensity and are
    economically irrational (selling below fair value).
    """
    eps = 1e-5
    lo = np.clip(p, eps, 1.0 - eps)  # lower bound is the mid-price
    hi = np.full_like(z, 1.0 - eps)

    u_lo = _u_a(lo, p, k, nu)  # u_a at π=p (most negative in feasible range)
    u_hi = _u_a(hi, p, k, nu)  # u_a at π≈1 (most positive)

    in_range = (z > u_lo) & (z < u_hi)
    # z ≥ u_hi → ask at max (π ≈ 1); z ≤ u_lo → ask at min (π = p, most aggressive)
    pi = np.where(z >= u_hi, hi, np.where(z <= u_lo, lo, lo))

    if np.any(in_range):
        a_f, b_f = lo[in_range], hi[in_range]
        p_f, z_f = p[in_range], z[in_range]
        for _ in range(n_iter):
            m = (a_f + b_f) / 2
            # u_a increasing: u_a(m) < z → π* > m → raise lower bound
            a_f = np.where(_u_a(m, p_f, k, nu) < z_f, m, a_f)
            b_f = np.where(_u_a(m, p_f, k, nu) < z_f, b_f, m)
        pi[in_range] = (a_f + b_f) / 2

    return pi


def _hamiltonian_b(t: float, p: np.ndarray, z: np.ndarray,
                   params: HJBParams) -> np.ndarray:
    """H^b(t,p;z) = Δ · Λ_b(π*) · max(p - π* - z, 0)."""
    k = _k(t, params)
    pi = _bisect_pi_b(z, p, k, params.nu)
    lam = _lambda_b(t, p, pi, params)
    return params.delta_q * lam * np.maximum(p - pi - z, 0.0)


def _hamiltonian_a(t: float, p: np.ndarray, z: np.ndarray,
                   params: HJBParams) -> np.ndarray:
    """H^a(t,p;z) = Δ · Λ_a(π*) · max(π* - p - z, 0)."""
    k = _k(t, params)
    pi = _bisect_pi_a(z, p, k, params.nu)
    lam = _lambda_a(t, p, pi, params)
    return params.delta_q * lam * np.maximum(pi - p - z, 0.0)


def optimal_bid(z: float, p: float, t: float, params: HJBParams) -> float:
    """Optimal bid price π_b* given marginal inventory cost z = (V(q) - V(q+Δ))/Δ."""
    k = _k(t, params)
    return float(_bisect_pi_b(np.array([z]), np.array([p]), k, params.nu)[0])


def optimal_ask(z: float, p: float, t: float, params: HJBParams) -> float:
    """Optimal ask price π_a* given marginal inventory cost z = (V(q) - V(q-Δ))/Δ."""
    k = _k(t, params)
    return float(_bisect_pi_a(np.array([z]), np.array([p]), k, params.nu)[0])


# ---------------------------------------------------------------------------
# HJB solver
# ---------------------------------------------------------------------------

def solve_hjb(params: HJBParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the HJB PDE and return the value function on a 3D grid.

    Uses an IMEX scheme: implicit Euler for the diffusion term (unconditionally
    stable) and explicit evaluation of the Hamiltonians from the previous time
    step (avoids checkerboard instability from Jacobi-style FP iteration on the
    inventory-coupled nonlinearity).

    Returns:
        V:        shape (n_t+1, n_p, n_q)  — V[tau_idx, p_idx, q_idx]
        tau_grid: shape (n_t+1,)  — τ=0 at resolution, τ=T at start
        p_grid:   shape (n_p,)
        q_levels: shape (n_q,)
    """
    q_levels = np.arange(-params.Q, params.Q + params.delta_q / 2, params.delta_q)
    n_q = len(q_levels)
    n_p = params.n_p

    p_grid = np.linspace(params.p_min, params.p_max, n_p)
    dp = p_grid[1] - p_grid[0]
    logit_p = np.log(p_grid / (1.0 - p_grid))

    tau_grid = np.linspace(0.0, params.T, params.n_t + 1)
    dtau = tau_grid[1] - tau_grid[0]

    V = np.zeros((params.n_t + 1, n_p, n_q))

    # Terminal condition at τ=0 (t=T): V(T,p,q) = Φ(p,q) = -γ_T q² p(1-p)
    for qi, q in enumerate(q_levels):
        V[0, :, qi] = -params.gamma_T * q**2 * p_grid * (1.0 - p_grid)

    for tau_idx in range(1, params.n_t + 1):
        t = params.T - tau_grid[tau_idx]  # real time t = T - τ
        V_prev = V[tau_idx - 1]           # shape (n_p, n_q)

        sigma = _sigma(t, logit_p, params)
        varsigma_sq = p_grid**2 * (1.0 - p_grid)**2 * sigma**2
        coeff = 0.5 * varsigma_sq / dp**2

        # Hamiltonians evaluated explicitly at V_prev (IMEX: no FP iteration).
        # This avoids the checkerboard instability that arises from Jacobi-style
        # updates on the inventory-coupled nonlinearity.
        Hb = np.zeros((n_p, n_q))
        Ha = np.zeros((n_p, n_q))
        for qi in range(n_q - 1):
            z_b = (V_prev[:, qi] - V_prev[:, qi + 1]) / params.delta_q
            Hb[:, qi] = _hamiltonian_b(t, p_grid, z_b, params)
        for qi in range(1, n_q):
            z_a = (V_prev[:, qi] - V_prev[:, qi - 1]) / params.delta_q
            Ha[:, qi] = _hamiltonian_a(t, p_grid, z_a, params)

        # RHS: shape (n_q, n_p)
        rhs = (V_prev.T / dtau
               + Hb.T + Ha.T
               - params.gamma * q_levels[:, None]**2 * varsigma_sq[None, :])

        # Tridiagonal coefficients (n_q, n_p)
        diag = 1.0 / dtau + 2.0 * coeff[None, :]
        sub = np.tile(-coeff, (n_q, 1))
        sup = np.tile(-coeff, (n_q, 1))

        # Neumann BCs via ghost-cell reflection:
        # j=0: V[-1]=V[1] → V[1] appears twice → double sup[:,0]
        # j=n_p-1: V[n_p]=V[-2] → V[-2] appears twice → double sub[:,-1]
        sup[:, 0] *= 2.0
        sub[:, -1] *= 2.0

        # diag/sub/sup are identical across q levels (coeff depends only on p).
        # solve_banded ab layout: ab[0,j]=a[j-1,j] (super), ab[2,j]=a[j+1,j] (sub)
        ab = np.empty((3, n_p))
        ab[0, 0] = 0.0
        ab[0, 1:] = sup[0, :-1]
        ab[1] = diag[0]
        ab[2, :-1] = sub[0, 1:]
        ab[2, -1] = 0.0
        # Pass rhs.T as multi-RHS: each column is one q level → result shape (n_p, n_q)
        V[tau_idx] = solve_banded((1, 1), ab, rhs.T, check_finite=False)

    return V, tau_grid, p_grid, q_levels


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_solution(V: np.ndarray, tau_grid: np.ndarray, p_grid: np.ndarray,
                  q_levels: np.ndarray, params: HJBParams, path: str) -> None:
    np.savez_compressed(path, V=V, tau_grid=tau_grid, p_grid=p_grid,
                        q_levels=q_levels, **asdict(params))


def load_solution(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HJBParams]:
    data = np.load(path)
    fields = HJBParams.__dataclass_fields__
    params = HJBParams(**{k: float(data[k]) for k in fields if k in data})
    return data['V'], data['tau_grid'], data['p_grid'], data['q_levels'], params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    params = HJBParams()
    n_q = int(2 * params.Q / params.delta_q + 1)
    print(f"Solving HJB: {params.n_t} time steps × {params.n_p} price pts × {n_q} inventory levels")
    t0 = time.time()
    V, tau_grid, p_grid, q_levels = solve_hjb(params)
    print(f"Done in {time.time() - t0:.1f}s  V.shape={V.shape}")
    out = Path(__file__).parent / 'value_function_default.npz'
    save_solution(V, tau_grid, p_grid, q_levels, params, str(out))
    print(f"Saved → {out}")
