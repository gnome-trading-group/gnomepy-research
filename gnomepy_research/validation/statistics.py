"""Statistical significance tests for backtest results.

All functions operate on scalar summary values — no JVM or parquet I/O needed.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist(mu=0, sigma=1)
_EULER_MASCHERONI = 0.5772156649015328


def expected_max_sharpe(n_trials: int) -> float:
    """Expected maximum Sharpe ratio from n_trials experiments under H₀.

    Bailey & Lopez de Prado (2014) approximation:
        E[max SR] ≈ (1 − γ) × Φ⁻¹(1 − 1/N) + γ × Φ⁻¹(1 − 1/(N × e))
    """
    if n_trials <= 1:
        return 0.0
    z1 = _N.inv_cdf(1 - 1.0 / n_trials)
    z2 = _N.inv_cdf(1 - 1.0 / (n_trials * math.e))
    return (1 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2


def sharpe_standard_error(
    observed_sharpe: float,
    n_bars: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Standard error of the Sharpe ratio estimate.

    Lo (2002): σ_SR = √[(1 − SR × skew + SR²(κ − 1)/4) / (T − 1)]
    where κ is kurtosis (not excess kurtosis).
    """
    if n_bars <= 1:
        return float("inf")
    variance = (1 - observed_sharpe * skew + observed_sharpe**2 * (kurtosis - 1) / 4) / (n_bars - 1)
    return math.sqrt(max(variance, 1e-20))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_bars: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that observed Sharpe reflects genuine skill after N trials.

    Returns DSR ∈ [0, 1]. DSR > 0.95 means significant at the 5% level.
    Equivalent p-value = 1 − DSR.

    Bailey & Lopez de Prado (2014):
        DSR = Φ[(SR_N − SR*) / σ_SR]
    where SR* = expected max SR under H₀ over N trials.
    """
    sr_star = expected_max_sharpe(n_trials)
    sigma = sharpe_standard_error(observed_sharpe, n_bars, skew, kurtosis)
    if sigma <= 0 or not math.isfinite(sigma):
        return 1.0
    z = (observed_sharpe - sr_star) / sigma
    return _N.cdf(z)


def trials_correction(n_trials: int) -> float:
    """Expected max Sharpe from N trials — the hurdle the observed Sharpe must clear."""
    return expected_max_sharpe(n_trials)


def bootstrap_sharpe_ci(
    bar_returns: np.ndarray,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    block_size: int | None = None,
) -> tuple[float, float]:
    """Block-bootstrap confidence interval for the Sharpe ratio.

    Preserves autocorrelation by resampling blocks rather than individual bars.
    Returns (lower, upper) at the requested confidence level.
    """
    arr = np.asarray(bar_returns, dtype=float)
    n = len(arr)
    if n < 4:
        return (float("-inf"), float("inf"))

    if block_size is None:
        block_size = max(1, int(n**0.5))

    n_blocks = max(1, n // block_size)
    rng = np.random.default_rng(seed=42)

    bootstrap_sharpes: list[float] = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([arr[s : s + block_size] for s in starts])[:n]
        std = float(np.std(sample, ddof=1))
        if std > 0:
            bootstrap_sharpes.append(float(np.mean(sample)) / std)

    if not bootstrap_sharpes:
        return (float("-inf"), float("inf"))

    bootstrap_sharpes.sort()
    alpha = (1 - ci) / 2
    lo = int(alpha * len(bootstrap_sharpes))
    hi = int((1 - alpha) * len(bootstrap_sharpes))
    return (bootstrap_sharpes[lo], bootstrap_sharpes[min(hi, len(bootstrap_sharpes) - 1)])


def minimum_backtest_length(
    target_sharpe: float,
    ci: float = 0.95,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> int:
    """Minimum number of bars needed for target_sharpe to be statistically significant.

    Derived by solving σ_SR(T) × Φ⁻¹(ci) = target_sharpe for T.
    """
    if target_sharpe <= 0:
        return 0
    z = _N.inv_cdf(ci)
    numerator = 1 - target_sharpe * skew + target_sharpe**2 * (kurtosis - 1) / 4
    denominator = (target_sharpe / z) ** 2
    if denominator <= 0:
        return 0
    return max(1, int(math.ceil(numerator / denominator + 1)))
