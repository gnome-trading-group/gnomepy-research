from __future__ import annotations

from gnomepy_research.signals.operations.base import Operation, apply


class KalmanOperation(Operation):
    """1D scalar Kalman filter.

    Args:
        Q: Process noise — how much the true value can change between updates.
        R: Measurement noise — how noisy the raw signal output is.
        adaptive_alpha: If set, R is adapted online from innovations.
        r_floor: Minimum R when using adaptive mode.
        warmup: Minimum updates before is_ready() returns True.
    """

    def __init__(
        self,
        Q: float = 1e-4,
        R: float = 1e-2,
        adaptive_alpha: float | None = None,
        r_floor: float = 1e-10,
        warmup: int = 1,
    ):
        self.Q = Q
        self.R_init = R
        self.adaptive_alpha = adaptive_alpha
        self.r_floor = r_floor
        self.warmup = warmup

        self._x = 0.0
        self._P = 0.0
        self._R = R
        self._count = 0

    def update(self, value: float) -> None:
        if self._count == 0:
            self._x = value
            self._P = self.R_init
            self._count += 1
            return

        x_pred = self._x
        p_pred = self._P + self.Q

        innovation = value - x_pred
        S = p_pred + self._R

        if self.adaptive_alpha is not None:
            r_candidate = innovation * innovation - p_pred
            self._R = (
                self.adaptive_alpha * self._R
                + (1.0 - self.adaptive_alpha) * max(r_candidate, self.r_floor)
            )

        K = p_pred / S if S > 0 else 0.0
        self._x = x_pred + K * innovation
        self._P = (1.0 - K) * p_pred

        if self._P < 0:
            self._P = self.r_floor

        self._count += 1

    def value(self) -> float:
        return self._x

    def is_ready(self) -> bool:
        return self._count >= self.warmup

    def reset(self) -> None:
        self._x = 0.0
        self._P = 0.0
        self._R = self.R_init
        self._count = 0


def Kalman(
    signal,
    Q: float = 1e-4,
    R: float = 1e-2,
    adaptive_alpha: float | None = None,
    r_floor: float = 1e-10,
    warmup: int = 1,
):
    """Apply Kalman filtering to any signal, preserving its type."""
    return apply(
        KalmanOperation(
            Q=Q, R=R, adaptive_alpha=adaptive_alpha,
            r_floor=r_floor, warmup=warmup,
        ),
        signal,
    )
