"""The residual layer: a penalised logistic regression of the series result on the kickoff price and a few corrections
to it, refitted each week on every past market-joined series.

    logit P(A) = a * z0 + b * (zbar - z0) + c * (z_e12 - z0) + sum_k d_k * (z_e12 - z0) * ctx_k + intercept

z0 is the kickoff price, zbar the price averaged over the last two hours (the last prints before kickoff are noisy
and partly revert), z_e12 the skill price. The context columns let the weight on the skill price's disagreement with
the market depend on how informed the market is likely to be. A series without a kickoff price is priced from the
skill price alone, by a second fit on the same training rows."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CLIP = 0.005


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class Spec:
    twap: str | None = "zbar_120m"
    skill: str | None = "q_e12"
    context: tuple[str, ...] = ()
    l2: float = 1.0
    l2_z0: float = 0.0
    half_life_days: float | None = None


def context_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Candidate context columns, each centred roughly on zero over a typical week."""
    thin = d.h_thin_a.fillna(5) + d.h_thin_b.fillna(5)
    lr = np.log(np.minimum(d.h_rank_a.fillna(300), d.h_rank_b.fillna(300)))
    return pd.DataFrame({
        "liq": np.log1p(d.n_changes_6h.fillna(0)) - 3.0,
        "stale": np.log1p(d.stale_min.fillna(60)) - 1.5,
        "young": (d.open_age_h.fillna(0) < 12).astype(float) - 0.5,
        "thin": np.minimum(thin, 4) / 4.0,
        "new": (d.h_new_a.fillna(0) + d.h_new_b.fillna(0)) / 2.0,
        "rank": lr - 4.0,
        "faceit": np.minimum(d.f_linked_a.fillna(0), d.f_linked_b.fillna(0)) / 5.0 - 0.5,
        "idle": (np.log1p(d.h_idle_a.fillna(30)) + np.log1p(d.h_idle_b.fillna(30))) / 2 - 2.0,
    }, index=d.index)


def design(d: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Columns and per-column penalty weights. Rows without a kickoff price use the last print as z0."""
    z0 = logit(d.m0.fillna(d.p_last))
    cols, pen, names = [z0], [spec.l2_z0], ["z0"]
    if spec.twap:
        cols.append(d[spec.twap].fillna(pd.Series(z0, index=d.index)).to_numpy() - z0)
        pen.append(spec.l2 * 0.1)
        names.append("twap")
    if spec.skill:
        de = logit(d[spec.skill]) - z0
        cols.append(de)
        pen.append(spec.l2 * 0.1)
        names.append("skill")
        ctx = context_columns(d)
        for c in spec.context:
            cols.append(de * ctx[c].to_numpy())
            pen.append(spec.l2)
            names.append(f"skill*{c}")
    cols.append(np.ones(len(d)))
    pen.append(0.0)
    names.append("1")
    return np.column_stack(cols), np.array(pen), names


def fit_logistic(X: np.ndarray, y: np.ndarray, pen: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(y)) if w is None else w
    beta = np.zeros(X.shape[1])
    P = np.diag(pen + 1e-8)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ beta))
        step = np.linalg.solve(X.T @ (X * (w * p * (1 - p))[:, None]) + P, X.T @ (w * (y - p)) - P @ beta)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


@dataclass
class Layer:
    spec: Spec
    beta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    beta_skill_only: np.ndarray = field(default_factory=lambda: np.zeros(0))
    names: list[str] = field(default_factory=list)

    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp) -> Layer:
        train = train[train.m0.notna() & train.y.notna()]
        y = train.y.to_numpy(float)
        w = None
        if self.spec.half_life_days:
            age = (cutoff - pd.to_datetime(train.time)).dt.total_seconds().to_numpy() / 86400.0
            w = 0.5 ** (age / self.spec.half_life_days)
        X, pen, self.names = design(train, self.spec)
        self.beta = fit_logistic(X, y, pen, w)
        if self.spec.skill:
            Xs = np.column_stack([logit(train[self.spec.skill]), np.ones(len(train))])
            self.beta_skill_only = fit_logistic(Xs, y, np.zeros(2), w)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        X, _, _ = design(d, self.spec)
        q = 1 / (1 + np.exp(-X @ self.beta))
        no_market = d.p_last.isna().to_numpy()
        if no_market.any():
            if self.spec.skill:
                Xs = np.column_stack([logit(d[self.spec.skill]), np.ones(len(d))])
                q = np.where(no_market, 1 / (1 + np.exp(-Xs @ self.beta_skill_only)), q)
            else:
                q = np.where(no_market, 0.5, q)
        return q
