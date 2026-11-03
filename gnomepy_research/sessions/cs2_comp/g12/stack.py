"""
Stacking the raters with the market.

Stage 1 (no market) is a ridge logistic regression of the series outcome on the raters' A-minus-B differences and
their reliability interactions, fitted on a trailing year of bo3 series. Its output z is one model price per series.

Stage 2 is a ridge logistic regression on logit(m0), z and a few terms that let z's weight grow where the market is
weak (thin, young or rarely moving books), fitted on every past bo3 series with a kickoff price. A series without a
kickoff price falls back to stage 1 alone, with the market-rating gap added.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RATERS = ["team_elo", "player_elo", "joint_elo", "hltv_rating", "faceit_level", "log_points", "market_elo"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: np.ndarray | float, w: np.ndarray | None = None) -> np.ndarray:
    """Newton's method with an L2 penalty per coefficient (deterministic)."""
    n, k = X.shape
    w = np.ones(n) if w is None else w
    pen = np.broadcast_to(np.asarray(l2, float), (k,)).copy()
    b = np.zeros(k)
    for _ in range(100):
        p = sigmoid(X @ b)
        g = X.T @ (w * (y - p)) - pen * b
        H = X.T @ (X * (w * p * (1 - p))[:, None]) + np.diag(pen) + 1e-9 * np.eye(k)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def stage1_design(d: pd.DataFrame) -> pd.DataFrame:
    """A-minus-B rater gaps (antisymmetric), with availability terms; missing raters are zero with an indicator."""
    X = {}
    X["team_elo"] = d.team_elo_a - d.team_elo_b
    X["player_elo"] = d.player_elo_a - d.player_elo_b
    X["joint_elo"] = d.joint_elo_a - d.joint_elo_b
    X["gauss"] = d.gauss_a - d.gauss_b
    X["gauss_h"] = d.gauss_h_a - d.gauss_h_b
    X["map_adj"] = d.map_adj if "map_adj" in d else 0.0
    X["gauss_unc"] = X["gauss"] * (d.gauss_var_a + d.gauss_var_b)
    both_rt = d.hltv_rating_a.notna() & d.hltv_rating_b.notna()
    X["hltv_rating"] = np.where(both_rt, d.hltv_rating_a - d.hltv_rating_b, 0.0)
    both_lv = (d.level_n_a >= 3) & (d.level_n_b >= 3)
    X["faceit_level"] = np.where(both_lv, d.faceit_level_a - d.faceit_level_b, 0.0)
    X["log_points"] = d.log_points_a - d.log_points_b
    X["ranked"] = d.ranked_a - d.ranked_b
    unk = (d.player_unknown_a + d.player_unknown_b).to_numpy(float)
    X["player_elo_unk"] = X["player_elo"] * unk
    X["joint_elo_unk"] = X["joint_elo"] * (d.joint_unknown_a + d.joint_unknown_b)
    X["unknown"] = d.player_unknown_a - d.player_unknown_b
    X["idle"] = np.log1p(d.idle_days_a.clip(0, 365)) - np.log1p(d.idle_days_b.clip(0, 365))
    X["new"] = d.new_players_a - d.new_players_b
    X["sub"] = d.sub_a.astype(float) - d.sub_b.astype(float)
    X["standin"] = d.standin_share_a - d.standin_share_b
    X["form"] = np.log1p(d.form_a) - np.log1p(d.form_b)
    X["market_elo"] = d.market_elo_a - d.market_elo_b
    X["market_bias"] = d.market_bias_a - d.market_bias_b
    return pd.DataFrame(X, index=d.index).astype(float)


def stage2_design(d: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    lm = logit(d.m0.to_numpy(float))
    young = (d.age_h.to_numpy(float) < 12).astype(float)
    still = (d.changes.to_numpy(float) < 30).astype(float)
    weak = ((d.rank_a.to_numpy(float) > 100) | (d.rank_b.to_numpy(float) > 100)).astype(float)
    lan = d.is_lan.to_numpy(float)
    drift = lm - logit(d.m_24h.fillna(d.m_first).to_numpy(float))
    mk = (d.market_elo_a - d.market_elo_b).to_numpy(float)
    mk_ok = ((d.market_n_a >= 2) & (d.market_n_b >= 2)).to_numpy(float)
    return pd.DataFrame({"m": lm, "z": z, "z_young": z * young, "z_still": z * still, "z_weak": z * weak, "z_lan": z * lan,
                         "m_weak": lm * weak, "drift": drift, "mk_gap": (mk - lm) * mk_ok, "one": 1.0}, index=d.index)


def market_terms(d: pd.DataFrame) -> pd.DataFrame:
    """Shape of the market's own price: curvature, books that are young or barely move, and the last day's drift."""
    lm = logit(d.m0.to_numpy(float))
    weak = ((d.rank_a.to_numpy(float) > 100) | (d.rank_b.to_numpy(float) > 100)).astype(float)
    return pd.DataFrame({"m_curve": lm * np.abs(lm), "m_weak": lm * weak,
                         "m_young": lm * (d.age_h.to_numpy(float) < 12),
                         "m_still": lm * (d.changes.to_numpy(float) < 30),
                         "drift": lm - logit(d.m_24h.fillna(d.m_first).to_numpy(float)),
                         "m_lan": lm * d.is_lan.to_numpy(float)}, index=d.index)


@dataclass
class Stack:
    l2_1: float = 3.0
    l2_2: float = 10.0
    window_days: int = 365
    cols2: list = field(default_factory=lambda: ["m", "z", "z_young", "z_still", "z_weak", "z_lan", "m_weak", "drift", "mk_gap", "one"])
    exclude1: tuple = ("market_elo", "market_bias")
    use1: tuple | None = None
    direct: tuple | None = None
    l2_d: float = 20.0
    market_terms: tuple = ()
    model_only: bool = False
    b1: np.ndarray | None = None
    b2: np.ndarray | None = None
    b_fallback: np.ndarray | None = None
    cols1: list | None = None

    def fit(self, hist: pd.DataFrame, cutoff: pd.Timestamp) -> Stack:
        h = hist[(hist.known < cutoff) & hist.y.notna() & (hist.bo_type == 3)
                 & (hist.week_cutoff >= cutoff - pd.Timedelta(days=self.window_days))]
        X1 = stage1_design(h)
        self.cols1 = list(self.use1) if self.use1 else [c for c in X1.columns if c not in self.exclude1]
        X1 = X1[self.cols1]
        y = h.y.to_numpy(float)
        self.b1 = fit_logistic(X1.to_numpy(), y, self.l2_1)
        z = X1.to_numpy() @ self.b1
        mk = (h.market_elo_a - h.market_elo_b).to_numpy(float)
        self.b_fallback = fit_logistic(np.column_stack([z, mk]), y, self.l2_1)
        has_m = h.m0.notna().to_numpy()
        X2 = stage2_design(h[has_m], z[has_m])[self.cols2]
        pen = np.full(len(self.cols2), self.l2_2)
        pen[self.cols2.index("m")] = 1e-3
        pen[self.cols2.index("z")] = 1e-3
        self.b2 = fit_logistic(X2.to_numpy(), y[has_m], pen)
        if self.direct:
            Xd = self._direct(h[has_m])
            self.sd = Xd.std(axis=0) + 1e-9
            self.sd[0] = 1.0
            self.sd[-1] = 1.0
            pen = np.full(Xd.shape[1], self.l2_d)
            pen[0] = pen[-1] = 1e-3
            self.bd = fit_logistic(Xd / self.sd, y[has_m], pen)
        self.n_train = (len(h), int(has_m.sum()))
        return self

    def _direct(self, d: pd.DataFrame) -> np.ndarray:
        X = stage1_design(d)[list(self.direct)].to_numpy()
        lm = logit(d.m0.to_numpy(float))
        extra = [market_terms(d)[list(self.market_terms)].to_numpy()] if self.market_terms else []
        return np.column_stack([lm, X] + extra + [np.ones(len(d))])

    def z(self, d: pd.DataFrame) -> np.ndarray:
        return stage1_design(d)[self.cols1].to_numpy() @ self.b1

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        z = self.z(d)
        if self.model_only:
            return sigmoid(z)
        has_m = d.m0.notna().to_numpy()
        mk = (d.market_elo_a - d.market_elo_b).to_numpy(float)
        out = sigmoid(np.column_stack([z, mk]) @ self.b_fallback)
        if has_m.any():
            if self.direct:
                out[has_m] = sigmoid((self._direct(d[has_m]) / self.sd) @ self.bd)
            else:
                out[has_m] = sigmoid(stage2_design(d[has_m], z[has_m])[self.cols2].to_numpy() @ self.b2)
        return out
