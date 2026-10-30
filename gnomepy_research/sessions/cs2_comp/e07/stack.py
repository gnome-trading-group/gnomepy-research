"""
The last stage: a logistic regression without an intercept (so a series stays symmetric in A and B) on features that
are each antisymmetric in the two teams, with weights fitted on HLTV series before the tune period and frozen in
final_config.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Q_LIMIT = 1e-4
THIN_MAPS = 15.0


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), Q_LIMIT, 1 - Q_LIMIT)
    return np.log(p / (1 - p))


def _thin_weight(maps: pd.Series) -> np.ndarray:
    return 1.0 / (1.0 + maps.to_numpy(float) / THIN_MAPS)


FEATURES = {
    "z_hltv": lambda d: logit(d.q_soft.to_numpy()),
    "z_last": lambda d: logit(d.q_last.to_numpy()),
    "z_ens": lambda d: logit(d.q_ens.to_numpy()),
    "z_avg": lambda d: 0.5 * (logit(d.q_soft.to_numpy()) + logit(d.q_last.to_numpy())),
    "faceit_skill": lambda d: (d.f_s_a - d.f_s_b).to_numpy(float),
    "faceit_skill_unsure": lambda d: (d.f_su_a - d.f_su_b).to_numpy(float),
    "faceit_skill_thin": lambda d: (d.f_s_a * _thin_weight(d.tm90_a) - d.f_s_b * _thin_weight(d.tm90_b)).to_numpy(float),
    "faceit_coverage": lambda d: ((d.f_n_a - d.f_n_b) / 5.0).to_numpy(float),
    "rank_points": lambda d: (d.rank_a - d.rank_b).to_numpy(float),
    "faceit_skill_online": lambda d: (d.f_s_a - d.f_s_b).to_numpy(float) * (1.0 - d.is_lan.to_numpy(float)),
    "faceit_skill_lan": lambda d: (d.f_s_a - d.f_s_b).to_numpy(float) * d.is_lan.to_numpy(float),
    "z_lan": lambda d: logit(d.q_soft.to_numpy()) * d.is_lan.to_numpy(float),
    "z_unstable": lambda d: logit(d.q_soft.to_numpy()) * ((d.inst_a + d.inst_b) / 2.0).to_numpy(float),
    "z_moved": lambda d: logit(d.q_soft.to_numpy()) * ((d.swap_a + d.swap_b) > 0).to_numpy(float),
    "elo_team": lambda d: (d.elo_h_a - d.elo_h_b).to_numpy(float),
    "elo_team_faceit": lambda d: (d.elo_f_a - d.elo_f_b).to_numpy(float),
    "rating_hltv": lambda d: (d.rat_h_a - d.rat_h_b).to_numpy(float),
    "rating_faceit": lambda d: (d.rat_f_a - d.rat_f_b).to_numpy(float),
    "practice_14": lambda d: (d.prac14_a - d.prac14_b).to_numpy(float),
    "practice_90": lambda d: (d.prac90_a - d.prac90_b).to_numpy(float),
    "five_maps": lambda d: (np.log1p(d.five_h_a) - np.log1p(d.five_h_b) + np.log1p(d.five_f_a) - np.log1p(d.five_f_b)).to_numpy(float),
    "z_thin": lambda d: logit(d.q_soft.to_numpy()) * (np.minimum(d.tm90_a, d.tm90_b) <= THIN_MAPS).to_numpy(float),
}


def design(d: pd.DataFrame, names: list[str]) -> np.ndarray:
    return np.column_stack([FEATURES[n](d) for n in names])


def fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-x @ w))
        hess = x.T @ (x * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(hess, x.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def apply(d: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return 1 / (1 + np.exp(-design(d, list(weights)) @ np.array(list(weights.values()))))
