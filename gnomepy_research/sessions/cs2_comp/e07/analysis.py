"""Out-of-sample read of a dev walk: lineup inference against the true five, and which stacker features pay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e07.stack import fit, logit

THIN_MAPS = 15.0
Q_FLOOR = 0.02


def ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, Q_FLOOR, 1 - Q_FLOOR)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _w(tm: pd.Series) -> np.ndarray:
    return 1.0 / (1.0 + tm.to_numpy(float) / THIN_MAPS)


def designs(d: pd.DataFrame, zcol: str = "q_h") -> dict[str, np.ndarray]:
    z = logit(d[zcol].to_numpy())
    out = {"A": np.c_[z]}
    if "f_s_a" in d:
        dfs = (d.f_s_a - d.f_s_b).to_numpy()
        thin = (d.f_s_a * _w(d.tm90_a) - d.f_s_b * _w(d.tm90_b)).to_numpy()
        thin_any = (np.minimum(d.tm90_a, d.tm90_b) <= THIN_MAPS).to_numpy(float)
        dn = ((d.f_n_a - d.f_n_b) / 5).to_numpy()
        d5 = (np.log1p(d.five_h_a) - np.log1p(d.five_h_b) + np.log1p(d.five_f_a) - np.log1p(d.five_f_b)).to_numpy()
        out.update({"B": np.c_[z, dfs], "K": np.c_[z, dfs, thin], "L": np.c_[z, dfs, thin, z * thin_any],
                    "O": np.c_[z, dfs, thin, dn], "P": np.c_[z, dfs, d5], "Q": np.c_[z, thin]})
    return out


def cv(x: np.ndarray, y: np.ndarray, months: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(y))
    for m in np.unique(months):
        te = months == m
        pred[te] = 1 / (1 + np.exp(-x[te] @ fit(x[~te], y[~te])))
    return pred


def report(p: pd.DataFrame) -> dict:
    y = p.y.to_numpy(float)
    months = pd.to_datetime(p.cutoff).dt.to_period("M").astype(str).to_numpy()
    thin = ((p.is_lan == 0) & (np.minimum(p.tm90_a, p.tm90_b) <= THIN_MAPS)).to_numpy()
    out = {"n": len(p), "n_thin": int(thin.sum())}
    for col in ("q_last", "q_h"):
        if col in p:
            out[f"ll_{col}"] = float(ll(p[col].to_numpy(), y).mean())
    if "q_oracle" in p:
        o = p.q_oracle.notna().to_numpy()
        changed = o & ((p.same_a < 5) | (p.same_b < 5)).to_numpy()
        out["oracle_n"] = int(o.sum())
        out["ll_last_on_oracle"] = float(ll(p.q_last.to_numpy()[o], y[o]).mean())
        out["ll_h_on_oracle"] = float(ll(p.q_h.to_numpy()[o], y[o]).mean())
        out["ll_oracle"] = float(ll(p.q_oracle.to_numpy()[o], y[o]).mean())
        out["share_changed"] = float(changed.sum() / o.sum())
        out["ll_last_changed"] = float(ll(p.q_last.to_numpy()[changed], y[changed]).mean())
        out["ll_h_changed"] = float(ll(p.q_h.to_numpy()[changed], y[changed]).mean())
        out["ll_oracle_changed"] = float(ll(p.q_oracle.to_numpy()[changed], y[changed]).mean())
    base = None
    for name, x in designs(p).items():
        pred = cv(x, y, months)
        loss = ll(pred, y)
        if base is None:
            base = loss
        d = loss - base
        out[f"cv_{name}"] = {"ll": float(loss.mean()), "d_vs_A": float(d.mean()), "se": float(d.std() / np.sqrt(len(y))),
                             "d_thin": float(d[thin].mean()) if thin.any() else None, "d_rest": float(d[~thin].mean()),
                             "w": [round(float(v), 3) for v in fit(x, y)]}
    return out
