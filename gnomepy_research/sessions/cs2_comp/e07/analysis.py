"""Out-of-sample read of a dev walk: lineup inference against the true five, and which stacker features pay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e07.stack import THIN_MAPS, design, fit

Q_FLOOR = 0.02


def ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, Q_FLOOR, 1 - Q_FLOOR)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def designs(d: pd.DataFrame, zcol: str = "q_soft") -> dict[str, np.ndarray]:
    """Candidate stackers by letter; the HLTV filter's price comes from `zcol`."""
    d = d.assign(q_soft=d[zcol])
    sets = {"A": ["z_hltv"], "RK": ["z_hltv", "rank_points"], "E": ["z_hltv", "elo_team"], "RH": ["z_hltv", "rating_hltv"], "ZU": ["z_hltv", "z_unstable"], "ZM": ["z_hltv", "z_moved"]}
    if "f_s_a" in d:
        sets.update({"B": ["z_hltv", "faceit_skill"], "K": ["z_hltv", "faceit_skill", "faceit_skill_thin"],
                     "R": ["z_hltv", "faceit_skill_unsure"], "S": ["z_hltv", "faceit_skill", "faceit_skill_unsure"],
                     "T": ["z_hltv", "faceit_skill_unsure", "faceit_skill_thin"],
                     "O": ["z_hltv", "faceit_skill", "faceit_skill_thin", "faceit_coverage"],
                     "P": ["z_hltv", "faceit_skill", "five_maps"], "L": ["z_hltv", "faceit_skill", "faceit_skill_thin", "z_thin"],
                     "U": ["z_hltv", "faceit_skill", "faceit_skill_unsure", "rank_points"],
                     "V": ["z_hltv", "faceit_skill", "elo_team", "elo_team_faceit"], "W": ["z_hltv", "faceit_skill", "elo_team"],
                     "RF": ["z_hltv", "faceit_skill", "rating_faceit"], "RR": ["z_hltv", "faceit_skill", "rating_hltv", "rating_faceit"],
                     "RE": ["z_hltv", "faceit_skill", "elo_team", "rating_hltv", "rating_faceit"],
                     "FL": ["z_hltv", "faceit_skill_online", "faceit_skill_lan"],
                     "FZ": ["z_hltv", "z_lan", "faceit_skill_online", "faceit_skill_lan"],
                     "FR": ["z_hltv", "faceit_skill_online", "faceit_skill_lan", "rank_points", "rating_hltv", "rating_faceit"],
                     "PR": ["z_hltv", "faceit_skill", "rank_points", "faceit_coverage", "practice_14", "practice_90"],
                     "PR1": ["z_hltv", "faceit_skill", "rank_points", "faceit_coverage", "practice_90"],
                     "FM": ["z_hltv", "faceit_skill", "faceit_form"], "FMY": ["z_hltv", "faceit_skill", "faceit_form", "rank_points"],
                     "X": ["z_hltv", "faceit_skill", "elo_team", "elo_team_faceit", "rank_points"]})
    return {k: design(d, names) for k, names in sets.items()}


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
    for col in ("q_last", "q_h", "q_soft", "q_ens"):
        if col in p:
            out[f"ll_{col}"] = float(ll(p[col].to_numpy(), y).mean())
    if "q_oracle" in p:
        o = p.q_oracle.notna().to_numpy()
        changed = o & ((p.same_a < 5) | (p.same_b < 5)).to_numpy()
        out["oracle_n"] = int(o.sum())
        out["ll_last_on_oracle"] = float(ll(p.q_last.to_numpy()[o], y[o]).mean())
        out["ll_h_on_oracle"] = float(ll(p.q_h.to_numpy()[o], y[o]).mean())
        out["ll_soft_on_oracle"] = float(ll(p.q_soft.to_numpy()[o], y[o]).mean())
        out["ll_oracle"] = float(ll(p.q_oracle.to_numpy()[o], y[o]).mean())
        out["share_changed"] = float(changed.sum() / o.sum())
        out["ll_last_changed"] = float(ll(p.q_last.to_numpy()[changed], y[changed]).mean())
        out["ll_h_changed"] = float(ll(p.q_h.to_numpy()[changed], y[changed]).mean())
        out["ll_oracle_changed"] = float(ll(p.q_oracle.to_numpy()[changed], y[changed]).mean())
    for zcol in ("q_last", "q_h", "q_soft", "q_ens"):
        base = None
        for name, x in designs(p, zcol).items():
            pred = cv(x, y, months)
            loss = ll(pred, y)
            if base is None:
                base = loss
            d = loss - base
            key = f"cv_{zcol}_{name}"
            out[key] = {"ll": float(loss.mean()), "d_vs_A": float(d.mean()), "se": float(d.std() / np.sqrt(len(y))),
                        "d_thin": float(d[thin].mean()) if thin.any() else None, "d_rest": float(d[~thin].mean()),
                        "w": [round(float(v), 3) for v in fit(x, y)]}
            if zcol not in ("q_soft", "q_ens") and name != "A":
                del out[key]
    return out
