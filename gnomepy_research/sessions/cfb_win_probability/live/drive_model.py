"""Drive-level transition model, stored as dense arrays the numba simulator reads directly.

    P(outcome | start ytg bin, situation, s) = softmax_k(base_logits[k, ybin, sit] + beta[k] * s)

where s is the offense's net strength (offense rating minus defense rating, plus home field) for
that simulated game. Next-possession field position and drive duration are empirical tables.

`fit_empirical` is the step-2 baseline: smoothed empirical frequencies with beta = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.live.rules import (
    DEF_TD,
    END_HALF,
    FG,
    N_DIFF,
    N_OUTCOMES,
    N_SIT,
    N_TIME,
    N_YTG,
    OUTCOME_INDEX,
    QUARTER_SECONDS,
    SAFETY,
    SIT_OT,
    TD,
    diff_bin,
    half_clock,
    situation,
    time_bin,
    ytg_bin,
)

DURATION_QUANTILES = np.linspace(0.0, 1.0, 21)
N_DUR_Q = len(DURATION_QUANTILES)
# Possession-change outcomes whose next start depends on where the drive ended.
SPOT_OUTCOMES = ("PUNT", "TURNOVER", "DOWNS", "MISSED_FG")
# Offense-perspective lead after the TD (before the try) at which 4th-quarter teams go for two.
DEFAULT_TWO_POINT_DIFFS = np.array([-10, -5, -2, 1, 5, 12], dtype=np.int64)


@dataclass
class DriveModel:
    base_logits: np.ndarray       # (N_OUTCOMES, N_YTG, N_SIT)
    beta: np.ndarray              # (N_OUTCOMES,)
    next_cdf: np.ndarray          # (N_OUTCOMES, N_YTG, 99): next offense's start ytg 1..99
    kick_cdf: np.ndarray          # (99,) receiving team's start after a kickoff
    safety_kick_cdf: np.ndarray   # (99,) receiving team's start after a safety free kick
    duration_q: np.ndarray        # (N_OUTCOMES, N_SIT, N_DUR_Q) seconds of game clock per drive
    p_xp: float = 0.95
    p_two: float = 0.45
    two_point_diffs: np.ndarray = None
    # Share of a trailing OT team's would-be field goals that become TDs when a FG can't tie: they
    # go for it on 4th down instead of kicking.
    ot_fg_to_td: float = 0.35

    def __post_init__(self) -> None:
        if self.two_point_diffs is None:
            self.two_point_diffs = DEFAULT_TWO_POINT_DIFFS.copy()

    def outcome_probs(self, ytg: int, sit: int, s: float = 0.0) -> np.ndarray:
        logits = self.base_logits[:, ytg_bin(ytg), sit] + self.beta * s
        p = np.exp(logits - logits.max())
        return p / p.sum()


def prepare_drives(drives: pd.DataFrame) -> pd.DataFrame:
    """Add simulator-space columns (half, half clock, situation, bins, next possession start)."""
    d = drives[drives["outcome"].isin(OUTCOME_INDEX)].copy()
    d = d.sort_values(["game_id", "drive_number"]).reset_index(drop=True)
    d["k"] = d["outcome"].map(OUTCOME_INDEX).astype(int)
    d["half"] = np.where(d["start_period"] <= 2, 1, np.where(d["start_period"] <= 4, 2, 3))
    d["end_half"] = np.where(d["end_period"] <= 2, 1, np.where(d["end_period"] <= 4, 2, 3))
    d["half_clock"] = [half_clock(p, c) for p, c in zip(d["start_period"], d["start_clock_s"])]
    end_clock = np.array([half_clock(p, c) for p, c in zip(d["end_period"], d["end_clock_s"])])
    d["duration"] = np.clip(np.where(d["half"] == d["end_half"], d["half_clock"] - end_clock, np.nan),
                            0, QUARTER_SECONDS)
    d["diff"] = (d["start_offense_score"] - d["start_defense_score"]).astype(int)
    d["ytg"] = d["start_yards_to_goal"].clip(1, 99).astype(int)
    d["ybin"] = [ytg_bin(y) for y in d["ytg"]]
    d["sit"] = [situation(h, c, x) for h, c, x in zip(d["half"], d["half_clock"], d["diff"])]
    d["tbin"] = [time_bin(c) for c in d["half_clock"]]
    d["dbin"] = [diff_bin(x) for x in d["diff"]]

    nxt = d.groupby("game_id").shift(-1)
    same_half = nxt["half"] == d["half"]
    d["next_ytg"] = np.where(same_half & (nxt["offense_id"] == d["defense_id"]), nxt["ytg"], np.nan)
    return d


def _smooth(counts: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    n = counts.sum(axis=0, keepdims=True)
    return (counts + alpha * prior) / (n + alpha)


def _cdf_from_values(values: np.ndarray, prior: np.ndarray | None = None, alpha: float = 0.0) -> np.ndarray:
    hist = np.bincount(np.clip(values.astype(int), 1, 99) - 1, minlength=99).astype(float)
    if prior is not None:
        hist = hist + alpha * np.diff(np.concatenate([[0.0], prior]))
    if hist.sum() == 0:
        hist[:] = 1.0
    cdf = np.cumsum(hist) / hist.sum()
    cdf[-1] = 1.0
    return cdf


def _quantiles(x: np.ndarray) -> np.ndarray:
    return np.quantile(x, DURATION_QUANTILES)


def fit_empirical(drives: pd.DataFrame, alpha: float = 20.0, min_n: int = 30,
                  score_effects: bool = False) -> DriveModel:
    """score_effects=False by default: raw outcome rates by score margin are confounded with team
    strength (the team up 17 is usually the better team), and feeding them to a strength-blind
    simulator inflates the margin sd from ~16 to ~21 points. Score effects belong in a model fitted
    jointly with team strength."""
    d = prepare_drives(drives)
    reg = d[d["half"] <= 2]
    ot = d[d["half"] == 3]

    # Outcome probabilities: ybin -> (ybin, half, time) -> (ybin, situation), each shrunk to its parent.
    c0 = np.zeros((N_OUTCOMES, N_YTG))
    np.add.at(c0, (reg["k"], reg["ybin"]), 1)
    p0 = _smooth(c0, np.full((N_OUTCOMES, 1), 1.0 / N_OUTCOMES), alpha)

    c1 = np.zeros((N_OUTCOMES, N_YTG, 2, N_TIME))
    np.add.at(c1, (reg["k"], reg["ybin"], reg["half"] - 1, reg["tbin"]), 1)
    p1 = _smooth(c1, p0[:, :, None, None], alpha)

    if score_effects:
        c2 = np.zeros((N_OUTCOMES, N_YTG, 2, N_TIME, N_DIFF))
        np.add.at(c2, (reg["k"], reg["ybin"], reg["half"] - 1, reg["tbin"], reg["dbin"]), 1)
        p2 = _smooth(c2, p1[..., None], alpha)
    else:
        p2 = np.repeat(p1[..., None], N_DIFF, axis=-1)
    p2 = p2.reshape(N_OUTCOMES, N_YTG, -1)

    probs = np.zeros((N_OUTCOMES, N_YTG, N_SIT))
    probs[:, :, :SIT_OT] = p2

    # OT possessions can't run out of clock; use the no-clock regulation mix as the prior.
    p0_ot = p0.copy()
    p0_ot[END_HALF] = 0.0
    p0_ot /= p0_ot.sum(axis=0, keepdims=True)
    c_ot = np.zeros((N_OUTCOMES, N_YTG))
    ot_valid = ot[ot["ytg"] > 3]  # 2-point shootout plays are coded as drives from the 3
    np.add.at(c_ot, (ot_valid["k"], ot_valid["ybin"]), 1)
    p_ot = _smooth(c_ot, p0_ot, alpha)
    p_ot[END_HALF] = 0.0
    probs[:, :, SIT_OT] = p_ot / p_ot.sum(axis=0, keepdims=True)

    base_logits = np.log(np.clip(probs, 1e-9, None))

    # Next possession start.
    kick_rows = reg[reg["k"].isin([TD, FG, DEF_TD])].dropna(subset=["next_ytg"])
    kick_cdf = _cdf_from_values(kick_rows["next_ytg"].to_numpy())
    safety_rows = reg[reg["k"] == SAFETY].dropna(subset=["next_ytg"])
    safety_kick_cdf = _cdf_from_values(safety_rows["next_ytg"].to_numpy(), kick_cdf, alpha)

    next_cdf = np.tile(kick_cdf, (N_OUTCOMES, N_YTG, 1))
    for name in SPOT_OUTCOMES:
        k = OUTCOME_INDEX[name]
        rows = reg[reg["k"] == k].dropna(subset=["next_ytg"])
        pooled = _cdf_from_values(rows["next_ytg"].to_numpy())
        for b in range(N_YTG):
            vals = rows.loc[rows["ybin"] == b, "next_ytg"].to_numpy()
            next_cdf[k, b] = _cdf_from_values(vals, pooled, alpha)

    # Drive duration: situation -> (half, time) -> outcome fallback when a cell is thin.
    dur = reg.dropna(subset=["duration"])
    duration_q = np.zeros((N_OUTCOMES, N_SIT, N_DUR_Q))
    for k in range(N_OUTCOMES):
        dk = dur[dur["k"] == k]
        overall = _quantiles(dk["duration"].to_numpy()) if len(dk) else np.zeros(N_DUR_Q)
        duration_q[k, SIT_OT] = 0.0
        by_ht = {key: g["duration"].to_numpy() for key, g in dk.groupby(["half", "tbin"])}
        by_sit = {key: g["duration"].to_numpy() for key, g in dk.groupby("sit")}
        for sit in range(SIT_OT):
            h, rem = divmod(sit, N_TIME * N_DIFF)
            t = rem // N_DIFF
            if len(by_sit.get(sit, ())) >= min_n:
                duration_q[k, sit] = _quantiles(by_sit[sit])
            elif len(by_ht.get((h + 1, t), ())) >= min_n:
                duration_q[k, sit] = _quantiles(by_ht[(h + 1, t)])
            else:
                duration_q[k, sit] = overall

    td = reg[reg["k"] == TD]
    kicked = td[td["half"] == 1]
    p_xp = float((kicked["off_points"] == 7).sum() / max(kicked["off_points"].isin([6, 7]).sum(), 1))

    return DriveModel(
        base_logits=base_logits,
        beta=np.zeros(N_OUTCOMES),
        next_cdf=next_cdf,
        kick_cdf=kick_cdf,
        safety_kick_cdf=safety_kick_cdf,
        duration_q=duration_q,
        p_xp=p_xp,
    )
