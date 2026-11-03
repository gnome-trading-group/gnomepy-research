"""Arrays for the dynamic efficiency model, built from point-in-time inputs only.

Index spaces:
- team-season (ts): one per (season, FBS team) plus one pooled FCS entity per season.
- period (k): the season's weekly cutoffs in order. A game belongs to the period of its week; the
  state at period k is what a prediction made at that week's cutoff uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.features.cutoffs import WEEK_KEYS, week_cutoffs
from gnomepy_research.models.cfb.features.ratings import FCS_TEAM

OFF_FEATURES = [
    "prior_off_rating", "talent_offense", "talent_qb", "recruiting_points", "returning_pct_ppa",
    "returning_pct_passing_ppa", "qb_prior_ppa", "qb_returning_starter", "coach_srs_resid", "coach_new",
    "portal_in_stars",
]
DEF_FEATURES = [
    "prior_def_rating", "talent_defense", "recruiting_points", "coach_srs_resid", "coach_new", "portal_in_stars",
]
# Flags that widen the preseason prior: less is known about a team that changed a lot.
UNCERTAINTY_FLAGS = ["coach_new", "qb_not_returning", "prior_missing"]
RETURNING_FEATURES = ["returning_pct_ppa", "returning_pct_passing_ppa"]
# CFBD's returning production comes from its own roster snapshot. When last season's primary passer has left (our roster
# check) yet at least this share of passing production "returns", that snapshot is stale.
STALE_RETURNING_PASSING = 0.8


def unknown_stale_returning(pre: pd.DataFrame) -> pd.DataFrame:
    """Returning production read from a stale CFBD roster snapshot becomes unknown (the season mean once standardized).
    2026 Southern Miss: 97% of passing production 'returning' after its 2025 starter left."""
    pre = pre.copy()
    stale = (pre["qb_returning_starter"] == 0) & (pre["returning_pct_passing_ppa"] >= STALE_RETURNING_PASSING)
    pre.loc[stale.to_numpy(), RETURNING_FEATURES] = np.nan
    return pre


@dataclass
class ModelData:
    ts_keys: pd.DataFrame          # (season, team_id) per team-season index; FCS rows have team_id == FCS_TEAM
    n_periods: int
    x_off: np.ndarray              # (TS, F_off) standardized preseason features + missing flags + is_fcs
    x_def: np.ndarray              # (TS, F_def)
    z_unc: np.ndarray              # (TS, U) uncertainty flags
    off_names: list[str]
    def_names: list[str]
    obs_off: np.ndarray            # (N,) offense ts index
    obs_def: np.ndarray            # (N,) defense ts index
    obs_period: np.ndarray         # (N,) period index
    obs_venue: np.ndarray          # (N,) +1 home, -1 away, 0 neutral
    obs_plays: np.ndarray          # (N,)
    obs_y: np.ndarray              # (N,) offensive PPA per play
    obs_game: np.ndarray           # (N,) game_id (for diagnostics)
    obs_season: np.ndarray         # (N,) index into `seasons`
    seasons: list[int]
    periods: pd.DataFrame          # (season, period) -> cutoff, season_type, week


def season_periods(games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    c = week_cutoffs(games)
    c = c[c["season"].isin(seasons)].sort_values("cutoff")
    c["period"] = c.groupby("season").cumcount()
    return c.reset_index(drop=True)


def _standardize(pre: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """z-score each feature within its season over FBS teams; missing -> 0 plus a missing flag
    when the feature is ever missing (prior ratings in 2018, the portal before 2021)."""
    out = pd.DataFrame(index=pre.index)
    names = []
    for col in cols:
        v = pre[col].astype(float)
        mean = v.groupby(pre["season"]).transform("mean")
        sd = v.groupby(pre["season"]).transform("std").replace(0, np.nan)
        z = ((v - mean) / sd).fillna(0.0)
        out[col] = z
        names.append(col)
    # Only flags that vary within a season: a season-constant flag (portal data missing before 2021)
    # is indistinguishable from that season's baseline. The z-scored feature is 0 when missing.
    if "prior_off_rating" in cols:
        out["prior_missing"] = pre["prior_off_rating"].isna().astype(float)
        names.append("prior_missing")
    return out, names


def build_model_data(preseason: pd.DataFrame, efficiency: pd.DataFrame, games: pd.DataFrame,
                     seasons: list[int]) -> ModelData:
    """`efficiency` must already be point-in-time filtered (as_of the fit's cutoff)."""
    pre = preseason[preseason["season"].isin(seasons)].sort_values(["season", "team_id"]).reset_index(drop=True)
    pre = unknown_stale_returning(pre)
    pre["qb_returning_starter"] = pre["qb_returning_starter"].astype(float)
    pre["coach_new"] = pre["coach_new"].astype(float)
    xo, off_names = _standardize(pre, OFF_FEATURES)
    xd, def_names = _standardize(pre, DEF_FEATURES)

    fcs = pd.DataFrame({"season": seasons, "team_id": FCS_TEAM})
    ts_keys = pd.concat([pre[["season", "team_id"]], fcs], ignore_index=True)
    n_fbs = len(pre)

    def with_fcs(x: pd.DataFrame) -> np.ndarray:
        m = np.vstack([x.to_numpy(float), np.zeros((len(seasons), x.shape[1]))])
        is_fcs = np.r_[np.zeros(n_fbs), np.ones(len(seasons))][:, None]
        return np.hstack([m, is_fcs])

    unc = pd.DataFrame({
        "coach_new": pre["coach_new"].fillna(1.0),
        "qb_not_returning": 1.0 - pre["qb_returning_starter"].fillna(0.0),
        "prior_missing": pre["prior_off_rating"].isna().astype(float),
    })
    z_unc = np.vstack([unc[UNCERTAINTY_FLAGS].to_numpy(float), np.ones((len(seasons), len(UNCERTAINTY_FLAGS)))])

    periods = season_periods(games, seasons)
    e = efficiency[efficiency["season"].isin(seasons) & efficiency["usable"]]
    e = e.merge(periods[[*WEEK_KEYS, "period"]], on=WEEK_KEYS)

    idx = {(s, t): i for i, (s, t) in enumerate(zip(ts_keys["season"], ts_keys["team_id"]))}
    off_team = np.where(e["team_classification"] == "fbs", e["team_id"], FCS_TEAM)
    def_team = np.where(e["opponent_classification"] == "fbs", e["opponent_id"], FCS_TEAM)
    obs_off = np.array([idx.get((s, t), -1) for s, t in zip(e["season"], off_team)], dtype=int)
    obs_def = np.array([idx.get((s, t), -1) for s, t in zip(e["season"], def_team)], dtype=int)
    # An FBS team missing from the preseason table (e.g. a program's first FBS year) is pooled
    # with FCS rather than silently dropped.
    fcs_idx = {s: idx[(s, FCS_TEAM)] for s in seasons}
    fcs_of_row = np.array([fcs_idx[s] for s in e["season"]], dtype=int)
    obs_off = np.where(obs_off < 0, fcs_of_row, obs_off)
    obs_def = np.where(obs_def < 0, fcs_of_row, obs_def)

    return ModelData(
        ts_keys=ts_keys,
        n_periods=int(periods.groupby("season")["period"].max().max()) + 1,
        x_off=with_fcs(xo), x_def=with_fcs(xd), z_unc=z_unc,
        off_names=[*off_names, "is_fcs"], def_names=[*def_names, "is_fcs"],
        obs_off=obs_off, obs_def=obs_def, obs_period=e["period"].to_numpy(int),
        obs_venue=e["venue"].to_numpy(float), obs_plays=e["plays"].to_numpy(float),
        obs_y=e["off_ppa"].to_numpy(float), obs_game=e["game_id"].to_numpy(),
        obs_season=e["season"].map({s: i for i, s in enumerate(seasons)}).to_numpy(int), seasons=list(seasons),
        periods=periods,
    )
