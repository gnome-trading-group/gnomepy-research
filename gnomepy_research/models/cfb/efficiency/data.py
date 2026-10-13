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
UNITS_ALL = ("all",)
UNITS_SPLIT = ("pass", "rush")
UNIT_COLUMNS = {"all": ("plays", "off_ppa"), "pass": ("pass_plays", "off_pass_ppa"),
                "rush": ("rush_plays", "off_rush_ppa")}


@dataclass
class ModelData:
    ts_keys: pd.DataFrame          # (season, team_id, unit) per team-season index; FCS rows have team_id == FCS_TEAM
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
    units: tuple[str, ...]
    alpha_keys: pd.DataFrame       # (season, unit) per intercept index
    obs_alpha: np.ndarray          # (N,) intercept index
    obs_unit: np.ndarray           # (N,) index into `units`


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


def _block(x: np.ndarray, u: int, n_units: int) -> np.ndarray:
    out = np.zeros((x.shape[0], x.shape[1] * n_units))
    out[:, u * x.shape[1]:(u + 1) * x.shape[1]] = x
    return out


def _unit_names(names: list[str], units: tuple[str, ...]) -> list[str]:
    return names if units == UNITS_ALL else [f"{u}:{n}" for u in units for n in names]


def build_model_data(preseason: pd.DataFrame, efficiency: pd.DataFrame, games: pd.DataFrame,
                     seasons: list[int], units: tuple[str, ...] = UNITS_ALL) -> ModelData:
    """`efficiency` must already be point-in-time filtered (as_of the fit's cutoff)."""
    pre = preseason[preseason["season"].isin(seasons)].sort_values(["season", "team_id"]).reset_index(drop=True)
    pre["qb_returning_starter"] = pre["qb_returning_starter"].astype(float)
    pre["coach_new"] = pre["coach_new"].astype(float)
    xo, off_names = _standardize(pre, OFF_FEATURES)
    xd, def_names = _standardize(pre, DEF_FEATURES)

    fcs = pd.DataFrame({"season": seasons, "team_id": FCS_TEAM})
    base_keys = pd.concat([pre[["season", "team_id"]], fcs], ignore_index=True)
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
    base_x_off, base_x_def = with_fcs(xo), with_fcs(xd)
    base_z = np.vstack([unc[UNCERTAINTY_FLAGS].to_numpy(float), np.ones((len(seasons), len(UNCERTAINTY_FLAGS)))])

    n_u = len(units)
    ts_keys = pd.concat([base_keys.assign(unit=u) for u in units], ignore_index=True)
    x_off = np.vstack([_block(base_x_off, i, n_u) for i in range(n_u)])
    x_def = np.vstack([_block(base_x_def, i, n_u) for i in range(n_u)])
    z_unc = np.vstack([base_z] * n_u)
    alpha_keys = pd.DataFrame([(s, u) for u in units for s in seasons], columns=["season", "unit"])
    a_idx = {(s, u): i for i, (s, u) in enumerate(zip(alpha_keys["season"], alpha_keys["unit"]))}

    periods = season_periods(games, seasons)
    e = efficiency[efficiency["season"].isin(seasons) & efficiency["usable"]]
    e = e.merge(periods[[*WEEK_KEYS, "period"]], on=WEEK_KEYS)
    idx = {(s, t, u): i for i, (s, t, u) in enumerate(zip(ts_keys["season"], ts_keys["team_id"], ts_keys["unit"]))}
    off_team = np.where(e["team_classification"] == "fbs", e["team_id"], FCS_TEAM)
    def_team = np.where(e["opponent_classification"] == "fbs", e["opponent_id"], FCS_TEAM)

    parts = []
    for ui, u in enumerate(units):
        plays_col, y_col = UNIT_COLUMNS[u]
        keep = (e[plays_col].to_numpy(float) > 0) & np.isfinite(e[y_col].to_numpy(float))
        eu = e[keep]
        fcs_row = np.array([idx[(s, FCS_TEAM, u)] for s in eu["season"]], dtype=int)
        # An FBS team missing from the preseason table is pooled with FCS rather than silently dropped.
        o = np.array([idx.get((s, t, u), -1) for s, t in zip(eu["season"], off_team[keep])], dtype=int)
        d = np.array([idx.get((s, t, u), -1) for s, t in zip(eu["season"], def_team[keep])], dtype=int)
        parts.append(pd.DataFrame({
            "off": np.where(o < 0, fcs_row, o), "def": np.where(d < 0, fcs_row, d),
            "period": eu["period"].to_numpy(int), "venue": eu["venue"].to_numpy(float),
            "plays": eu[plays_col].to_numpy(float), "y": eu[y_col].to_numpy(float), "game": eu["game_id"].to_numpy(),
            "season": eu["season"].map({s: i for i, s in enumerate(seasons)}).to_numpy(int),
            "alpha": [a_idx[(s, u)] for s in eu["season"]], "unit": ui,
        }))
    obs = pd.concat(parts, ignore_index=True)

    return ModelData(
        ts_keys=ts_keys,
        n_periods=int(periods.groupby("season")["period"].max().max()) + 1,
        x_off=x_off, x_def=x_def, z_unc=z_unc,
        off_names=_unit_names([*off_names, "is_fcs"], units), def_names=_unit_names([*def_names, "is_fcs"], units),
        obs_off=obs["off"].to_numpy(int), obs_def=obs["def"].to_numpy(int), obs_period=obs["period"].to_numpy(int),
        obs_venue=obs["venue"].to_numpy(float), obs_plays=obs["plays"].to_numpy(float), obs_y=obs["y"].to_numpy(float),
        obs_game=obs["game"].to_numpy(), obs_season=obs["season"].to_numpy(int), seasons=list(seasons), periods=periods,
        units=tuple(units), alpha_keys=alpha_keys, obs_alpha=obs["alpha"].to_numpy(int), obs_unit=obs["unit"].to_numpy(int),
    )
