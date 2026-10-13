"""Team-level feature tables, each computed from a single as_of view.

- preseason: one row per (season, team) as of the season's first cutoff. Inputs to the structural
  priors on O and D.
- season_to_date: one row per (cutoff, team): what's been observed this season, opponent-adjusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.features.coach_effects import coach_features
from gnomepy_research.models.cfb.features.qb import current_qb, preseason_qb
from gnomepy_research.models.cfb.features.ratings import ridge_ratings
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.models.cfb.features.team_talent import (
    blended_recruiting,
    portal_flow,
    position_group_talent,
    returning_production,
    team_talent,
)


def fbs_teams(src: SourceData, season: int) -> pd.DataFrame:
    g = src.games[src.games["season"] == season]
    ids = pd.concat([g.loc[g["home_classification"] == "fbs", "home_id"],
                     g.loc[g["away_classification"] == "fbs", "away_id"]]).unique()
    return pd.DataFrame({"team_id": np.sort(ids).astype("int64")})


def preseason_features(src: SourceData, season: int) -> pd.DataFrame:
    """`src` must be as_of the season's first cutoff."""
    out = fbs_teams(src, season)
    prior_eff = src.efficiency[src.efficiency["season"] == season - 1]
    prior, _ = ridge_ratings(prior_eff)
    prior = prior.rename(columns={"off_rating": "prior_off_rating", "def_rating": "prior_def_rating",
                                  "games": "prior_games", "plays": "prior_plays"})
    parts = [
        prior,
        team_talent(src, season),
        blended_recruiting(src, season),
        position_group_talent(src, season),
        returning_production(src, season),
        portal_flow(src, season),
        preseason_qb(src, season),
        coach_features(src, season),
    ]
    for p in parts:
        out = out.merge(p, on="team_id", how="left")
    out.insert(0, "season", season)
    return out


def _allowed(eff: pd.DataFrame) -> pd.DataFrame:
    """What each team's defense allowed: its opponents' offensive rows, re-keyed to the defense."""
    w = eff.assign(w_ppa=eff["off_ppa"] * eff["plays"], w_sr=eff["success_rate"] * eff["plays"])
    a = w.groupby("opponent_id").agg(def_plays=("plays", "sum"), def_ppa_sum=("w_ppa", "sum"),
                                     def_sr_sum=("w_sr", "sum"))
    a["def_ppa_allowed"] = a["def_ppa_sum"] / a["def_plays"]
    a["def_success_allowed"] = a["def_sr_sum"] / a["def_plays"]
    return a[["def_plays", "def_ppa_allowed", "def_success_allowed"]].rename_axis("team_id").reset_index()


def season_to_date_features(src: SourceData, season: int) -> pd.DataFrame:
    """`src` must be as_of the cutoff. Teams with no games yet get NaN observations and zero counts."""
    out = fbs_teams(src, season)
    eff = src.efficiency[(src.efficiency["season"] == season) & src.efficiency["usable"]]
    if not eff.empty:
        w = eff.assign(w_ppa=eff["off_ppa"] * eff["plays"], w_sr=eff["success_rate"] * eff["plays"],
                       w_ex=eff["explosiveness"] * eff["plays"])
        off = w.groupby("team_id").agg(games=("game_id", "nunique"), off_plays=("plays", "sum"),
                                       off_ppa_sum=("w_ppa", "sum"), off_sr_sum=("w_sr", "sum"),
                                       off_ex_sum=("w_ex", "sum"))
        off["off_ppa"] = off["off_ppa_sum"] / off["off_plays"]
        off["off_success"] = off["off_sr_sum"] / off["off_plays"]
        off["off_explosiveness"] = off["off_ex_sum"] / off["off_plays"]
        off = off[["games", "off_plays", "off_ppa", "off_success", "off_explosiveness"]].reset_index()
        ratings, _ = ridge_ratings(eff)
        ratings = ratings[["team_id", "off_rating", "def_rating"]]
        for p in (off, _allowed(eff), ratings):
            out = out.merge(p, on="team_id", how="left")
    else:
        for col in ("off_ppa", "off_success", "off_explosiveness", "def_ppa_allowed", "def_success_allowed",
                    "off_rating", "def_rating", "off_plays", "def_plays"):
            out[col] = np.nan
    out["games"] = out.get("games", pd.Series(0, index=out.index)).fillna(0).astype(int)
    out[["off_plays", "def_plays"]] = out[["off_plays", "def_plays"]].fillna(0.0)
    out = out.merge(current_qb(src, season), on="team_id", how="left")
    out.insert(0, "season", season)
    return out
