"""Roster talent priors: team talent composite, blended recruiting, position-group talent,
returning production and transfer-portal flow.

Recruiting ratings measure potential, not current performance; the efficiency model decides how
much they matter. All functions take an as_of view and a season, and return one row per team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.features.source import SourceData

# The 1st percentile of rated high-school recruits; unrated roster players are mostly walk-ons.
UNRATED = 0.70
# (roster positions, number of starters whose ratings are averaged)
POSITION_GROUPS = {
    "qb": ({"QB"}, 1),
    "ol": ({"OL", "OT", "G", "C"}, 5),
    "skill": ({"RB", "WR", "TE", "FB"}, 5),
    "dl": ({"DL", "DE", "DT", "NT", "EDGE"}, 4),
    "lb": ({"LB", "OLB", "ILB"}, 3),
    "db": ({"DB", "S", "CB"}, 4),
}
RECRUITING_CLASSES = 4
PORTAL_FIRST_SEASON = 2021


def team_talent(src: SourceData, season: int) -> pd.DataFrame:
    t = src.talent[src.talent["season"] == season].dropna(subset=["team_id"])
    out = t[["team_id", "talent"]].copy()
    out["talent"] = out["talent"].where(out["talent"] > 0)
    return out.assign(team_id=out["team_id"].astype("int64"))


def blended_recruiting(src: SourceData, season: int) -> pd.DataFrame:
    """Mean 247 class points over the classes that make up a season's roster (this year's and
    the three before it)."""
    r = src.recruiting_teams
    r = r[r["class_year"].between(season - RECRUITING_CLASSES + 1, season)].dropna(subset=["team_id"])
    out = r.groupby("team_id").agg(recruiting_points=("points", "mean"), recruiting_classes=("points", "size"))
    return out.reset_index().assign(team_id=lambda d: d["team_id"].astype("int64"))


def roster_ratings(src: SourceData, season: int) -> pd.DataFrame:
    """Season roster with each player's recruiting rating (UNRATED when they weren't a rated HS recruit)."""
    ro = src.rosters[src.rosters["season"] == season].dropna(subset=["team_id"])
    rec = src.recruits.dropna(subset=["athlete_id"])
    rec = rec.assign(athlete_id=rec["athlete_id"].astype(str)).sort_values("rating", ascending=False)
    rating = rec.drop_duplicates("athlete_id").set_index("athlete_id")["rating"]
    ro = ro.assign(rating=ro["player_id"].astype(str).map(rating))
    ro["rating"] = ro["rating"].clip(lower=UNRATED, upper=1.0).fillna(UNRATED)
    return ro.assign(team_id=ro["team_id"].astype("int64"))


def position_group_talent(src: SourceData, season: int) -> pd.DataFrame:
    ro = roster_ratings(src, season)
    out = pd.DataFrame({"team_id": ro["team_id"].unique()})
    for group, (positions, k) in POSITION_GROUPS.items():
        sub = ro[ro["position"].isin(positions)]

        def top_k(r: pd.Series, k: int = k) -> float:
            vals = np.sort(r.to_numpy())[::-1][:k]
            return float(np.concatenate([vals, np.full(k - len(vals), UNRATED)]).mean())

        out = out.merge(sub.groupby("team_id")["rating"].agg(top_k).rename(f"talent_{group}"),
                        left_on="team_id", right_index=True, how="left")
        out[f"talent_{group}"] = out[f"talent_{group}"].fillna(UNRATED)
    out["talent_offense"] = out[["talent_qb", "talent_ol", "talent_skill"]].mean(axis=1)
    out["talent_defense"] = out[["talent_dl", "talent_lb", "talent_db"]].mean(axis=1)
    return out


def returning_production(src: SourceData, season: int) -> pd.DataFrame:
    r = src.returning[src.returning["season"] == season].dropna(subset=["team_id"])
    cols = ["pct_ppa", "pct_passing_ppa", "pct_rushing_ppa", "pct_receiving_ppa", "usage"]
    # A few teams report shares outside [0, 1] (negative prior-season PPA totals); clip, don't drop.
    out = r[["team_id", *cols]].copy()
    out[cols] = out[cols].clip(0.0, 1.0)
    out = out.rename(columns={c: f"returning_{c}" for c in cols})
    return out.assign(team_id=out["team_id"].astype("int64"))


def portal_flow(src: SourceData, season: int) -> pd.DataFrame:
    """Transfers into and out of each team for the coming season. NaN before the portal data
    starts (2021): missing, not zero."""
    p = src.portal
    if season < PORTAL_FIRST_SEASON or p.empty:
        return pd.DataFrame(columns=["team_id", "portal_in", "portal_in_stars", "portal_out", "portal_out_stars"])
    p = p[p["season"] == season].assign(stars=lambda d: d["stars"].fillna(2.0))
    inc = p.dropna(subset=["destination_id"]).groupby("destination_id").agg(
        portal_in=("stars", "size"), portal_in_stars=("stars", "sum"))
    out = p.dropna(subset=["origin_id"]).groupby("origin_id").agg(
        portal_out=("stars", "size"), portal_out_stars=("stars", "sum"))
    flow = inc.join(out, how="outer").fillna(0.0)
    flow.index = flow.index.astype("int64")
    return flow.rename_axis("team_id").reset_index()
