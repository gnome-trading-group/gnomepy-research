"""Head-coach features at season start.

Raw win percentage mostly measures the program, so a coach's record is scored as SRS above what
the team's previous-season SRS predicted (regression to the mean included), averaged over the
coach's completed seasons and shrunk toward zero for coaches with few of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.features.source import SourceData

# Seasons of evidence at which a coach's own residual gets half weight.
COACH_SHRINK_SEASONS = 3.0


def team_srs(coaches: pd.DataFrame) -> pd.DataFrame:
    """One SRS per visible team-season (every coach row of a team-season carries the same SRS)."""
    s = coaches.dropna(subset=["srs", "team_id"]).drop_duplicates(["team_id", "season"])
    return s[["team_id", "season", "srs"]]


def srs_residuals(coaches: pd.DataFrame) -> pd.DataFrame:
    """Per visible coach-season: SRS minus the linear prediction from the team's previous SRS."""
    srs = team_srs(coaches)
    prev = srs.assign(season=srs["season"] + 1).rename(columns={"srs": "prev_srs"})
    pairs = srs.merge(prev, on=["team_id", "season"])
    if len(pairs) < 20:
        return pd.DataFrame(columns=["coach_id", "season", "resid"])
    slope, intercept = np.polyfit(pairs["prev_srs"], pairs["srs"], 1)
    pairs["expected"] = intercept + slope * pairs["prev_srs"]
    c = coaches.dropna(subset=["srs"]).merge(pairs[["team_id", "season", "expected"]], on=["team_id", "season"])
    c["resid"] = c["srs"] - c["expected"]
    return c[["coach_id", "season", "resid", "games"]]


def coach_features(src: SourceData, season: int) -> pd.DataFrame:
    c = src.coaches
    start = c[c["season_start_coach"] & (c["season"] == season)].dropna(subset=["team_id"])
    prev_start = c[c["season_start_coach"] & (c["season"] == season - 1)].set_index("team_id")["coach_id"]
    past = c[c["season"] < season]
    resid = srs_residuals(c)

    rows = []
    for r in start.itertuples():
        mine = past[past["coach_id"] == r.coach_id]
        here = mine[mine["team_id"] == r.team_id]
        rr = resid[resid["coach_id"] == r.coach_id]
        # Interim coaches share a season's SRS; weight each season by games coached.
        if len(rr) and rr["games"].fillna(0).sum() > 0:
            w = rr["games"].fillna(0).to_numpy(float)
            mean_resid = float(np.average(rr["resid"], weights=w))
        else:
            mean_resid = 0.0
        n = rr["season"].nunique()
        rows.append({
            "team_id": int(r.team_id),
            "coach_id": r.coach_id,
            "coach_new": bool(prev_start.get(r.team_id) != r.coach_id),
            "coach_tenure": int(here["season"].nunique()),
            "coach_hc_seasons": int(mine["season"].nunique()),
            "coach_srs_resid": mean_resid * n / (n + COACH_SHRINK_SEASONS),
        })
    return pd.DataFrame(rows)
