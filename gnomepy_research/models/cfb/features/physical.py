"""Point-in-time physical and talent-gap features: talent composite, blue-chip ratio and
offensive/defensive line weights, one row per FBS team-season."""
from __future__ import annotations

import pandas as pd

from gnomepy_research.models.cfb.features.source import SourceData, season_starts
from gnomepy_research.models.cfb.features.team_talent import team_talent

# Rosters and talent are stamped with the season-start date itself, which the strict gate would drop.
SEASON_START_GRACE = pd.Timedelta(days=1)
BLUE_CHIP_STARS = 4
BLUE_CHIP_CLASSES = 4
MIN_SIGNEES = 20
OL_POSITIONS = {"OL", "OT", "G", "C"}
OL_STARTERS = 5
DL_POSITIONS = {"DL", "DT", "DE", "NT", "EDGE"}
DL_STARTERS = 4
FEATURES = ["talent", "blue_chip_ratio", "ol_weight", "dl_weight"]


def blue_chip_ratio(recruits: pd.DataFrame, season: int) -> pd.Series:
    r = recruits[(recruits["recruit_type"] == "HighSchool") & recruits["class_year"].between(season - BLUE_CHIP_CLASSES + 1, season)]
    r = r.dropna(subset=["team_id", "stars"])
    g = r.assign(blue=r["stars"] >= BLUE_CHIP_STARS).groupby(r["team_id"].astype("int64"))["blue"]
    return (g.mean()).where(g.size() >= MIN_SIGNEES).rename("blue_chip_ratio")


def top_weight(rosters: pd.DataFrame, positions: set[str], k: int) -> pd.Series:
    r = rosters[rosters["position"].isin(positions)].dropna(subset=["team_id"])
    r = r.assign(weight=r["weight"].where(r["weight"] > 0), team_id=r["team_id"].astype("int64")).dropna(subset=["weight"])
    return r.groupby("team_id")["weight"].agg(lambda w: w.nlargest(k).mean())


def physical_features(src: SourceData, season: int) -> pd.DataFrame:
    pit = src.as_of(season_starts(src.games)[season] + SEASON_START_GRACE)
    ro = pit.rosters[pit.rosters["season"] == season]
    out = team_talent(pit, season).set_index("team_id")[["talent"]]
    out = out.join(blue_chip_ratio(pit.recruits, season), how="outer")
    out = out.join(top_weight(ro, OL_POSITIONS, OL_STARTERS).rename("ol_weight"), how="outer")
    out = out.join(top_weight(ro, DL_POSITIONS, DL_STARTERS).rename("dl_weight"), how="outer")
    return out.rename_axis("team_id").reset_index()


def physical_table(src: SourceData, seasons: list[int]) -> pd.DataFrame:
    return pd.concat([physical_features(src, s).assign(season=s) for s in seasons], ignore_index=True)[["season", "team_id", *FEATURES]]


def zscore_by_season(table: pd.DataFrame) -> pd.DataFrame:
    z = table.copy()
    for c in FEATURES:
        g = z.groupby("season")[c]
        z[c] = ((z[c] - g.transform("mean")) / g.transform("std")).fillna(0.0)
    return z
