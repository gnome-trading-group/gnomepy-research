"""Build the point-in-time feature tables for every historical cutoff.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.features.snapshots build 2018 2026
    poetry run python -m gnomepy_research.sessions.cfb_win_probability.features.snapshots publish 2018 2026

Tables:
- cfb_team_preseason_features: (season, team) as of the season's first cutoff
- cfb_team_week_features: (cutoff, team) season-to-date
- cfb_pregame_features: one row per game: its week's cutoff, both teams' preseason and
  season-to-date features, venue context, and `label_*` outcome columns (never features)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.features.cutoffs import WEEK_KEYS, week_cutoffs
from gnomepy_research.sessions.cfb_win_probability.features.source import DATASET_VERSIONS, SourceData
from gnomepy_research.sessions.cfb_win_probability.features.team_features import (
    preseason_features,
    season_to_date_features,
)

FEATURE_VERSION = "pit-1"
SESSION = "cfb_win_probability"
GAME_CONTEXT = ["game_id", "season", "week", "season_type", "kickoff", "neutral_site", "conference_game",
                "home_id", "away_id", "home_classification", "away_classification"]


def build_preseason(src: SourceData, seasons: list[int]) -> pd.DataFrame:
    cuts = week_cutoffs(src.games)
    frames = []
    for season in seasons:
        first = cuts.loc[cuts["season"] == season, "cutoff"].min()
        frames.append(preseason_features(src.as_of(first), season).assign(cutoff=first))
    return pd.concat(frames, ignore_index=True)


def build_team_week(src: SourceData, seasons: list[int]) -> pd.DataFrame:
    cuts = week_cutoffs(src.games)
    frames = []
    for c in cuts[cuts["season"].isin(seasons)].itertuples():
        f = season_to_date_features(src.as_of(c.cutoff), c.season)
        frames.append(f.assign(season_type=c.season_type, week=c.week, cutoff=c.cutoff))
    out = pd.concat(frames, ignore_index=True)
    return out[["cutoff", *WEEK_KEYS, *[c for c in out.columns if c not in ("cutoff", *WEEK_KEYS)]]]


def build_pregame(src: SourceData, preseason: pd.DataFrame, team_week: pd.DataFrame) -> pd.DataFrame:
    seasons = preseason["season"].unique()
    g = src.games[src.games["season"].isin(seasons)]
    g = g[(g["home_classification"] == "fbs") | (g["away_classification"] == "fbs")]
    g = g.merge(week_cutoffs(src.games), on=WEEK_KEYS)
    out = g[[*GAME_CONTEXT, "cutoff"]].copy()

    pre = preseason.drop(columns=["cutoff"])
    tw = team_week.drop(columns=["season_type", "week"])
    for side in ("home", "away"):
        p = pre.rename(columns={c: f"{side}_pre_{c}" for c in pre.columns if c not in ("season", "team_id")})
        out = out.merge(p.rename(columns={"team_id": f"{side}_id"}), on=["season", f"{side}_id"], how="left")
        t = tw.rename(columns={c: f"{side}_std_{c}" for c in tw.columns if c not in ("season", "team_id", "cutoff")})
        out = out.merge(t.rename(columns={"team_id": f"{side}_id"}), on=["season", "cutoff", f"{side}_id"], how="left")

    labels = src.games.set_index("game_id")
    out["label_completed"] = out["game_id"].map(labels["completed"])
    out["label_home_points"] = out["game_id"].map(labels["home_points"])
    out["label_away_points"] = out["game_id"].map(labels["away_points"])
    out["label_home_win"] = (out["label_home_points"] > out["label_away_points"]).astype("Int64").where(
        out["label_completed"])
    out["feature_version"] = FEATURE_VERSION
    return out.sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def build_all(src: SourceData, seasons: list[int]) -> dict[str, pd.DataFrame]:
    preseason = build_preseason(src, seasons)
    team_week = build_team_week(src, seasons)
    pregame = build_pregame(src, preseason, team_week)
    for df in (preseason, team_week):
        df["feature_version"] = FEATURE_VERSION
    return {"cfb_team_preseason_features": preseason, "cfb_team_week_features": team_week,
            "cfb_pregame_features": pregame}


DESCRIPTIONS = {
    "cfb_team_preseason_features": "Point-in-time preseason features per (season, team): prior-season ridge "
                                   "ratings, talent, recruiting, position groups, returning production, portal, QB, coach",
    "cfb_team_week_features": "Point-in-time season-to-date features per (weekly cutoff, team): PPA, success, "
                              "allowed, ridge-adjusted O/D, current QB",
    "cfb_pregame_features": "One row per game: weekly cutoff, both teams' preseason + season-to-date features, "
                            "venue context, label_* outcomes",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "publish"])
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    args = ap.parse_args()
    src = SourceData.load()
    tables = build_all(src, list(range(args.first, args.last + 1)))
    for name, df in tables.items():
        print(f"{name}: {len(df):,} rows x {df.shape[1]} cols")
    if args.cmd == "publish":
        store = DatasetStore()
        provenance = json.dumps({"feature_version": FEATURE_VERSION, "sources": DATASET_VERSIONS})
        for name, df in tables.items():
            print(store.publish(df, name, description=f"{DESCRIPTIONS[name]} | {provenance}",
                                producing_session=SESSION))


if __name__ == "__main__":
    main()
