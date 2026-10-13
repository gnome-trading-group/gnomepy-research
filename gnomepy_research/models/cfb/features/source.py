"""The single point-in-time gate for every feature.

Feature code never receives raw tables: it receives `SourceData.as_of(cutoff)`, which keeps only
rows (and fields) that were public strictly before the cutoff. Leak tests perturb everything at or
after a cutoff and assert features don't change.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import datetime

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore

DATASETS = (
    "cfb_games",
    "cfb_team_game_efficiency",
    "cfb_passers",
    "cfb_talent",
    "cfb_recruiting_teams",
    "cfb_recruits",
    "cfb_returning",
    "cfb_portal",
    "cfb_coaches",
    "cfb_rosters",
)

RESULT_DELAY = pd.Timedelta(days=1)
GAME_RESULT_COLUMNS = ("home_points", "away_points", "periods")
PREGAME_COLUMNS = ("home_pregame_elo", "away_pregame_elo")


@dataclass(frozen=True)
class SourceData:
    games: pd.DataFrame
    efficiency: pd.DataFrame
    passers: pd.DataFrame
    talent: pd.DataFrame
    recruiting_teams: pd.DataFrame
    recruits: pd.DataFrame
    returning: pd.DataFrame
    portal: pd.DataFrame
    coaches: pd.DataFrame
    rosters: pd.DataFrame
    cutoff: pd.Timestamp | None = None

    @classmethod
    def load(cls, ds: DatasetStore | None = None, as_of: datetime | None = None) -> SourceData:
        """The sources as known at as_of (default now); ds records the loads for lineage."""
        ds = ds or DatasetStore()
        return cls.from_tables({name: ds.load(name, as_of=as_of) for name in DATASETS})

    @classmethod
    def from_tables(cls, t: dict[str, pd.DataFrame]) -> SourceData:
        games = t["cfb_games"]
        passers = t["cfb_passers"].copy()
        passers["available_at"] = passers["game_id"].map(games.set_index("game_id")["kickoff"]) + RESULT_DELAY
        return cls(
            games=games,
            efficiency=t["cfb_team_game_efficiency"],
            passers=passers,
            talent=t["cfb_talent"],
            recruiting_teams=t["cfb_recruiting_teams"],
            recruits=t["cfb_recruits"],
            returning=t["cfb_returning"],
            portal=t["cfb_portal"],
            coaches=with_coach_availability(t["cfb_coaches"], games),
            rosters=t["cfb_rosters"],
        )

    def as_of(self, cutoff: pd.Timestamp) -> SourceData:
        """Everything public strictly before `cutoff`."""
        if self.cutoff is not None and cutoff > self.cutoff:
            raise ValueError("cannot widen an as_of view")

        def before(df: pd.DataFrame, col: str = "available_at") -> pd.DataFrame:
            return df[df[col] < cutoff]

        games = self.games.copy()
        hidden = games["kickoff"] + RESULT_DELAY >= cutoff
        for col in GAME_RESULT_COLUMNS:
            games[col] = games[col].astype("float64").where(~hidden)
        games["completed"] = games["completed"] & ~hidden
        # CFBD's pregame Elo for a future game already reflects results between the cutoff and that game.
        not_started = games["kickoff"] >= cutoff
        for col in PREGAME_COLUMNS:
            games[col] = games[col].astype("float64").where(~not_started)

        portal = before(self.portal, "origin_available_at").copy()
        unknown_dest = ~(portal["destination_available_at"] < cutoff)
        portal.loc[unknown_dest, ["destination", "destination_id"]] = pd.NA

        coaches = before(self.coaches, "assignment_available_at").copy()
        hidden_results = ~(coaches["results_available_at"] < cutoff)
        for col in ("games", "wins", "losses", "srs", "sp_overall", "sp_offense", "sp_defense", "postseason_rank"):
            coaches[col] = coaches[col].astype("float64").where(~hidden_results)

        return replace(
            self,
            games=games,
            efficiency=before(self.efficiency),
            passers=before(self.passers),
            talent=before(self.talent),
            recruiting_teams=before(self.recruiting_teams),
            recruits=before(self.recruits),
            returning=before(self.returning),
            portal=portal,
            coaches=coaches,
            rosters=before(self.rosters),
            cutoff=cutoff,
        )

    def tables(self) -> dict[str, pd.DataFrame]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "cutoff"}


def season_starts(games: pd.DataFrame) -> pd.Series:
    return games.groupby("season")["kickoff"].min().dt.normalize() - pd.Timedelta(days=1)


def with_coach_availability(coaches: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """The season-start head coach is public at season start. Any other coach of that team in that
    season (interim or mid-season replacement) reveals a firing, so it's only public once the
    season's results are (Feb 1 of the next year).

    The season-start coach is chosen without future information: the coach who also led the team
    the previous season, else the earliest-hired candidate (interims are hired later)."""
    c = coaches.copy()
    prev = set(zip(c["coach_id"], c["team_id"], c["season"] + 1))
    c["continuing"] = [(cid, tid, s) in prev for cid, tid, s in zip(c["coach_id"], c["team_id"], c["season"])]
    c = c.sort_values(["team_id", "season", "continuing", "hire_date", "coach_id"],
                      ascending=[True, True, False, True, True])
    c["season_start_coach"] = ~c.duplicated(["team_id", "season"])
    # Coach history predates the modelled seasons; with no games loaded, Aug 1 is earlier than
    # any season's first kickoff.
    fallback = pd.to_datetime(c["season"].astype(str) + "-08-01", utc=True)
    start = c["season"].map(season_starts(games)).fillna(fallback)
    c["assignment_available_at"] = np.where(c["season_start_coach"], start, c["results_available_at"])
    c["assignment_available_at"] = pd.to_datetime(c["assignment_available_at"], utc=True)
    return c.reset_index(drop=True)
