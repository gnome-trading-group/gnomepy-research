"""
Roster stability and recent moves, per team, from the announced lineup.

Player skill already prices who is playing and how well that five knows each
other. What it cannot see is the roster's recent history as HLTV logs it:
whether the announced players are the team's regulars, how many of them joined
in the last month, whether core players or the coach just left. These carry the
disruption a market reads off the news before any results arrive.

"Core" is the players in at least CORE_SHARE of the team's last CORE_WINDOW
series; usage is each announced player's share of those series. Moves come from
cs2_roster_moves and count from their HLTV event date.

The lineup is the announced one (cs2_match_lineups), public before the match,
and it is also what each series folds into the team's history. A side with no
announced lineup gets NaN.
"""
from __future__ import annotations

import collections
import logging

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.pit import SeriesBatch, pit_scan

logger = logging.getLogger(__name__)

LINEUPS_AUX = "lineups"
CORE_WINDOW, CORE_SHARE = 20, 0.6
_JOIN_KINDS = ("join", "transfer", "loan_return", "activated")

_PER_TEAM = ("standins", "standin_coach", "core_present", "core_missing", "new_faces", "usage_mean", "usage_min",
             "rotation_30", "days_since_move", "moves_30", "moves_90", "recent_joiners", "core_lost_30",
             "coach_change_90")
_DIFFS = ("standins", "core_present", "core_missing", "new_faces", "usage_mean", "usage_min", "rotation_30",
          "days_since_move", "moves_30", "recent_joiners", "core_lost_30")
ROSTER_FEATURES = (
    [f"team_{side}_roster_{k}" for side in "ab" for k in _PER_TEAM]
    + [f"roster_{k}_diff" for k in _DIFFS]
)


class _RosterAccumulator:
    def __init__(self, moves: pd.DataFrame | None):
        self._series: dict[int, list[tuple[pd.Timestamp, frozenset]]] = collections.defaultdict(list)
        self._events: dict[int, list[tuple]] = collections.defaultdict(list)
        if moves is not None and len(moves):
            m = moves.assign(known=pd.to_datetime(moves["event_date"]).dt.normalize(),
                             from_team_id=moves["from_team_id"].fillna(0).astype(int),
                             to_team_id=moves["to_team_id"].fillna(0).astype(int))
            for mv in m.itertuples():
                for team in {mv.from_team_id, mv.to_team_id} - {0}:
                    self._events[team].append((mv.known, mv.kind, mv.person_id, mv.person_role, mv.from_team_id,
                                               mv.to_team_id, mv.from_status, mv.to_status))
            for events in self._events.values():
                events.sort(key=lambda e: e[0])

    @staticmethod
    def _announced(batch: SeriesBatch) -> dict[int, list]:
        by_team = collections.defaultdict(list)
        for r in batch.aux_rows(LINEUPS_AUX):
            by_team[int(r.team_id)].append(r)
        return by_team

    def _team(self, team_id: int, rows: list, day: pd.Timestamp) -> dict:
        players = [int(r.player_id) for r in rows]
        past = self._series[team_id]
        recent = past[-CORE_WINDOW:]
        usage = {p: sum(p in s for _, s in recent) / len(recent) for p in players} if recent else {}
        core = ({p for p in set().union(*(s for _, s in recent))
                 if sum(p in s for _, s in recent) >= CORE_SHARE * len(recent)}
                if len(recent) >= 5 else set())
        last30 = [s for d, s in past if (day - d).days <= 30]

        events = [e for e in self._events.get(team_id, []) if e[0] < day]
        e30 = [e for e in events if (day - e[0]).days <= 30]
        e90 = [e for e in events if (day - e[0]).days <= 90]
        joined_on = {}
        for e in events:
            if e[5] == team_id and e[7] in ("active", "substitute") and e[3] == "player":
                joined_on.setdefault(e[2], e[0])
                if e[1] in _JOIN_KINDS:
                    joined_on[e[2]] = e[0]
        coaches = {e[2] for e in events if e[3] == "coach" or e[7] == "coach"}
        standin_ids = [int(r.player_id) for r in rows if r.is_standin == 1]
        present = sum(p in core for p in players)
        return {
            "standins": float(len(standin_ids)),
            "standin_coach": float(any(p in coaches for p in standin_ids)),
            "core_present": float(present) if core else np.nan,
            "core_missing": float(len(core) - present) if core else np.nan,
            "new_faces": float(sum(usage[p] == 0 for p in players)) if recent else np.nan,
            "usage_mean": float(np.mean(list(usage.values()))) if recent else np.nan,
            "usage_min": float(min(usage.values())) if recent else np.nan,
            "rotation_30": float(len(set().union(*last30))) if last30 else np.nan,
            "days_since_move": float((day - events[-1][0]).days) if events else np.nan,
            "moves_30": float(len(e30)),
            "moves_90": float(len(e90)),
            "recent_joiners": float(sum(1 for p in players if p in joined_on and (day - joined_on[p]).days <= 30)),
            "core_lost_30": float(sum(1 for e in e30 if e[4] == team_id and e[6] == "active"
                                      and (e[7] != "active" or e[5] != team_id)
                                      and e[3] == "player" and e[2] in core)),
            "coach_change_90": float(any(e[3] == "coach" or "coach" in (e[6] or "", e[7] or "") for e in e90)),
        }

    def snapshot(self, batch: SeriesBatch) -> dict:
        first = batch.records[0]
        announced = self._announced(batch)
        row = {}
        for side in "ab":
            team_id = int(getattr(first, f"team_{side}_id"))
            rows = announced.get(team_id)
            feats = self._team(team_id, rows, batch.match_date) if rows else dict.fromkeys(_PER_TEAM, np.nan)
            row.update({f"team_{side}_roster_{k}": v for k, v in feats.items()})
        for k in _DIFFS:
            row[f"roster_{k}_diff"] = row[f"team_a_roster_{k}"] - row[f"team_b_roster_{k}"]
        return row

    def update(self, batch: SeriesBatch) -> None:
        first = batch.records[0]
        announced = self._announced(batch)
        teams = (int(first.team_a_id), int(first.team_b_id))
        # A series is folded in only once both lineups are known, as the features were researched.
        if not all(t in announced for t in teams):
            return
        for t in teams:
            self._series[t].append((batch.match_date, frozenset(int(r.player_id) for r in announced[t])))


def compute_roster_features(
    history: pd.DataFrame,
    lineups: pd.DataFrame | None = None,
    roster_moves: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pre-series roster features for every map row, keyed (match_id, map_name)."""
    if lineups is None or lineups.empty:
        logger.warning("cs2_match_lineups absent — roster features unavailable, emitting no rows")
        return pd.DataFrame(columns=["match_id", "map_name"] + ROSTER_FEATURES)
    return pit_scan(history, _RosterAccumulator(roster_moves), aux={LINEUPS_AUX: lineups})
