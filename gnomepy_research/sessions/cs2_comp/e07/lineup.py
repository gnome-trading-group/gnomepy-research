"""
Who plays for a team at a cutoff. Targets carry no announced lineup, so a team starts from the five of its last map
and the roster moves HLTV recorded after that map (players leaving or joining as active) edit it.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e07.data import LINEUP, NS_PER_DAY


class RosterMoves:
    def __init__(self, moves: pd.DataFrame):
        self._events: dict[int, list[tuple[float, int, int]]] = collections.defaultdict(list)
        day = pd.to_datetime(moves.event_date).to_numpy("datetime64[ns]").astype(np.int64) / NS_PER_DAY
        for d, person, kind, ft, fs, tt, ts in zip(day, moves.person_id, moves.kind, moves.from_team_id, moves.from_status,
                                                   moves.to_team_id, moves.to_status):
            same_slot = ft == tt and fs == ts
            if ft and ft > 0 and fs == "active" and not same_slot:
                self._events[int(ft)].append((float(d), int(person), -1))
            if tt and tt > 0 and ts == "active" and not same_slot:
                self._events[int(tt)].append((float(d), int(person), 1))
        for v in self._events.values():
            v.sort()

    def roster(self, team: int, five: list, last_map_day: float, cutoff_day: float) -> list:
        """`five` are player keys (HLTV ids); moves dated from the last map's day (exclusive) to the cutoff edit it."""
        roster = list(five)
        floor = np.floor(last_map_day)
        for d, person, sign in self._events.get(team, ()):
            if d <= floor or d >= cutoff_day:
                continue
            if sign < 0 and person in roster:
                roster.remove(person)
            elif sign > 0 and person not in roster:
                roster.append(person)
        return roster[-LINEUP:] if len(roster) > LINEUP else roster
