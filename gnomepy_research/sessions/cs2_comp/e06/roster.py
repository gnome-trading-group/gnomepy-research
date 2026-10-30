"""Who will play for a team, from what is known at a cutoff.

The announced lineup of a series is not in a point-in-time bundle, so each team's five is inferred from its recent HLTV
maps and roster moves: every recent player gets an appearance frequency (recent maps weigh more), a join after the
team's last map lifts a player, a leave or bench after the last map he played for the team drops him.
"""
from __future__ import annotations

import bisect
import collections
from dataclasses import dataclass

import pandas as pd

DEPTH = 14


@dataclass(frozen=True)
class RosterParams:
    freq_decay: float = 0.85
    join_floor: float = 0.8
    leave_cap: float = 0.05


class Rosters:
    def __init__(self, params: RosterParams | None = None):
        self.p = params or RosterParams()
        self.recent: dict[int, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=DEPTH))
        self.days: dict[int, list[float]] = collections.defaultdict(list)
        self.joined: dict[int, dict[int, float]] = collections.defaultdict(dict)
        self.left: dict[int, dict[int, float]] = collections.defaultdict(dict)

    def observe_map(self, team: int, day: float, five: tuple) -> None:
        self.recent[team].append((day, five))
        self.days[team].append(day)

    def maps_within(self, team: int, day: float, window: float) -> int:
        d = self.days.get(team, [])
        return bisect.bisect_left(d, day) - bisect.bisect_left(d, day - window)

    def observe_move(self, row) -> None:
        day = pd.Timestamp(row.event_date).value / 1e9 / 86400.0
        to_team, from_team = int(row.to_team_id or 0), int(row.from_team_id or 0)
        player = int(row.person_id)
        if to_team and row.to_status == "active":
            self.joined[to_team][player] = day
            self.left[to_team].pop(player, None)
        if from_team and row.from_status in ("active", "benched", "substitute") and (to_team != from_team or row.to_status != "active"):
            self.left[from_team][player] = day
            self.joined[from_team].pop(player, None)

    def last_map_day(self, team: int) -> float | None:
        q = self.recent.get(team)
        return q[-1][0] if q else None

    def frequencies(self, team: int) -> dict[int, float]:
        q = self.recent.get(team)
        freq: dict[int, float] = {}
        total = 0.0
        if q:
            for k, (_, five) in enumerate(reversed(q)):
                w = self.p.freq_decay ** k
                total += w
                for pl in five:
                    freq[pl] = freq.get(pl, 0.0) + w
            freq = {pl: v / total for pl, v in freq.items()}
        last = self.last_map_day(team)
        for pl, day in self.joined.get(team, {}).items():
            if last is None or day > last:
                freq[pl] = max(freq.get(pl, 0.0), self.p.join_floor)
        for pl, day in self.left.get(team, {}).items():
            if pl in freq and (last is None or day > self._last_played(team, pl)):
                freq[pl] = min(freq[pl], self.p.leave_cap)
        return freq

    def _last_played(self, team: int, player: int) -> float:
        for day, five in reversed(self.recent[team]):
            if player in five:
                return day
        return float("-inf")

    def expected_five(self, team: int) -> tuple[list[tuple[int, float]], list[int]]:
        """The five most likely players with their frequencies, and the other recent players as replacements."""
        freq = self.frequencies(team)
        order = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        return order[:5], [pl for pl, _ in order[5:] if freq[pl] > self.p.leave_cap]
