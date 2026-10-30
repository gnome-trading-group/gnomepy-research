from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Params:
    base: float = 1500.0
    scale: float = 400.0
    k_team: float = 32.0
    k_player: float = 12.0
    k_faceit_mult: float = 0.6
    faceit_level_shift: float = 0.0
    provisional_games: int = 20
    provisional_k_mult: float = 1.6
    score_clip: float = 0.05
    overlap_power: float = 2.0
    shrink_games: float = 12.0
    calibration: float = 1.0


@dataclass
class _Entry:
    value: float
    games: int = 0


@dataclass
class _TeamState:
    rating: _Entry
    recent: list = field(default_factory=list)


def _expected(ra: float, rb: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))


def _shrink(games: float, k: float) -> float:
    if k <= 0:
        return 1.0
    return games / (games + k)


def _k(entry: _Entry, base_k: float, p: Params) -> float:
    if entry.games < p.provisional_games:
        return base_k * p.provisional_k_mult
    return base_k


class Ladder:
    """Team and player Elo ladders over one chronological stream of map results.

    Teams are keyed by an HLTV id where linked, otherwise a FACEIT-only key. Player ladders are updated
    from the team-level expectation, so a player keeps the level of the teams they played for when they
    move: the roster strength of a new lineup is the mean of its players' ratings.
    """

    RECENT_MAPS = 10

    def __init__(self, params: Params):
        self.p = params
        self.teams: dict[str, _TeamState] = {}
        self.players: dict[str, _Entry] = {}
        self.offset_sum = 0.0
        self.offset_n = 0

    def _team(self, key: str) -> _TeamState:
        st = self.teams.get(key)
        if st is None:
            st = _TeamState(rating=_Entry(self.p.base))
            self.teams[key] = st
        return st

    def _player(self, key: str, seed: float) -> _Entry:
        e = self.players.get(key)
        if e is None:
            e = _Entry(seed)
            self.players[key] = e
        return e

    def roster_rating(self, lineup: list[str]) -> tuple[float | None, float]:
        if not lineup:
            return None, 0.0
        entries = [self.players[k] for k in lineup if k in self.players and self.players[k].games > 0]
        if not entries:
            return None, 0.0
        return float(np.mean([e.value for e in entries])), float(np.mean([e.games for e in entries]))

    def overlap(self, key: str, lineup: list[str]) -> float:
        st = self.teams.get(key)
        if st is None or not lineup:
            return 0.0
        seen = set()
        for lu in st.recent:
            seen.update(lu)
        return sum(1 for p in lineup if p in seen) / len(lineup)

    def effective(self, key: str, lineup: list[str]) -> tuple[float, float]:
        """Team rating blended with the roster rating by how much of the announced lineup the team fielded recently.

        The blend weight is the overlap raised to a power, so a team that changed two players sits mostly on its
        own history while a fresh five with no shared history starts from what its players already proved.
        """
        st = self.teams.get(key)
        team_r = st.rating.value if st else self.p.base
        team_n = st.rating.games if st else 0
        if not lineup:
            return team_r, float(team_n)
        w = self.overlap(key, lineup) ** self.p.overlap_power if st else 0.0
        roster_r, roster_n = self.roster_rating(lineup)
        if roster_r is None:
            return team_r, float(team_n)
        roster_as_team = roster_r + self._offset()
        rating = w * team_r + (1.0 - w) * roster_as_team
        games = w * team_n + (1.0 - w) * roster_n
        return rating, games

    def _offset(self) -> float:
        if self.offset_n == 0:
            return 0.0
        return self.offset_sum / self.offset_n

    def price_map(self, key_a: str, lineup_a: list[str], key_b: str, lineup_b: list[str],
                  faceit_map: bool) -> tuple[float, float, float]:
        ra, na = self.effective(key_a, lineup_a)
        rb, nb = self.effective(key_b, lineup_b)
        shrink_a = _shrink(na, self.p.shrink_games)
        shrink_b = _shrink(nb, self.p.shrink_games)
        ra = self.p.base + shrink_a * (ra - self.p.base)
        rb = self.p.base + shrink_b * (rb - self.p.base)
        shift = self.p.faceit_level_shift if faceit_map else 0.0
        return _expected(ra + shift, rb, self.p.scale), ra, rb

    def update_map(self, key_a: str, lineup_a: list[str], key_b: str, lineup_b: list[str],
                   share_a: float, faceit_map: bool) -> None:
        p = self.p
        share = min(max(share_a, p.score_clip), 1.0 - p.score_clip)
        sa, sb = self._team(key_a), self._team(key_b)
        ta, tb = sa.rating.value, sb.rating.value
        shift = p.faceit_level_shift if faceit_map else 0.0
        exp_a = _expected(ta + shift, tb, p.scale)
        mult = p.k_faceit_mult if faceit_map else 1.0
        ka = _k(sa.rating, p.k_team, p) * mult
        kb = _k(sb.rating, p.k_team, p) * mult
        sa.rating.value = ta + ka * (share - exp_a)
        sb.rating.value = tb + kb * ((1.0 - share) - (1.0 - exp_a))
        sa.rating.games += 1
        sb.rating.games += 1
        sa.recent.append(list(lineup_a))
        sb.recent.append(list(lineup_b))
        del sa.recent[:-self.RECENT_MAPS]
        del sb.recent[:-self.RECENT_MAPS]

        for lineup, team_exp, team_r_before, sign in (
            (lineup_a, exp_a, ta, 1.0),
            (lineup_b, 1.0 - exp_a, tb, -1.0),
        ):
            if not lineup:
                continue
            players = [self._player(k, team_r_before) for k in lineup]
            roster_r = float(np.mean([e.value for e in players]))
            self.offset_sum += team_r_before - roster_r
            self.offset_n += 1
            s = share if sign > 0 else 1.0 - share
            for e in players:
                kp = _k(e, p.k_player, p) * mult
                e.value += kp * (s - team_exp)
                e.games += 1


def series_probability(map_p: float, best_of: int) -> float:
    if best_of <= 1:
        return map_p
    need = best_of // 2 + 1
    total = 0.0
    for j in range(need):
        total += math.comb(need - 1 + j, j) * (map_p ** need) * ((1.0 - map_p) ** j)
    return total
