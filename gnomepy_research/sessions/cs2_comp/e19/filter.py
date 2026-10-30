"""
Per-player Gaussian skill filter over HLTV and FACEIT maps.

Copied from models/cs2/hltv/player_skill.py and changed in three ways: every observation carries a source, and each
source has its own beta-binomial overdispersion and likelihood weight (FACEIT league maps are played with less at
stake than HLTV-covered series, so they are allowed to say less per round); a player who first appears without an
HLTV team starts from a prior set by the division of that competition rather than from ranking points; and the
filter keeps each team's most recent five, which is the only lineup a pre-match prediction can know when the
announced lineup is not in the data yet.
"""
from __future__ import annotations

import bisect
import collections
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import TIE_TOLERANCE, regulation_map_win

HLTV, FACEIT = "hltv", "faceit"

# Round-logit prior means by ESEA division (and the qualifier/other buckets) for players with no HLTV team at
# their debut; the HLTV prior for an unranked team is `a` (about -0.11), ESEA Open sits well below that.
DIVISION_PRIOR = {"entry": -0.55, "open": -0.50, "intermediate": -0.35, "main": -0.20, "advanced": -0.05,
                  "ecl": 0.0, "qualifier": -0.10, "other": -0.20}
LOW_DIVISIONS = {"entry", "open", "intermediate"}

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()
_GRID = np.linspace(-8.0, 8.0, 3201)


@dataclass(frozen=True)
class FilterConfig:
    tau: float = 0.009319191322975635
    sigma0: float = 0.2536091079975324
    a: float = -0.1081114476105491
    b: float = 0.14696262712175345
    lam: float = 0.0761331162651603
    kappa_max: float = 0.03328732428013525
    c0: float = 7.134081474707468
    rho: float = 0.017231894894383412
    chem_days: int = 180
    faceit_weight: float = 1.0
    # Multiplies faceit_weight for Open, Entry and Intermediate maps, the furthest from HLTV level.
    faceit_low_div_mult: float = 1.0
    faceit_rho: float = 0.017231894894383412
    faceit_sigma0: float = 0.2536091079975324
    faceit_prior_shift: float = 0.0
    # Pricing: map-level and series-level noise added to the filter's own gap variance.
    sigma_map: float = 0.0
    sigma_series: float = 0.0
    # Adjust a team's last five with HLTV roster moves announced after that map and before the cutoff.
    roster_moves: bool = True
    # When a team's last five held a stand-in (HLTV lineups flag), expect the most recent absent regular back.
    standin_fix: bool = False
    # Stack interaction: separate channel weights for thin series (either team with at most thin_maps HLTV maps in
    # the prior thin_days days, or neither team ranked), where FACEIT has the most to say.
    thin_interaction: bool = False
    thin_maps: int = 15
    thin_days: int = 90
    # Team effect on top of the five: what a lineup is worth beyond its players (roles, preparation, form), a
    # random walk per team with faster drift than a player's skill. 0 switches it off.
    team_sigma0: float = 0.0
    team_tau: float = 0.02
    # Ridge channel (ridge.py): a trailing-window refit of player and team strengths every Monday. A lineup there
    # is a sum of five, so the filter's per-player prior sd of sigma0 on the lineup-mean scale is sigma0 / 5 per
    # player: a penalty of 25 / sigma0^2, about 400; the team effect is held to a fraction of that.
    ridge: bool = False
    ridge_window_weeks: int = 52
    ridge_half_life_weeks: float = 26.0
    ridge_lam_player: float = 400.0
    ridge_lam_team: float = 2000.0
    ridge_team: bool = True
    ridge_players: bool = True
    ridge_faceit_weight: float = 0.5
    # Team-only channel: the same trailing-window fit with no players, a decayed Bradley-Terry over map rounds
    # on team ids, for what a lineup is worth beyond its five; sd about 0.2 on the lineup scale.
    team_ridge: bool = False
    team_ridge_lam: float = 25.0
    # Ranking channel: the logit is the gap in log(1 + HLTV ranking points) at the latest ranking before the cutoff.
    rank_channel: bool = False
    # Rating channel: a player's map rating (HLTV 2.0; FACEIT maps through the kills, deaths and ADR terms of the
    # same formula) minus 1 + rating_slope times the filter's pre-map round-logit gap from his side, tracked per
    # player as an exponentially weighted mean with weight rating_alpha; a lineup is the mean of its five and the
    # channel's logit is rating_kappa times the gap. Rounds say how a five did; this says who did it.
    rating_channel: bool = False
    rating_alpha: float = 0.1
    rating_slope: float = 0.25
    rating_kappa: float = 4.0


def division(competition_name: str | None) -> str:
    n = (competition_name or "").lower()
    if "qualifier" in n:
        return "qualifier"
    for key in ("entry", "open", "intermediate", "main", "advanced", "ecl", "challenger"):
        if key in n:
            return "ecl" if key == "challenger" else key
    return "other"


@dataclass
class Lineup:
    players: list[int]
    mus: list[float]
    variances: list[float]
    weights: list[float]
    strength: float
    kappa: float
    total: float


class SkillFilter:
    def __init__(self, cfg: FilterConfig, rankings: pd.DataFrame | None):
        self.cfg = cfg
        self._mu: dict[int, float] = {}
        self._var: dict[int, float] = {}
        self._seen: dict[int, pd.Timestamp] = {}
        self._cov: dict[tuple[int, int], float] = {}
        self.games: collections.Counter = collections.Counter()
        self._five_days: dict[frozenset, collections.deque] = collections.defaultdict(collections.deque)
        self.last_five: dict[int, tuple[int, ...]] = {}
        self.last_five_day: dict[int, pd.Timestamp] = {}
        self.last_standins: dict[int, tuple[int, ...]] = {}
        self.earlier_fives: dict[int, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=12))
        # The pre-update round-logit gap of the last map observed, None when that map was skipped.
        self.last_gap: float | None = None
        self._rank_dates: list[pd.Timestamp] = []
        self._rank_points: list[dict[int, float]] = []
        if rankings is not None and len(rankings):
            r = rankings.assign(date=pd.to_datetime(rankings["date"]))
            for day, g in r.groupby("date", sort=True):
                self._rank_dates.append(pd.Timestamp(day))
                self._rank_points.append(dict(zip(g["team_id"].astype(int), g["points"].astype(float))))

    def points(self, team_id: int, day: pd.Timestamp) -> float:
        i = bisect.bisect_left(self._rank_dates, day) - 1
        return self._rank_points[i].get(team_id, 0.0) if i >= 0 else 0.0

    def prior(self, hltv_team: int | None, day: pd.Timestamp, div: str) -> tuple[float, float]:
        if hltv_team is not None:
            return self.cfg.a + self.cfg.b * math.log1p(self.points(hltv_team, day)), self.cfg.sigma0 ** 2
        return DIVISION_PRIOR[div] + self.cfg.faceit_prior_shift, self.cfg.faceit_sigma0 ** 2

    def _state(self, player: int, hltv_team: int | None, day: pd.Timestamp, div: str) -> tuple[float, float]:
        team = player <= TEAM_ID_BASE
        if player not in self._mu:
            return (0.0, self.cfg.team_sigma0 ** 2) if team else self.prior(hltv_team, day, div)
        idle = max((day - self._seen[player]).days, 0)
        return self._mu[player], self._var[player] + (self.cfg.team_tau if team else self.cfg.tau) ** 2 * idle

    def _five_maps(self, five: tuple[int, ...], day: pd.Timestamp) -> float:
        q = self._five_days.get(frozenset(five))
        if not q:
            return 0.0
        cutoff = day - pd.Timedelta(days=self.cfg.chem_days)
        return float(sum(1 for d in q if d >= cutoff))

    def _weights(self, mus: list[float]) -> list[float]:
        n = len(mus)
        top = max(mus)
        best = next(i for i, m in enumerate(mus) if m >= top - TIE_TOLERANCE)
        return [1 / n + self.cfg.lam * ((1.0 if i == best else 0.0) - 1 / n) for i in range(n)]

    def _block(self, players: list[int], variances: list[float]) -> np.ndarray:
        cov = np.diag(variances)
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p, q = sorted((players[i], players[j]))
                cov[i, j] = cov[j, i] = self._cov.get((p, q), 0.0)
        return cov

    def lineup(self, players: tuple[int, ...], key: int, hltv_team: int | None, day: pd.Timestamp, div: str = "other") -> Lineup:
        states = [self._state(p, hltv_team, day, div) for p in players]
        mus = [s[0] for s in states]
        weights = self._weights(mus)
        five_maps = self._five_maps(players, day) if len(players) == 5 else 0.0
        kappa = self.cfg.kappa_max * (1 - math.exp(-five_maps / self.cfg.c0))
        ids = list(players)
        if self.cfg.team_sigma0 > 0:
            ids.append(team_state_id(key))
            states.append(self._state(ids[-1], None, day, div))
            mus.append(states[-1][0])
            weights.append(1.0)
        strength = sum(w * m for w, m in zip(weights, mus))
        return Lineup(ids, mus, [s[1] for s in states], weights, strength, kappa, strength + kappa)

    def gap(self, a: Lineup, b: Lineup) -> tuple[float, float, np.ndarray, np.ndarray]:
        w = np.array(a.weights + [-x for x in b.weights])
        cov = self._block(a.players + b.players, a.variances + b.variances)
        return a.total - b.total, float(w @ cov @ w), cov, w

    def observe_map(self, day: pd.Timestamp, key_a: int, key_b: int, hltv_a: int | None, hltv_b: int | None,
                    players_a: tuple[int, ...], players_b: tuple[int, ...], rounds_a: int, rounds_b: int,
                    source: str, div: str, standins_a: tuple[int, ...] = (), standins_b: tuple[int, ...] = ()) -> None:
        self.last_gap = None
        if not players_a or not players_b:
            return
        for key, five, standins in ((key_a, players_a, standins_a), (key_b, players_b, standins_b)):
            if len(five) == 5:
                if key in self.last_five and self.last_five[key] != five:
                    self.earlier_fives[key].appendleft(self.last_five[key])
                self.last_five[key], self.last_five_day[key], self.last_standins[key] = five, day, standins
        a = self.lineup(players_a, key_a, hltv_a, day, div)
        b = self.lineup(players_b, key_b, hltv_b, day, div)
        players = a.players + b.players
        if len(set(players)) != len(players):
            return
        d, s2, cov, w = self.gap(a, b)
        self.last_gap = d
        n = rounds_a + rounds_b
        pi = 1 / (1 + math.exp(-d))
        rho, weight = (self.cfg.faceit_rho, self.cfg.faceit_weight) if source == FACEIT else (self.cfg.rho, 1.0)
        if source == FACEIT and div in LOW_DIVISIONS:
            weight *= self.cfg.faceit_low_div_mult
        inflation = (1 + (n - 1) * rho) / weight
        grad, curv = (rounds_a - n * pi) / inflation, n * pi * (1 - pi) / inflation
        cw = cov @ w
        denom = 1 + curv * s2
        mus = np.array(a.mus + b.mus) + cw * grad / denom
        cov = cov - np.outer(cw, cw) * curv / denom
        for i, p in enumerate(players):
            self._mu[p] = float(mus[i])
            self._var[p] = max(1e-6, float(cov[i, i]))
            self._seen[p] = day
            self.games[p] += 1
            for j in range(i + 1, len(players)):
                self._cov[tuple(sorted((p, players[j])))] = float(cov[i, j])
        cutoff = day - pd.Timedelta(days=self.cfg.chem_days)
        for five in (players_a, players_b):
            if len(five) == 5:
                q = self._five_days[frozenset(five)]
                q.append(day)
                while q and q[0] < cutoff:
                    q.popleft()


def regulars_back(f: SkillFilter, key: int, five: tuple[int, ...]) -> tuple[int, ...]:
    """The last five with each flagged stand-in replaced by the most recently seen regular who is absent."""
    standins = [p for p in f.last_standins.get(key, ()) if p in five]
    if not standins:
        return five
    players = list(five)
    absent = [p for earlier in f.earlier_fives.get(key, ()) for p in earlier if p not in players]
    for s in standins:
        if not absent:
            break
        players[players.index(s)] = absent.pop(0)
    return tuple(players)


class SeriesPricer:
    """Round-logit gap ~ N(d, s2) to P(bo3): series-shared uncertainty integrated outside the bo3 polynomial,
    map noise inside it, on a grid so pricing is a lookup."""

    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        table = regulation_map_win(_GRID[:, None] + cfg.sigma_map * _GH_X[None, :]) @ _GH_W
        self._table = table

    def series(self, d: float, s2: float) -> float:
        sd = math.sqrt(max(s2, 0.0) + self.cfg.sigma_series ** 2)
        p = np.interp(d + sd * _GH_X, _GRID, self._table)
        return float((p * p * (3 - 2 * p)) @ _GH_W)


def placeholder_five(team_key: int, n: int = 5) -> tuple[int, ...]:
    """Ids no player ever has, so a team with no map in the data prices from its prior alone."""
    return tuple(-(10 ** 15) - abs(int(team_key)) * 10 - k for k in range(n))


# FACEIT-derived player ids come from 15 hex digits, so they stay above -2^60; team states live below -2^61.
TEAM_ID_BASE = -(2 ** 61)


def team_state_id(team_key: int) -> int:
    return TEAM_ID_BASE - abs(int(team_key))


ADD_KINDS = {"join", "transfer", "activated", "loan_return", "loan"}
REMOVE_KINDS = {"leave", "transfer", "benched", "to_coach", "to_substitute", "loan"}
MAX_LINEUP = 6


class RosterMoves:
    """HLTV roster moves as per-team dated add/remove events, applied to a last-known five up to a cutoff. A team
    with a join and no leave yet is priced as the mean of six: which player sits is not known pre-match."""

    def __init__(self, moves: pd.DataFrame):
        self._events: dict[int, list[tuple[pd.Timestamp, str, int]]] = collections.defaultdict(list)
        m = moves[moves.person_role == "player"].copy()
        m["event_date"] = pd.to_datetime(m.event_date)
        m = m.sort_values(["event_date", "person_id"], kind="mergesort")
        for r in m.itertuples():
            if r.kind in REMOVE_KINDS and pd.notna(r.from_team_id) and int(r.from_team_id) > 0:
                self._events[int(r.from_team_id)].append((r.event_date, "remove", int(r.person_id)))
            if r.kind in ADD_KINDS and r.to_status == "active" and pd.notna(r.to_team_id) and int(r.to_team_id) > 0:
                self._events[int(r.to_team_id)].append((r.event_date, "add", int(r.person_id)))

    def adjust(self, team: int, five: tuple[int, ...], since: pd.Timestamp, before: pd.Timestamp) -> tuple[int, ...]:
        players = list(five)
        for day, op, p in self._events.get(team, ()):
            if day < since or day >= before:
                continue
            if op == "remove" and p in players:
                players.remove(p)
            elif op == "add" and p not in players:
                players.append(p)
        players = players[-MAX_LINEUP:]
        if len(players) < 5:
            players += list(placeholder_five(team, 5 - len(players)))
        return tuple(players)
