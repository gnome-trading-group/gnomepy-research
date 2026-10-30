"""One causal pass over every map before the last cutoff, with a snapshot of lineup gaps at each weekly cutoff.

A cutoff is a Monday 00:00 UTC. Maps with a time before it have updated the filter; the series of the week after it are
priced from that state with each team's most recent fives (their own announced lineups are not known that early).
The same code prices historical series, whose outcomes are known later and train the calibration, and the targets.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import map_win_prob
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e10.data import (
    LINEUP, PlayerIndex, day_number, faceit_maps, faceit_stats, first_skill_level, hltv_maps, hltv_rating_dev, new_index,
)
from gnomepy_research.sessions.cs2_comp.e10.ratings import fit_rating_maps
from gnomepy_research.sessions.cs2_comp.e10.skill import NEWCOMER, SkillConfig, SkillFilter

HLTV, FACEIT = 0, 1

TRAIN_START = pd.Timestamp("2024-07-01")
WEEK = pd.Timedelta(days=7)
DAY_NS = 86_400 * 10 ** 9
TOGETHER_DAYS = 60


@dataclass(frozen=True)
class RunConfig:
    hltv: SkillConfig = field(default_factory=lambda: SkillConfig(tau=0.014, gamma=0.5, sigma_r=0.8))
    faceit_skill: SkillConfig = field(default_factory=lambda: SkillConfig(tau=0.02, sigma0=0.30, a=-0.25, level_slope=0.06, gamma=0.5, sigma_r=0.8))
    use_faceit: bool = True
    faceit_rating: bool = True
    k_series: int = 5
    games_half: float = 20.0
    shift_min_games: int = 20
    thin_games: int = 20
    use_moves: bool = True
    decay: float = 0.4
    faceit_offset_var: float = 0.04
    standin_weight: float = 0.3


class Event(NamedTuple):
    time: int
    source: int
    day: int
    a: np.ndarray
    b: np.ndarray
    ra: int
    rb: int
    dev_a: np.ndarray | None
    dev_b: np.ndarray | None
    team_a: int
    team_b: int
    match_id: object
    fa: tuple | None
    fb: tuple | None


class Rankings:
    def __init__(self, rankings: pd.DataFrame):
        r = rankings.assign(date=pd.to_datetime(rankings["date"])).sort_values("date")
        self._h = {int(t): (g.date.to_numpy(), g.points.to_numpy(float)) for t, g in r.groupby("team_id")}

    def points(self, team: int, day: pd.Timestamp) -> float:
        h = self._h.get(int(team))
        if h is None:
            return 0.0
        i = np.searchsorted(h[0], np.datetime64(day), side="left") - 1
        return float(h[1][i]) if i >= 0 else 0.0


def roster_moves(data_dir: Path, before: pd.Timestamp) -> dict[int, list[tuple[np.datetime64, int, int]]]:
    """Per team, the dated joins (+1) and departures (-1) of players, from the transfer log."""
    m = load(data_dir, "hltv/cs2_roster_moves", before)
    m = m[m.person_role == "player"]
    out: dict[int, list] = collections.defaultdict(list)
    leaving = {"leave", "transfer", "benched", "loan", "to_substitute", "to_coach", "loan_return"}
    joining = {"join", "transfer", "activated", "loan", "loan_return"}
    for r in m.itertuples(index=False):
        date = np.datetime64(pd.Timestamp(r.event_date))
        if r.kind in leaving and r.from_team_id and r.from_team_id > 0:
            out[int(r.from_team_id)].append((date, int(r.person_id), -1))
        if r.kind in joining and r.to_team_id and r.to_team_id > 0:
            out[int(r.to_team_id)].append((date, int(r.person_id), 1))
    for v in out.values():
        v.sort(key=lambda x: x[0])
    return out


def _team_id(value) -> int:
    return -1 if pd.isna(value) else int(value)


def _arr(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def build_events(data_dir: Path, before: pd.Timestamp, index: PlayerIndex, cfg: RunConfig):
    hm = hltv_maps(data_dir, before)
    ratings = hltv_rating_dev(data_dir, before) if cfg.hltv.gamma > 0 else None
    dev_lookup: dict = {}
    if ratings is not None:
        for (m, n), g in ratings.groupby(["match_id", "map_name"]):
            dev_lookup[(m, n)] = dict(zip(g.player_id, g.dev))
    events: list[Event] = []
    days = day_number(hm.time)
    for r, day in zip(hm.itertuples(index=False), days):
        pa, pb = [int(x) for x in r.team_a_player_ids], [int(x) for x in r.team_b_player_ids]
        if len(set(pa + pb)) != 2 * LINEUP:
            continue
        d = dev_lookup.get((r.match_id, r.map_name))
        dev_a = np.array([d[p] for p in pa]) if d and all(p in d for p in pa) else None
        dev_b = np.array([d[p] for p in pb]) if d and all(p in d for p in pb) else None
        events.append(Event(int(r.time.value), HLTV, int(day), _arr([index.from_hltv(p) for p in pa]),
                            _arr([index.from_hltv(p) for p in pb]), int(r.team_a_score), int(r.team_b_score), dev_a, dev_b,
                            int(r.team_a_id), int(r.team_b_id), r.match_id, None, None))
    info = {"hltv_maps": len(events)}
    if cfg.use_faceit:
        fm = faceit_maps(data_dir, before)
        dev_f: dict = {}
        if cfg.faceit_rating:
            maps, fit_info = fit_rating_maps(data_dir)
            info["rating_fit"] = fit_info
            st = faceit_stats(data_dir, before, set(fm.match_id))
            rounds = fm.assign(rounds=fm.team_a_score + fm.team_b_score)[["match_id", "map_name", "rounds"]].drop_duplicates(["match_id", "map_name"])
            st = st.merge(rounds, on=["match_id", "map_name"])
            st = st[st.rounds > 0].reset_index(drop=True)
            st["dev"] = maps.rating_dev(st, st.rounds).to_numpy()
            for (m, n), g in st.groupby(["match_id", "map_name"]):
                dev_f[(m, n)] = dict(zip(g.faceit_id, g.dev))
            del st
        fdays = day_number(fm.time)
        for r, day in zip(fm.itertuples(index=False), fdays):
            fa, fb = tuple(r.team_a_faceit_ids), tuple(r.team_b_faceit_ids)
            if len(set(fa + fb)) != 2 * LINEUP:
                continue
            d = dev_f.get((r.match_id, r.map_name))
            dev_a = np.array([d[p] for p in fa]) if d and all(p in d for p in fa) else None
            dev_b = np.array([d[p] for p in fb]) if d and all(p in d for p in fb) else None
            ia, ib = _arr([index.from_faceit(p) for p in fa]), _arr([index.from_faceit(p) for p in fb])
            if len(set(ia.tolist() + ib.tolist())) != 2 * LINEUP:
                continue
            events.append(Event(int(r.time.value), FACEIT, int(day), ia, ib, int(r.team_a_score), int(r.team_b_score), dev_a, dev_b,
                                _team_id(r.team_a_id), _team_id(r.team_b_id), r.match_id, fa, fb))
        info["faceit_maps"] = len(events) - info["hltv_maps"]
    events.sort(key=lambda e: e.time)
    return hm, events, info


@dataclass
class Shared:
    """The config-independent inputs of a pass: events, rankings, skill levels. Several passes can reuse one."""

    index: PlayerIndex
    hm: pd.DataFrame
    events: list[Event]
    info: dict
    rank: Rankings
    level: dict


def build_shared(data_dir: Path, last_cutoff: pd.Timestamp, cfg: RunConfig) -> Shared:
    index = new_index(data_dir)
    hm, events, info = build_events(data_dir, last_cutoff, index, cfg)
    return Shared(index, hm, events, info, Rankings(load(data_dir, "hltv/cs2_team_rankings", last_cutoff)),
                  first_skill_level(data_dir, last_cutoff) if cfg.use_faceit else {})


class Pass:
    def __init__(self, data_dir: Path, last_cutoff: pd.Timestamp, cfg: RunConfig, shared: Shared | None = None):
        self.cfg = cfg
        sh = shared or build_shared(data_dir, last_cutoff, cfg)
        self.index, self.hm, self.events, self.info, self.rank, self.level = sh.index, sh.hm, sh.events, sh.info, sh.rank, sh.level
        self.fh = SkillFilter(self.index.size, cfg.hltv)
        self.ff = SkillFilter(self.index.size, cfg.faceit_skill)
        self.recent: dict[int, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=cfg.k_series))
        self.pos = 0
        self.moves = roster_moves(data_dir, last_cutoff) if cfg.use_moves else {}
        self.standins = self._standins(data_dir, last_cutoff)
        self.last_time: dict[int, int] = {}
        self.by_player: dict[int, list[int]] = collections.defaultdict(list)
        self.faceit_recent: dict[int, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=cfg.k_series))
        self.series = self._series_table()
        self.actual: dict = {}
        for ev in self.events:
            if ev.source == HLTV and ev.match_id not in self.actual:
                self.actual[ev.match_id] = (set(ev.a.tolist()), set(ev.b.tolist()))

    def _standins(self, data_dir: Path, last_cutoff: pd.Timestamp) -> dict[tuple, set]:
        lu = load(data_dir, "hltv/cs2_match_lineups", last_cutoff, ["match_id", "match_date", "team_id", "player_id", "is_standin"])
        out: dict[tuple, set] = collections.defaultdict(set)
        for r in lu[lu.is_standin == True].itertuples(index=False):
            idx = self.index.hltv.get(int(r.player_id))
            if idx is not None:
                out[(r.match_id, int(r.team_id))].add(idx)
        return out

    def _series_table(self) -> pd.DataFrame:
        h = self.hm[self.hm.bo_type == 3]
        g = h.groupby("match_id", sort=False)
        t = g.agg(time=("time", "min"), team_a_id=("team_a_id", "first"), team_b_id=("team_b_id", "first"),
                  wins=("team_a_won", "sum"), n=("team_a_won", "size"))
        t["y"] = (t.wins * 2 > t.n).astype(int)
        return t.reset_index()

    def _introduce(self, ev: Event) -> None:
        for idxs, team, fids in ((ev.a, ev.team_a, ev.fa), (ev.b, ev.team_b, ev.fb)):
            for k, p in enumerate(idxs):
                if ev.source == HLTV:
                    if not self.fh.known(int(p)):
                        pts = self.rank.points(team, pd.Timestamp(ev.time))
                        self.fh.introduce(int(p), self.cfg.hltv.a + self.cfg.hltv.b * math.log1p(pts), ev.day)
                elif not self.ff.known(int(p)):
                    c = self.cfg.faceit_skill
                    self.ff.introduce(int(p), c.a + c.level_slope * (self.level.get(fids[k], 5) - 5), ev.day)

    def _process(self, ev: Event) -> None:
        self._introduce(ev)
        f = self.fh if ev.source == HLTV else self.ff
        f.observe_map(ev.day, ev.a, ev.b, ev.ra, ev.rb)
        if ev.source == FACEIT and not self.cfg.faceit_rating:
            return
        if ev.dev_a is not None:
            f.observe_rating(ev.day, ev.a, ev.dev_a)
        if ev.dev_b is not None:
            f.observe_rating(ev.day, ev.b, ev.dev_b)
        if ev.source == FACEIT:
            for team, players in ((ev.team_a, ev.a), (ev.team_b, ev.b)):
                if team >= 0:
                    q = self.faceit_recent[team]
                    if q and q[-1][0] == ev.match_id:
                        q[-1][1] = tuple(players.tolist())
                    else:
                        q.append([ev.match_id, tuple(players.tolist())])
        if ev.source == HLTV:
            for team, players in ((ev.team_a, ev.a), (ev.team_b, ev.b)):
                self.last_time[team] = ev.time
                q = self.recent[team]
                if q and q[-1][0] == ev.match_id:
                    q[-1][1] = tuple(players.tolist())
                else:
                    q.append([ev.match_id, tuple(players.tolist())])

    def advance(self, cutoff: pd.Timestamp) -> None:
        limit = int(cutoff.value)
        while self.pos < len(self.events) and self.events[self.pos].time < limit:
            ev = self.events[self.pos]
            if ev.source == HLTV or self.cfg.use_faceit:
                self._process(ev)
                for p in ev.a.tolist() + ev.b.tolist():
                    self.by_player[p].append(self.pos)
            self.pos += 1

    def lineup(self, team: int, cutoff: pd.Timestamp) -> np.ndarray | None:
        """The team's five at the cutoff."""
        ordered = self._ranked(team, cutoff)
        return _arr(ordered[:LINEUP]) if ordered else None

    def _ranked(self, team: int, cutoff: pd.Timestamp) -> list[int]:
        """Candidate players, most likely first: recency-weighted appearances over the team's last series (stand-ins
        count less), then the transfer log's departures since its last series removed and arrivals put first."""
        q = self.recent.get(int(team))
        if not q:
            fallback = self._faceit_lineup(int(team))
            return [] if fallback is None else fallback.tolist()
        score: dict[int, tuple[float, int]] = {}
        for age, (match_id, players) in enumerate(reversed(q)):
            standin = self.standins.get((match_id, int(team)), ())
            for p in players:
                weight = self.cfg.decay ** age * (self.cfg.standin_weight if p in standin else 1.0)
                total, latest = score.get(p, (0.0, -len(q)))
                score[p] = (total + weight, max(latest, -age))
        ordered = sorted(score, key=lambda p: (-score[p][0], -score[p][1]))
        moves = self.moves.get(int(team))
        if moves:
            last = np.datetime64(pd.Timestamp(self.last_time[int(team)]).normalize())
            end = np.datetime64(cutoff)
            gone, arrived = set(), []
            for date, pid, delta in moves:
                if last < date < end:
                    idx = self.index.hltv.get(pid)
                    if idx is None:
                        continue
                    if delta < 0:
                        gone.add(idx)
                        arrived = [x for x in arrived if x != idx]
                    elif idx not in arrived:
                        gone.discard(idx)
                        arrived.append(idx)
            ordered = arrived[::-1] + [p for p in ordered if p not in gone and p not in arrived]
        return ordered

    def _faceit_lineup(self, team: int) -> np.ndarray | None:
        """A team with no HLTV map yet is known, where FACEIT matched it, by the players of its latest FACEIT matches."""
        q = self.faceit_recent.get(team)
        if not q:
            return None
        score: dict[int, tuple[float, int]] = {}
        for age, (_, players) in enumerate(reversed(q)):
            for p in players:
                total, latest = score.get(p, (0.0, -len(q)))
                score[p] = (total + self.cfg.decay ** age, max(latest, -age))
        return _arr(sorted(score, key=lambda p: (-score[p][0], -score[p][1]))[:LINEUP])

    def _virtual(self, team: int, day: pd.Timestamp) -> tuple[float, float]:
        cfg = self.cfg.hltv
        return cfg.a + cfg.b * math.log1p(self.rank.points(team, day)), cfg.sigma0 ** 2 / LINEUP

    def _shift(self) -> float:
        """The level difference between the two skill scales, from players with plenty of maps on both."""
        both = (self.fh.s.games >= self.cfg.shift_min_games) & (self.ff.s.games >= self.cfg.shift_min_games)
        if both.sum() < 30:
            return 0.0
        return float(np.median(self.ff.s.mu[both] - self.fh.s.mu[both]))

    def _state(self, lineup: np.ndarray, team: int, day: int, cutoff: pd.Timestamp, shift: float) -> tuple[np.ndarray, np.ndarray]:
        """HLTV-filter mean and variance of the lineup. A player with no HLTV map yet takes his FACEIT skill, moved to
        the HLTV scale, if he has one, and the newcomer prior otherwise."""
        mu, var = self.fh.current(lineup, day)
        unknown = self.fh.s.seen[lineup] == NEWCOMER
        if unknown.any():
            prior, _ = self._virtual(team, cutoff)
            mu, var = mu.copy(), var.copy()
            on_faceit = self.ff.s.seen[lineup] != NEWCOMER
            mu_f, var_f = self.ff.current(lineup, day)
            mu[unknown] = np.where(on_faceit, mu_f - shift, prior)[unknown]
            var[unknown] = np.where(on_faceit, var_f + self.cfg.faceit_offset_var, self.cfg.hltv.sigma0 ** 2)[unknown]
        return mu, var

    def _faceit_residual(self, lineup: np.ndarray, mu_h: np.ndarray, shift: float) -> tuple[float, float, float, int]:
        """Mean over the lineup of how far each player's FACEIT skill sits above his HLTV skill (net of the level
        shift), weighted by how many FACEIT maps stand behind it (zero with none). Split by whether the HLTV skill
        is itself thin: that is where a second measurement should count most."""
        g = self.ff.s.games[lineup].astype(float)
        weight = g / (g + self.cfg.games_half)
        resid = weight * (self.ff.s.mu[lineup] - shift - mu_h)
        thin = self.fh.s.games[lineup] < self.cfg.thin_games
        return float(resid[thin].sum() / LINEUP), float(resid[~thin].sum() / LINEUP), float(weight.mean()), int((g >= 10).sum())

    def _together(self, lineup: np.ndarray, cutoff: pd.Timestamp) -> int:
        """Maps, from either source, in the last TOGETHER_DAYS where at least four of the five played side by side."""
        now = cutoff.value
        count: collections.Counter = collections.Counter()
        for p in lineup.tolist():
            for eid in reversed(self.by_player.get(p, ())):
                if self.events[eid].time < now - TOGETHER_DAYS * DAY_NS:
                    break
                count[eid] += 1
        return sum(1 for c in count.values() if c >= 4)

    def _series_price(self, a: np.ndarray, b: np.ndarray, team_a: int, team_b: int, day: int, cutoff: pd.Timestamp,
                      shift: float) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        (mu_a, var_a), (mu_b, var_b) = self._state(a, team_a, day, cutoff, shift), self._state(b, team_b, day, cutoff, shift)
        d, s2 = self.fh.gap(a, b, day, (mu_a, mu_b), (var_a, var_b))
        p_map = map_win_prob(d, s2)
        return p_map * p_map * (3 - 2 * p_map), d, s2, mu_a, mu_b

    def price(self, cutoff: pd.Timestamp, team_a: int, team_b: int, shift: float = 0.0) -> dict:
        f, day = self.fh, day_number(cutoff)
        a, b = self.lineup(team_a, cutoff), self.lineup(team_b, cutoff)
        row = {"known_a": 0 if a is None else len(a), "known_b": 0 if b is None else len(b)}
        if a is not None and b is not None and len(a) == LINEUP and len(b) == LINEUP and not set(a.tolist()) & set(b.tolist()):
            p_series, d, s2, mu_a, mu_b = self._series_price(a, b, team_a, team_b, day, cutoff, shift)
            lo_a, hi_a, wa, na = self._faceit_residual(a, mu_a, shift)
            lo_b, hi_b, wb, nb = self._faceit_residual(b, mu_b, shift)
            row["coh_a"], row["coh_b"] = self._together(a, cutoff), self._together(b, cutoff)
        else:
            ma, va = self._virtual(team_a, cutoff) if a is None else (float(f.current(a, day)[0].mean()), float(f.current(a, day)[1].sum()) / len(a) ** 2)
            mb, vb = self._virtual(team_b, cutoff) if b is None else (float(f.current(b, day)[0].mean()), float(f.current(b, day)[1].sum()) / len(b) ** 2)
            d, s2 = ma - mb, va + vb
            p_map = map_win_prob(d, s2)
            p_series = p_map * p_map * (3 - 2 * p_map)
            lo_a = hi_a = lo_b = hi_b = wa = wb = 0.0
            na = nb = 0
            row.update({"coh_a": 0, "coh_b": 0})
        row.update({"d": d, "sd": math.sqrt(max(s2, 0.0)), "p_series": min(max(p_series, 1e-4), 1 - 1e-4),
                    "fres_lo": lo_a - lo_b, "fres_hi": hi_a - hi_b, "fw_a": wa, "fw_b": wb, "nf_a": na, "nf_b": nb})
        for side, lu in (("a", a), ("b", b)):
            row[f"nnew_{side}"] = LINEUP if lu is None else int((f.s.games[lu] < 20).sum()) + LINEUP - len(lu)
        return row

    def run(self, cutoffs: list[pd.Timestamp], targets: pd.DataFrame | None) -> pd.DataFrame:
        rows = []
        for c in cutoffs:
            self.advance(c)
            shift = self._shift() if self.cfg.use_faceit else 0.0
            week = self.series[(self.series.time >= c) & (self.series.time < c + WEEK)]
            seen = set()
            if targets is not None:
                for r in targets[targets.week_cutoff == c].itertuples(index=False):
                    rows.append({"match_id": r.match_id, "week_cutoff": c, "is_target": True,
                                 "y": float(week.set_index("match_id").y.get(r.match_id, np.nan)),
                                 **self.price(c, int(r.team_a_id), int(r.team_b_id), shift)})
                    seen.add(r.match_id)
            for r in week.itertuples(index=False):
                if r.match_id in seen:
                    continue
                row = self.price(c, int(r.team_a_id), int(r.team_b_id), shift)
                act = self.actual.get(r.match_id)
                if act is not None:
                    la, lb = self.lineup(r.team_a_id, c), self.lineup(r.team_b_id, c)
                    row["ovl_a"] = -1 if la is None else len(act[0] & set(la.tolist()))
                    row["ovl_b"] = -1 if lb is None else len(act[1] & set(lb.tolist()))
                rows.append({"match_id": r.match_id, "week_cutoff": c, "is_target": False, "y": float(r.y),
                             **row})
        return pd.DataFrame(rows)


def monday_range(first: pd.Timestamp, last: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(first, last, freq="7D"))
