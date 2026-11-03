
"""
g14's rating engine: one forward pass over every decisive map in the order it became known, keeping a Gaussian
belief about each player's skill, each team's system offset and each team's offset per map, all in round-logit
units. Every belief drifts: its variance grows with the days since it was last observed, so a long break or a thin
history means a wide belief, and a roster change widens the team offset (the system was built for another five).

A map is an observation of d = (mean skill of five A + offset A + map offset A) - (same for B) with precision
n_eff * p * (1 - p); rounds inside a map are correlated, so n_eff is a fraction of the rounds played, smaller still
on FACEIT. The update is the joint Kalman step for the linearised observation, so the players, the team offset and
the map offset split the surprise in proportion to how uncertain each is.

Series rows (for the learned layers) and target rows are priced from the belief as it stood before the series,
with the announced fives, so what the layers learn on history is what they see on a target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binom

from gnomepy_research.sessions.cs2_comp.g14.data import FIVE, LOG_UNRANKED, SOURCE_HLTV, UNRANKED, Inputs

DAY = 86400.0
_ROUNDS = np.arange(25)


def map_win(d) -> np.ndarray:
    """P(win a regulation-24 map) for each round-logit gap; 12-12 is a coin flip."""
    d = np.asarray(d, float)
    pi = expit(d)
    pmf = binom.pmf(_ROUNDS.reshape((-1,) + (1,) * d.ndim), 24, pi[None])
    return pmf[13:].sum(axis=0) + 0.5 * pmf[12]


def bo3(p) -> np.ndarray:
    p = np.asarray(p, float)
    return p * p * (3 - 2 * p)


@dataclass(frozen=True)
class RatingConfig:
    rho_hltv: float = 0.5           # effective independent rounds per round played on HLTV maps
    rho_faceit: float = 0.25        # the same on FACEIT maps
    drift_player: float = 2e-4      # skill variance added per day since last seen
    drift_team: float = 4e-4        # team offset variance per day
    drift_map: float = 4e-4         # team-map offset variance per day
    var_player0: float = 0.09       # prior variance of a player's skill around the team prior
    var_team0: float = 0.02         # prior variance of a team offset
    var_map0: float = 0.02          # prior variance of a team-map offset
    var_player_max: float = 0.16
    var_team_max: float = 0.06
    var_map_max: float = 0.06
    churn_widen: float = 0.6        # share of var_team0 restored to the team offset per changed player (of five)
    rank_slope: float = 0.08        # prior skill per unit of log-rank strength for a player the pass never saw
    rank_intercept: float = -0.25
    form_decay: float = 0.8         # per-map decay of a team's running surprise (form)
    pool_half_life_days: float = 45.0
    pool_min_share: float = 0.03
    faceit: bool = True
    # Team Elo: the team's strength as one belief (the sum a map identifies), moved by the skill difference when the
    # five changes; player skills are the team strength plus the player's within-team HLTV rating deviation.
    elo_var0: float = 0.03
    elo_drift: float = 3e-4
    elo_var_max: float = 0.08
    elo_churn_widen: float = 0.5
    pskill_var0: float = 0.04
    pskill_drift: float = 1e-4
    pskill_var_max: float = 0.08
    rating_gamma: float = 0.5
    rating_obs_var: float = 0.05
    norating_obs_var: float = 0.4


@dataclass
class Belief:
    mu: float
    var: float
    last: float                     # seconds since epoch of the last observation
    n: int = 0


@dataclass
class TeamState:
    offset: Belief
    last_five: tuple[int, ...] | None = None
    last_time: float = -1.0
    form: float = 0.0
    maps: int = 0
    hltv_maps: int = 0
    by_map: dict[str, Belief] = field(default_factory=dict)
    fives: dict[tuple[int, ...], int] = field(default_factory=dict)
    elo_five: tuple[int, ...] | None = None


class Engine:
    """The forward pass. `run` walks the maps once and calls back at every series start and every target cutoff."""

    def __init__(self, inputs: Inputs, cfg: RatingConfig):
        self.inputs = inputs
        self.cfg = cfg
        self.players: dict[int, Belief] = {}
        self.teams: dict[int, TeamState] = {}
        self.elo: dict[int, Belief] = {}
        self.pskill: dict[int, Belief] = {}
        self.h2h: dict[tuple[int, int], list[tuple[float, float]]] = {}
        self.pool_weight: dict[str, float] = {}
        self.pool_time: float = -1.0
        self.moves = inputs.moves
        self._move_dates = {int(t): g.event_date.to_numpy() for t, g in self.moves.groupby("team_id")}

    # --- beliefs ---------------------------------------------------------------------------------------------------

    def _player(self, pid: int, team: TeamState | None, rank: float, now: float) -> Belief:
        b = self.players.get(pid)
        if b is None:
            prior = self.cfg.rank_intercept + self.cfg.rank_slope * (LOG_UNRANKED - math.log(max(rank, 1.0)))
            if team is not None and team.maps > 0 and team.last_five:
                known = [self.players[p].mu for p in team.last_five if p in self.players]
                if len(known) >= 2:
                    prior = 0.5 * prior + 0.5 * float(np.mean(known))
            b = Belief(prior, self.cfg.var_player0, now)
            self.players[pid] = b
            return b
        b.var = min(b.var + self.cfg.drift_player * max(0.0, now - b.last) / DAY, self.cfg.var_player_max)
        b.last = now
        return b

    def _team(self, tid: int, now: float) -> TeamState:
        t = self.teams.get(tid)
        if t is None:
            t = TeamState(Belief(0.0, self.cfg.var_team0, now))
            self.teams[tid] = t
            return t
        o = t.offset
        o.var = min(o.var + self.cfg.drift_team * max(0.0, now - o.last) / DAY, self.cfg.var_team_max)
        o.last = now
        return t

    def _team_map(self, t: TeamState, map_name: str, now: float) -> Belief:
        b = t.by_map.get(map_name)
        if b is None:
            b = Belief(0.0, self.cfg.var_map0, now)
            t.by_map[map_name] = b
            return b
        b.var = min(b.var + self.cfg.drift_map * max(0.0, now - b.last) / DAY, self.cfg.var_map_max)
        b.last = now
        return b

    def _churn(self, t: TeamState, five: tuple[int, ...]) -> int:
        if t.last_five is None:
            return 0
        return FIVE - len(set(five) & set(t.last_five))

    def _elo(self, tid: int, rank: float, now: float) -> Belief:
        e = self.elo.get(tid)
        if e is None:
            prior = self.cfg.rank_intercept + self.cfg.rank_slope * (LOG_UNRANKED - math.log(max(rank, 1.0)))
            e = Belief(prior, self.cfg.elo_var0, now)
            self.elo[tid] = e
            return e
        e.var = min(e.var + self.cfg.elo_drift * max(0.0, now - e.last) / DAY, self.cfg.elo_var_max)
        e.last = now
        return e

    def _skill_mean(self, five: tuple[int, ...], fallback: float) -> float:
        return float(np.mean([self.pskill[p].mu if p in self.pskill else fallback for p in five]))

    def _elo_lineup(self, t: TeamState, e: Belief, five: tuple[int, ...]) -> None:
        if t.elo_five is None or five == t.elo_five:
            t.elo_five = five
            return
        changed = FIVE - len(set(five) & set(t.elo_five))
        e.mu += self._skill_mean(five, e.mu) - self._skill_mean(t.elo_five, e.mu)
        e.var = min(e.var + self.cfg.elo_churn_widen * self.cfg.elo_var0 * changed / FIVE, self.cfg.elo_var_max)
        t.elo_five = five

    def _pskill_update(self, e: Belief, five: tuple[int, ...], ratings: tuple[float, ...], now: float) -> None:
        cfg = self.cfg
        rated = all(r == r for r in ratings)
        mean_rating = sum(ratings) / FIVE if rated else 0.0
        for p, r in zip(five, ratings):
            b = self.pskill.get(p)
            if b is None:
                b = Belief(e.mu, cfg.pskill_var0, now)
                self.pskill[p] = b
            else:
                b.var = min(b.var + cfg.pskill_drift * max(0.0, now - b.last) / DAY, cfg.pskill_var_max)
                b.last = now
            if rated:
                target, ov = e.mu + cfg.rating_gamma * (r - mean_rating), cfg.rating_obs_var
            else:
                target, ov = e.mu, cfg.norating_obs_var
            k = b.var / (b.var + ov)
            b.mu += k * (target - b.mu)
            b.var *= 1.0 - k
            b.n += 1

    # --- the observation -------------------------------------------------------------------------------------------

    def observe(self, i: int) -> None:
        now = self._now[i]
        team_a, team_b = int(self._team_a[i]), int(self._team_b[i])
        pairs_a = sorted(zip((int(p) for p in self.inputs.five_a[i]), self.inputs.rating_a[i]))
        pairs_b = sorted(zip((int(p) for p in self.inputs.five_b[i]), self.inputs.rating_b[i]))
        five_a, ratings_a = tuple(p for p, _ in pairs_a), tuple(float(r) for _, r in pairs_a)
        five_b, ratings_b = tuple(p for p, _ in pairs_b), tuple(float(r) for _, r in pairs_b)
        map_name = self._map_name[i]
        is_hltv = self._source[i] == SOURCE_HLTV
        ta, tb = self._team(team_a, now), self._team(team_b, now)
        churn_a, churn_b = self._churn(ta, five_a), self._churn(tb, five_b)
        if churn_a:
            ta.offset.var = min(ta.offset.var + self.cfg.churn_widen * self.cfg.var_team0 * churn_a / FIVE, self.cfg.var_team_max)
        if churn_b:
            tb.offset.var = min(tb.offset.var + self.cfg.churn_widen * self.cfg.var_team0 * churn_b / FIVE, self.cfg.var_team_max)
        pa = [self._player(p, ta, self._rank_a[i], now) for p in five_a]
        pb = [self._player(p, tb, self._rank_b[i], now) for p in five_b]
        ma, mb = self._team_map(ta, map_name, now), self._team_map(tb, map_name, now)
        d = (sum(b.mu for b in pa) - sum(b.mu for b in pb)) / FIVE + ta.offset.mu - tb.offset.mu + ma.mu - mb.mu
        rho = self.cfg.rho_hltv if is_hltv else self.cfg.rho_faceit
        n_eff = self._rounds[i] * rho
        ea, eb = self._elo(team_a, self._rank_a[i], now), self._elo(team_b, self._rank_b[i], now)
        self._elo_lineup(ta, ea, five_a)
        self._elo_lineup(tb, eb, five_b)
        d_e = ea.mu - eb.mu
        p_e = 1.0 / (1.0 + math.exp(-d_e))
        info_e = n_eff * p_e * (1.0 - p_e)
        resid_e = (self._rounds_a[i] * rho - n_eff * p_e) / info_e
        s_e = 1.0 / info_e + ea.var + eb.var
        ea.mu += ea.var / s_e * resid_e
        eb.mu -= eb.var / s_e * resid_e
        ea.var -= ea.var * ea.var / s_e
        eb.var -= eb.var * eb.var / s_e
        self._pskill_update(ea, five_a, ratings_a, now)
        self._pskill_update(eb, five_b, ratings_b, now)
        p = 1.0 / (1.0 + math.exp(-d))
        info = n_eff * p * (1.0 - p)
        resid_d = (self._rounds_a[i] * rho - n_eff * p) / info
        beliefs = [(b, 1.0 / FIVE) for b in pa] + [(b, -1.0 / FIVE) for b in pb] + \
                  [(ta.offset, 1.0), (tb.offset, -1.0), (ma, 1.0), (mb, -1.0)]
        s = 1.0 / info + sum(b.var * a * a for b, a in beliefs)
        for b, a in beliefs:
            k = b.var * a / s
            b.mu += k * resid_d
            b.var -= k * a * b.var
            b.n += 1
        for t, five, sign in ((ta, five_a, 1.0), (tb, five_b, -1.0)):
            t.form = self.cfg.form_decay * t.form + sign * resid_d * min(1.0, info * 0.1)
            t.last_five = five
            t.last_time = now
            t.maps += 1
            t.hltv_maps += int(is_hltv)
            t.fives[five] = t.fives.get(five, 0) + 1
        if is_hltv and map_name:
            self._pool_add(map_name, now)
        if is_hltv:
            key = (min(team_a, team_b), max(team_a, team_b))
            won = 1.0 if self._rounds_a[i] * 2 > self._rounds[i] else -1.0
            self.h2h.setdefault(key, []).append((now, won if team_a == key[0] else -won))

    def head_to_head(self, team_a: int, team_b: int, now: float, days: float = 365.0) -> tuple[float, int]:
        """Shrunk map-win margin of A over B in their last year of meetings, and how many maps that is."""
        key = (min(team_a, team_b), max(team_a, team_b))
        past = [r for t, r in self.h2h.get(key, ()) if now - t <= days * DAY]
        n = len(past)
        margin = sum(past) if team_a == key[0] else -sum(past)
        return margin / (n + 4.0), n

    def _pool_add(self, map_name: str, now: float) -> None:
        if self.pool_time >= 0 and now > self.pool_time:
            decay = 0.5 ** ((now - self.pool_time) / DAY / self.cfg.pool_half_life_days)
            for k in self.pool_weight:
                self.pool_weight[k] *= decay
        self.pool_time = max(self.pool_time, now)
        self.pool_weight[map_name] = self.pool_weight.get(map_name, 0.0) + 1.0

    def pool(self) -> list[str]:
        total = sum(self.pool_weight.values())
        if total <= 0:
            return []
        return sorted(k for k, v in self.pool_weight.items() if v / total >= self.cfg.pool_min_share)

    # --- pricing a series from the current belief ---------------------------------------------------------------

    def _side(self, tid: int, five: tuple[int, ...], rank: float, now: float) -> dict:
        t = self.teams.get(tid)
        cfg = self.cfg
        if t is None:
            off_mu, off_var, churn, idle, form, maps, hltv_maps, together = 0.0, cfg.var_team0, FIVE, 365.0, 0.0, 0, 0, 0
        else:
            off_mu = t.offset.mu
            off_var = min(t.offset.var + cfg.drift_team * max(0.0, now - t.offset.last) / DAY, cfg.var_team_max)
            churn = self._churn(t, five)
            off_var = min(off_var + cfg.churn_widen * cfg.var_team0 * churn / FIVE, cfg.var_team_max)
            idle = (now - t.last_time) / DAY if t.last_time >= 0 else 365.0
            form, maps, hltv_maps = t.form, t.maps, t.hltv_maps
            together = t.fives.get(five, 0)
        prior = cfg.rank_intercept + cfg.rank_slope * (LOG_UNRANKED - math.log(max(rank, 1.0)))
        if t is not None and t.last_five:
            known = [self.players[p].mu for p in t.last_five if p in self.players]
            if len(known) >= 2:
                prior = 0.5 * prior + 0.5 * float(np.mean(known))
        mus, vars_, new, thin, nmaps = [], [], 0, 0, []
        for p in five:
            b = self.players.get(p)
            if b is None:
                mus.append(prior)
                vars_.append(cfg.var_player0)
                new += 1
                thin += 1
                nmaps.append(0)
            else:
                mus.append(b.mu)
                vars_.append(min(b.var + cfg.drift_player * max(0.0, now - b.last) / DAY, cfg.var_player_max))
                thin += int(b.n < 5)
                nmaps.append(b.n)
        moves = self._move_dates.get(tid)
        if moves is not None:
            j = np.searchsorted(moves, np.datetime64(int(now), "s").astype("datetime64[ns]"), side="left") - 1
            since_move = (now - moves[j].astype("datetime64[s]").astype(np.int64)) / DAY if j >= 0 else 365.0
        else:
            since_move = 365.0
        return {"skill": float(np.mean(mus)), "skill_var": float(np.sum(vars_)) / (FIVE * FIVE), "off": off_mu, "off_var": off_var,
                "churn": churn, "new": new, "thin": thin, "idle": min(idle, 365.0), "form": form, "maps": maps,
                "hltv_maps": hltv_maps, "together": together, "since_move": min(since_move, 365.0),
                "min_maps": min(nmaps), "map_offsets": (t.by_map if t is not None else {})}

    def _side_elo(self, tid: int, five: tuple[int, ...], rank: float, now: float) -> tuple[float, float, float]:
        cfg = self.cfg
        e = self.elo.get(tid)
        t = self.teams.get(tid)
        if e is None:
            prior = cfg.rank_intercept + cfg.rank_slope * (LOG_UNRANKED - math.log(max(rank, 1.0)))
            return prior, cfg.elo_var0, 0.0
        var = min(e.var + cfg.elo_drift * max(0.0, now - e.last) / DAY, cfg.elo_var_max)
        adj = 0.0
        if t is not None and t.elo_five is not None and five != t.elo_five:
            changed = FIVE - len(set(five) & set(t.elo_five))
            adj = self._skill_mean(five, e.mu) - self._skill_mean(t.elo_five, e.mu)
            var = min(var + cfg.elo_churn_widen * cfg.elo_var0 * changed / FIVE, cfg.elo_var_max)
        return e.mu + adj, var, adj

    def price(self, team_a: int, five_a: tuple[int, ...], team_b: int, five_b: tuple[int, ...],
              rank_a: float, rank_b: float, now: float) -> dict:
        a = self._side(team_a, five_a, rank_a, now)
        b = self._side(team_b, five_b, rank_b, now)
        ea, va, adj_a = self._side_elo(team_a, five_a, rank_a, now)
        eb, vb, adj_b = self._side_elo(team_b, five_b, rank_b, now)
        d_elo = ea - eb
        pe = bo3(map_win(d_elo))
        d_player = a["skill"] - b["skill"]
        d_team = a["off"] - b["off"]
        d = d_player + d_team
        var_d = a["skill_var"] + b["skill_var"] + a["off_var"] + b["off_var"]
        pool = self.pool()
        veto = self._veto(d, a["map_offsets"], b["map_offsets"], pool, now)
        h2h, h2h_n = self.head_to_head(team_a, team_b, now)
        out = {"d_player": d_player, "d_team": d_team, "sd_d": math.sqrt(var_d), "h2h": h2h, "h2h_n": h2h_n,
               "z_series": float(np.log(bo3(map_win(d)) / (1 - bo3(map_win(d))))),
               "z_veto": veto, "pool_n": len(pool), "d_elo": d_elo, "sd_elo": math.sqrt(va + vb),
               "z_elo": float(np.log(pe / (1 - pe))), "adj_a": adj_a, "adj_b": adj_b}
        for k in ("churn", "new", "thin", "idle", "form", "maps", "hltv_maps", "together", "since_move", "min_maps"):
            out[f"{k}_a"], out[f"{k}_b"] = a[k], b[k]
        return out

    def _veto(self, d: float, offs_a: dict, offs_b: dict, pool: list[str], now: float) -> float:
        """Series logit when each side vetoes its worst maps of the active pool (bo3: ban, ban, pick, pick, ban, ban,
        decider), from the per-map offsets as believed now."""
        if len(pool) < 3:
            return float(np.log(bo3(map_win(d)) / (1 - bo3(map_win(d)))))
        dm = []
        for name in pool:
            oa, ob = offs_a.get(name), offs_b.get(name)
            dm.append(d + (oa.mu if oa else 0.0) - (ob.mu if ob else 0.0))
        r = sorted(dm)
        if len(r) >= 5:
            r = r[1:-1]
        p1, p2 = map_win(r[-1]), map_win(r[0])
        rem = r[1:-1]
        turn_a = True
        while len(rem) > 1:
            rem = rem[1:] if turn_a else rem[:-1]
            turn_a = not turn_a
        p3 = map_win(rem[0]) if rem else 0.5 * (p1 + p2)
        ps = p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
        ps = min(max(float(ps), 1e-4), 1 - 1e-4)
        return float(np.log(ps / (1 - ps)))

    # --- the pass ---------------------------------------------------------------------------------------------------

    def run(self, cutoffs: list[pd.Timestamp], on_series, on_cutoff) -> None:
        """Walk the maps in known order. on_series(index, now) is called before the first map of each HLTV series in
        inputs.series (state before the series); on_cutoff(cutoff) before the first map known at or after it."""
        m = self.inputs.maps
        self._now = (m.time.to_numpy().astype("datetime64[s]").astype(np.int64)).astype(float)
        known = m.known.to_numpy().astype("datetime64[ns]")
        self._team_a, self._team_b = m.team_a.to_numpy(), m.team_b.to_numpy()
        self._map_name = m.map_name.to_numpy()
        self._source = m.source.to_numpy()
        self._rounds, self._rounds_a = m.rounds.to_numpy(float), m.rounds_a.to_numpy(float)
        self._rank_a, self._rank_b = m.rank_a.to_numpy(float), m.rank_b.to_numpy(float)
        match_ids = m.match_id.to_numpy()
        series_pos = {str(mid): j for j, mid in enumerate(self.inputs.series.match_id.astype(str))}
        emitted: set[str] = set()
        cut = sorted(pd.Timestamp(c) for c in cutoffs)
        cut64 = [np.datetime64(c, "ns") for c in cut]
        ci = 0
        for i in range(len(m)):
            while ci < len(cut) and known[i] >= cut64[ci]:
                on_cutoff(cut[ci])
                ci += 1
            mid = match_ids[i]
            if self._source[i] == SOURCE_HLTV:
                j = series_pos.get(str(mid))
                if j is not None and mid not in emitted:
                    emitted.add(mid)
                    on_series(j, self._now[i])
            self.observe(i)
        while ci < len(cut):
            on_cutoff(cut[ci])
            ci += 1
