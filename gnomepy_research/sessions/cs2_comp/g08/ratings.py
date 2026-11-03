"""A recurrent player-skill state run over every decisive map in time order, read out at weekly cutoffs.

Each player carries a Gaussian belief (mean, variance) about their skill. A map is one observation of the gap between
the two fives' mean skills through its regulation rounds:

    rounds_A ~ Binomial(n, sigmoid(mean(s_A) - mean(s_B)))

and the beliefs of the ten players are updated by one extended-Kalman step, with the round likelihood tempered by
`round_weight` because rounds within a map are not independent. Between appearances a player's variance grows with
time (`drift_per_day`), so a team that changed or went quiet is believed less firmly. This is the sequence model: the
state after a team's latest maps summarises its whole history, and the update weights are set by the beliefs' own
uncertainty rather than a fixed K.

At every Monday 00:00 UTC cutoff the state is read out for the series of that week, using only maps known before the
cutoff, so every row (training series and target series alike) is point-in-time for its own week.

Alongside the skill state, each team keeps a short exponentially weighted record of how far its recent maps beat the
state's own expectation (form), the dates it played, and which fives it fielded, from which cohesion and activity are
read."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

FIVE = 5


@dataclass(frozen=True)
class RatingConfig:
    prior_var: float = 0.30
    drift_per_day: float = 0.0006
    round_weight: float = 0.35
    faceit_weight: float = 0.5
    new_offset: float = -0.15
    form_half_life_maps: float = 6.0
    lan_bonus: float = 0.0


def regulation(score_a: np.ndarray, score_b: np.ndarray, overtime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An overtime map counts as 12-12 of 24 regulation rounds."""
    ot = overtime | (score_a + score_b > 24)
    return np.where(ot, 12, score_a), np.where(ot, 24, score_a + score_b)


class SkillState:
    def __init__(self, cfg: RatingConfig):
        self.cfg = cfg
        self.mu: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.seen: dict[int, float] = {}
        self.maps: dict[int, int] = defaultdict(int)
        self.team_form: dict[int, float] = defaultdict(float)
        self.team_days: dict[int, deque] = defaultdict(lambda: deque(maxlen=400))
        self.team_fives: dict[int, deque] = defaultdict(lambda: deque(maxlen=60))

    def _prior(self, p: int, mates: list[int]) -> tuple[float, float]:
        known = [self.mu[m] for m in mates if m in self.mu]
        base = float(np.mean(known)) if len(known) >= 2 else 0.0
        return base + self.cfg.new_offset, self.cfg.prior_var

    def belief(self, p: int, day: float, mates: list[int]) -> tuple[float, float, bool]:
        if p in self.mu:
            v = self.var[p] + self.cfg.drift_per_day * max(day - self.seen[p], 0.0)
            return self.mu[p], min(v, self.cfg.prior_var * 2), True
        m, v = self._prior(p, mates)
        return m, v, False

    def update(self, five_a: list[int], five_b: list[int], rounds_a: float, rounds: float, day: float, weight: float,
               team_a: int, team_b: int, lan: int) -> None:
        cfg = self.cfg
        ba = [self.belief(p, day, five_a) for p in five_a]
        bb = [self.belief(p, day, five_b) for p in five_b]
        d = np.mean([b[0] for b in ba]) - np.mean([b[0] for b in bb])
        s = (sum(b[1] for b in ba) + sum(b[1] for b in bb)) / FIVE ** 2
        pr = 1.0 / (1.0 + np.exp(-d))
        w = cfg.round_weight * weight
        g = w * (rounds_a - rounds * pr)
        info = w * rounds * pr * (1 - pr)
        denom = 1.0 + info * s
        for sign, five, bel in ((1.0, five_a, ba), (-1.0, five_b, bb)):
            for p, (m, v, _) in zip(five, bel):
                k = v / FIVE / denom
                self.mu[p] = m + sign * k * g
                self.var[p] = max(v - (v / FIVE) ** 2 * info / denom, 1e-4)
                self.seen[p] = day
                self.maps[p] += 1
        if rounds > 0:
            share = (rounds_a + 0.5) / (rounds + 1.0)
            resid = np.log(share / (1 - share)) - d
            a = 1.0 - 0.5 ** (1.0 / cfg.form_half_life_maps)
            for team, r in ((team_a, resid), (team_b, -resid)):
                if team >= 0:
                    self.team_form[team] = (1 - a) * self.team_form[team] + a * r * weight
        for team, five in ((team_a, five_a), (team_b, five_b)):
            if team >= 0:
                self.team_days[team].append(day)
                self.team_fives[team].append((day, frozenset(five)))

    def five_features(self, five: list[int] | None, team: int, day: float) -> dict:
        if not five:
            five = []
            last = self.team_fives.get(team)
            if last:
                five = list(last[-1][1])
        bel = [self.belief(p, day, five) for p in five]
        if not bel:
            return {"mu": self.cfg.new_offset, "var": self.cfg.prior_var, "unknown": FIVE, "lowmaps": FIVE, "cohesion": 0.0,
                    "maps90": 0, "idle": 365.0, "form": 0.0, "sd_mu": 0.0}
        mus = np.array([b[0] for b in bel])
        days = np.array(self.team_days.get(team, []))
        fives = self.team_fives.get(team, [])
        fset = set(five)
        recent = [f for dd, f in fives if day - dd <= 90]
        coh = float(np.mean([len(f & fset) >= 4 for f in recent])) if recent else 0.0
        return {"mu": float(mus.mean()), "var": float(np.mean([b[1] for b in bel])),
                "unknown": int(sum(not b[2] for b in bel)), "lowmaps": int(sum(self.maps.get(p, 0) < 10 for p in five)),
                "cohesion": coh, "maps90": int(((day - days) <= 90).sum()) if len(days) else 0,
                "idle": float(day - days[-1]) if len(days) else 365.0, "form": float(self.team_form.get(team, 0.0)),
                "sd_mu": float(mus.std())}


def _days(ts: pd.Series | np.ndarray) -> np.ndarray:
    return (pd.to_datetime(ts).to_numpy().astype("datetime64[s]").astype(np.int64)) / 86400.0


def run(maps: pd.DataFrame, queries: pd.DataFrame, cfg: RatingConfig) -> pd.DataFrame:
    """maps: match_id, known, source, five_a, five_b, rounds_a, rounds, team_a, team_b, is_lan (any order).
    queries: qid, cutoff, team_a, team_b, five_a, five_b (fives may be None).
    Returns one feature row per query, read from the state holding every map known before the query's cutoff."""
    maps = maps.sort_values(["known", "match_id"], kind="stable").reset_index(drop=True)
    mk = maps.known.to_numpy()
    mday = _days(maps.known)
    fa, fb = maps.five_a.to_numpy(), maps.five_b.to_numpy()
    ra, n = maps.rounds_a.to_numpy(float), maps.rounds.to_numpy(float)
    wt = np.where(maps.source.to_numpy() == "faceit", cfg.faceit_weight, 1.0)
    ta, tb = maps.team_a.to_numpy(np.int64), maps.team_b.to_numpy(np.int64)
    lan = maps.is_lan.to_numpy(int)
    state = SkillState(cfg)
    out = []
    i = 0
    for cutoff, qs in queries.groupby("cutoff", sort=True):
        c64 = np.datetime64(pd.Timestamp(cutoff))
        stop = np.searchsorted(mk, c64, side="left")
        for j in range(i, stop):
            state.update(list(fa[j]), list(fb[j]), ra[j], n[j], mday[j], wt[j], ta[j], tb[j], lan[j])
        i = stop
        cday = _days(np.array([c64]))[0]
        for q in qs.itertuples(index=False):
            r = {"qid": q.qid}
            for side, team, five in (("a", q.team_a, q.five_a), ("b", q.team_b, q.five_b)):
                f = state.five_features(None if five is None or len(five) == 0 else [int(x) for x in five], int(team), cday)
                r.update({f"{k}_{side}": v for k, v in f.items()})
            out.append(r)
    return pd.DataFrame(out)
