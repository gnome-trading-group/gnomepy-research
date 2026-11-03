"""
Player-skill filters replayed week by week.

Each rater is a Glicko-style filter over player skills (round-logit units) fed with HLTV maps and FACEIT organised
maps. The replay walks every Monday from the start of the data, applies the rows that became known during the week
before it, then snapshots each rater's price of the two announced fives for the series of the week that starts on that
Monday. The order of updates depends only on the rows themselves, so a run that ends at an earlier Monday reproduces
the same numbers for every week it shares with a longer run.

The raters differ only in their settings (prior variance, drift, newcomer discount, FACEIT weight, how much the
performance measurements count). Several settings near the best pre-dev log loss are stacked rather than one picked,
which averages out the arbitrary part of any one setting.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.data import FIVE, Inputs


@dataclass(frozen=True)
class RaterConfig:
    var0: float = 0.8           # prior variance of a new player's skill
    tau2: float = 2e-5          # variance added per idle day
    init: float = -0.2          # skill of a player in a five nobody in it is known
    fill_drop: float = 0.75     # a newcomer starts this far below the known teammates' mean
    hltv_step: float = 1.0      # likelihood weight of an HLTV map
    faceit_step: float = 0.75   # likelihood weight of a FACEIT organised map
    faceit_scale: float = 1.0   # round-logit gap on FACEIT maps per unit of skill gap
    r_gamma: float = 0.3        # HLTV rating deviation per unit of skill deviation within a five
    r_sd: float = 0.8
    fr_gamma: float = 0.3       # the same for the FACEIT stat proxy of the rating
    fr_sd: float = 0.3
    # More HLTV per-player stats as measurements of the same within-five deviation: (column, gamma, sd).
    channels: tuple = ()


# Name and overrides of each rater; "gauss" is the best pre-dev setting, the others are near it.
RATERS = (
    ("gauss", {}),
    ("gx_a", {"var0": 0.5, "tau2": 1e-5, "faceit_step": 1.0, "r_sd": 1.0, "fr_sd": 0.4, "hltv_step": 1.3}),
    ("gx_b", {"var0": 1.2, "tau2": 5e-5, "fill_drop": 0.6, "faceit_step": 0.5, "faceit_scale": 1.2, "r_sd": 0.6,
              "fr_sd": 0.4, "hltv_step": 0.8}),
    ("gx_fast", {"tau2": 2e-4}),
    ("gx_slow", {"tau2": 0.0, "var0": 0.5}),
)


def _tuples(over: dict) -> dict:
    """JSON overrides carry channels as lists; the frozen config wants tuples."""
    return {k: tuple(tuple(x) for x in v) if k == "channels" else v for k, v in over.items()}


def _days(t) -> float:
    return pd.Timestamp(t).value / 86400e9


class GaussPlayer:
    """Player skills as independent Gaussians updated by one Newton step per map on the round likelihood: steps shrink
    as a player's variance does, and variance grows again while a player is idle."""

    def __init__(self, cfg: RaterConfig):
        self.cfg = cfg
        self.mu: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.t: dict[int, float] = {}

    def _state(self, five, now: float) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        known = [self.mu[p] for p in five if p in self.mu]
        fill = (np.mean(known) - c.fill_drop) if known else c.init
        mu = np.array([self.mu.get(p, fill) for p in five])
        var = np.array([min(self.var[p] + c.tau2 * max(now - self.t[p], 0.0), c.var0) if p in self.var else c.var0
                        for p in five])
        return mu, var

    def update(self, five_a, five_b, rounds_a: float, rounds: float, now: float, step: float, scale: float = 1.0) -> None:
        ma, va = self._state(five_a, now)
        mb, vb = self._state(five_b, now)
        d = scale * (ma.mean() - mb.mean())
        vd = scale * scale * (va.sum() + vb.sum()) / (FIVE * FIVE)
        p = 1.0 / (1.0 + np.exp(-d))
        h = step * rounds * p * (1 - p)
        g = step * (rounds_a - rounds * p)
        k = scale / FIVE / (1.0 + vd * h)
        for five, m, v, sign in ((five_a, ma, va, 1.0), (five_b, mb, vb, -1.0)):
            for i, pl in enumerate(five):
                self.mu[pl] = m[i] + sign * v[i] * k * g
                self.var[pl] = max(v[i] - (v[i] * scale / FIVE) ** 2 * h / (1.0 + vd * h), 1e-4)
                self.t[pl] = now

    def rate(self, five, ratings, gamma: float, sd: float) -> None:
        """Map ratings relative to teammates as a measurement of who carries the five:
        rating_p - mean(rating) ~ N(gamma * (s_p - mean(s)), sd^2), one player at a time. Round results only see a
        five's mean; this is what moves a player's skill apart from the teammates', which matters whenever the five
        a team fields next is not the five it was rated with."""
        rt = np.asarray(ratings, float)
        if not np.isfinite(rt).all() or any(p not in self.mu for p in five):
            return
        obs = (rt - rt.mean()) / gamma
        r2 = (sd / gamma) ** 2
        mu = np.array([self.mu[p] for p in five])
        dev = mu - mu.mean()
        lead = 1.0 - 1.0 / FIVE
        for i, p in enumerate(five):
            v = self.var[p]
            k = v * lead / (v * lead * lead + r2)
            self.mu[p] = mu[i] + k * (obs[i] - dev[i])
            self.var[p] = max(v * (1.0 - k * lead), 1e-4)

    def price(self, five, now: float) -> float:
        if not five:
            return self.cfg.init
        mu, _ = self._state(five, now)
        return float(mu.mean())


class Replay:
    def __init__(self, inputs: Inputs, raters=RATERS):
        self.inp = inputs
        self.raters = [(name, GaussPlayer(dataclasses.replace(RaterConfig(), **_tuples(over)))) for name, over in raters]
        self._maps = list(inputs.maps.itertuples(index=False))
        self._fmaps = list(inputs.faceit_maps.itertuples(index=False)) if inputs.faceit_maps is not None else []
        self._hi = self._fi = 0

    def _advance(self, cutoff: pd.Timestamp) -> None:
        while self._hi < len(self._maps) and self._maps[self._hi].known < cutoff:
            r = self._maps[self._hi]
            now = _days(r.time)
            for _, gm in self.raters:
                c = gm.cfg
                gm.update(r.five_a, r.five_b, r.rounds_a, r.rounds, now, c.hltv_step)
                if c.r_gamma > 0:
                    gm.rate(r.five_a, r.rating_a, c.r_gamma, c.r_sd)
                    gm.rate(r.five_b, r.rating_b, c.r_gamma, c.r_sd)
                for col, gamma, sd in c.channels:
                    gm.rate(r.five_a, getattr(r, f"{col}_a"), gamma, sd)
                    gm.rate(r.five_b, getattr(r, f"{col}_b"), gamma, sd)
            self._hi += 1
        while self._fi < len(self._fmaps) and self._fmaps[self._fi].known < cutoff:
            r = self._fmaps[self._fi]
            now = _days(r.time)
            for _, gm in self.raters:
                c = gm.cfg
                if c.faceit_step <= 0:
                    continue
                gm.update(r.five_a, r.five_b, r.rounds_a, r.rounds, now, c.faceit_step, c.faceit_scale)
                if c.fr_gamma > 0:
                    gm.rate(r.five_a, r.rating_a, c.fr_gamma, c.fr_sd)
                    gm.rate(r.five_b, r.rating_b, c.fr_gamma, c.fr_sd)
            self._fi += 1

    def run(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: match_id, team_a, team_b, week_cutoff, five_a, five_b (+ anything else, passed through). Returns one
        row per input row with each rater's A-minus-B skill gap, from the state at its week_cutoff."""
        starts = [self.inp.maps.known.min()]
        if self.inp.faceit_maps is not None and len(self.inp.faceit_maps):
            starts.append(self.inp.faceit_maps.known.min())
        first = pd.Timestamp(min(starts)).floor("D")
        first = first - pd.Timedelta(days=first.weekday())
        last = pd.Timestamp(rows.week_cutoff.max())
        by_week = {pd.Timestamp(k): g for k, g in rows.groupby("week_cutoff")}
        out = []
        for c in pd.date_range(first, last, freq="7D"):
            self._advance(c)
            g = by_week.get(c)
            if g is None:
                continue
            now = _days(c)
            for r in g.itertuples(index=False):
                rec = r._asdict()
                for name, gm in self.raters:
                    rec[name] = gm.price(r.five_a, now) - gm.price(r.five_b, now)
                out.append(rec)
        return pd.DataFrame(out)
