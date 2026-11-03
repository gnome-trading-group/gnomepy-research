"""g07's skill model: a Bayesian filter over every map, in the order the maps became known.

Each player p has a skill s_p ~ N(mu_p, var_p) on the round-logit scale, each HLTV team t an offset u_t for what the
five does together beyond its players, and each region R (EU, NA, SA, ASIA, OCE, MENA, CIS, read from the event names
a team plays in) an offset r_R. On a map,

    d = mean(s_A) + u_A + r_A - mean(s_B) - u_B - r_B,     rounds_A ~ Binomial(n, sigmoid(scale_src * d))

with the binomial likelihood tempered by w_src (rounds inside a map are not independent, and FACEIT rounds measure the
same skill with more noise). Region offsets only enter cross-region maps, so a cross-region result moves the whole
region instead of only the ten players in it: the partial pooling that lets results travel between scenes that rarely
meet. Each map is one assumed-density update: the likelihood is expanded to second order at the prior mean of d and
every state moves by its share of the prior variance of d (a diagonal Gaussian posterior, as in TrueSkill). Between
maps every state random-walks: var grows by tau2 per day, capped at its prior variance.

Further measurements of the same states:
- map ratings relative to teammates (HLTV rating; on FACEIT, kills and damage per round) say who in a five carries it;
- past kickoff prices, turned into the gap they imply, are noisy measurements of d at the kickoff of every earlier
  priced series.

Partial pooling of newcomers: a player's first appearance draws its prior from the five it appears in. Its mean is the
mean of the teammates the filter already knows (each shrunk by how well it is known) plus a newcomer offset; a five
with no known player starts at a running mean for its division (HLTV, FACEIT ESEA division) plus its own offset.

A series is priced by integrating the bo3 win probability over d ~ N(D, S), the three maps sharing the gap."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom

FIVE = 5
_DAY = np.timedelta64(1, "D")
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()
_ROUNDS = np.arange(25)


@dataclass(frozen=True)
class Config:
    v0: float = 0.12            # within-team spread of a newcomer around its known teammates
    v_cap: float = 0.30         # largest variance a player can drift back to
    tau2: float = 0.0004        # player random walk per day
    newcomer: float = -0.03     # newcomer offset relative to known teammates
    newcomer_faceit: float | None = None  # the same on FACEIT maps (None: as on HLTV)
    # A five with no known player starts at its division's running mean plus new_team_offset, with variance
    # new_team_var (None: v_cap).
    new_team_offset: float = 0.0
    new_team_var: float | None = None
    tv0: float = 0.02           # team offset prior variance
    ttau2: float = 0.0001       # team offset random walk per day
    w_hltv: float = 0.5         # likelihood tempering per map source
    w_faceit: float = 0.25
    scale_faceit: float = 1.0
    div_rate: float = 0.002     # how fast the division running means follow their players
    # Region offsets: every side carries its team's region offset (EU, NA, SA, ASIA, OCE, MENA, CIS, read from the
    # event names a team plays in), learned only from cross-region maps. A cross-region result then moves the whole
    # region, not just the ten players in it. Zero r_v0 turns it off.
    r_v0: float = 0.0
    r_tau2: float = 0.0
    # Nonzero: each player carries the region it has played most in, and a side's region term is the mean over its
    # five, so mixed rosters and players who moved scenes keep their own region.
    region_by_player: float = 0.0
    # Past kickoff prices as noisy measurements of the gap: the price is turned into the round-logit gap that would
    # give it, d_mkt ~ N(d, mkt_sd^2), at the kickoff of every earlier priced series. Zero turns it off.
    mkt_sd: float = 0.0
    # HLTV map ratings relative to teammates measure who in a five carries it: rating_p - mean(rating) ~
    # N(rating_gamma * (s_p - mean(s)), rating_sd^2), tempered by w_rating. Zero rating_gamma turns it off.
    rating_gamma: float = 0.0
    rating_sd: float = 0.3
    w_rating: float = 0.5
    w_rating_faceit: float = 0.0  # the same measurement from FACEIT kills and damage per round; zero turns it off


def map_win(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    pi = 1 / (1 + np.exp(-d))
    pmf = binom.pmf(_ROUNDS.reshape((-1,) + (1,) * d.ndim), 24, pi[None])
    return pmf[13:].sum(axis=0) + 0.5 * pmf[12]


_GAP_GRID = np.linspace(-6.0, 6.0, 2401)


def gap_of_price(p: np.ndarray) -> np.ndarray:
    """The round-logit gap whose bo3 price (no uncertainty) is p."""
    grid_price = map_win(_GAP_GRID)
    grid_price = grid_price * grid_price * (3 - 2 * grid_price)
    return np.interp(np.clip(np.asarray(p, float), 0.01, 0.99), grid_price, _GAP_GRID)


def bo3_price(mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """P(win a bo3) for a round-logit gap ~ N(mean, var), the three maps sharing the gap."""
    mean, var = np.asarray(mean, float), np.asarray(var, float)
    d = mean[:, None] + np.sqrt(np.maximum(var, 0.0))[:, None] * _GH_X[None, :]
    p = map_win(d)
    return (p * p * (3 - 2 * p)) @ _GH_W


class Filter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mu: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.last: dict[int, float] = {}
        self.n: dict[int, int] = {}
        self.tmu: dict[int, float] = {}
        self.tvar: dict[int, float] = {}
        self.tlast: dict[int, float] = {}
        self.div_mean: dict[str, float] = {}
        self.rmu: dict[str, float] = {}
        self.rvar: dict[str, float] = {}
        self.rlast: dict[str, float] = {}
        self.team_regions: dict[int, dict[str, int]] = {}
        self.player_regions: dict[int, dict[str, int]] = {}
        self.cursor = 0

    def _player(self, p: int, now: float) -> tuple[float, float]:
        v = self.var.get(p)
        if v is None:
            return None
        v = min(v + self.cfg.tau2 * (now - self.last[p]), max(v, self.cfg.v_cap))
        return self.mu[p], v

    def _team(self, t: int, now: float) -> tuple[float, float]:
        if t < 0:
            return 0.0, 0.0
        v = self.tvar.get(t)
        if v is None:
            return 0.0, self.cfg.tv0
        return self.tmu[t], min(v + self.cfg.ttau2 * (now - self.tlast[t]), max(v, self.cfg.tv0))

    def region_of(self, team: int, map_region: str) -> str:
        counts = self.team_regions.get(team)
        if counts:
            return max(sorted(counts), key=counts.get)
        return map_region

    def _region(self, r: str, now: float) -> tuple[float, float]:
        v = self.rvar.get(r)
        if v is None:
            return 0.0, self.cfg.r_v0
        return self.rmu[r], min(v + self.cfg.r_tau2 * (now - self.rlast[r]), max(v, self.cfg.r_v0))

    def five(self, five, now: float, div: str) -> tuple[list[float], list[float], int]:
        """Prior means and variances for a five, newcomers drawn from their known teammates."""
        cfg = self.cfg
        mus, vs, unknown = [], [], []
        for i, p in enumerate(five):
            st = self._player(p, now)
            if st is None:
                unknown.append(i)
                mus.append(0.0)
                vs.append(0.0)
            else:
                mus.append(st[0])
                vs.append(st[1])
        if unknown:
            known = [i for i in range(FIVE) if i not in unknown]
            if known:
                w = [1.0 / (vs[i] + cfg.v0) for i in known]
                off = cfg.newcomer if div == "hltv" or cfg.newcomer_faceit is None else cfg.newcomer_faceit
                m = sum(wi * mus[i] for wi, i in zip(w, known)) / sum(w) + off
                v = cfg.v0 + 1.0 / sum(w)
            else:
                m = self.div_mean.get(div, 0.0) + cfg.new_team_offset
                v = cfg.v_cap if cfg.new_team_var is None else cfg.new_team_var
            for i in unknown:
                mus[i], vs[i] = m, v
        return mus, vs, len(unknown)

    def _ratings(self, five, ratings, w: float) -> None:
        """Sequential contrast updates: each player's rating minus the five's mean rating."""
        cfg = self.cfg
        gam = cfg.rating_gamma
        r_mean = sum(ratings) / FIVE
        noise = cfg.rating_sd * cfg.rating_sd / w
        for j in range(FIVE):
            mus = [self.mu[p] for p in five]
            vs = [self.var[p] for p in five]
            coef = [-1.0 / FIVE] * FIVE
            coef[j] += 1.0
            pred = gam * sum(c * m for c, m in zip(coef, mus))
            s_obs = gam * gam * sum(c * c * v for c, v in zip(coef, vs)) + noise
            innov = (ratings[j] - r_mean - pred) / s_obs
            for p, c, m, v in zip(five, coef, mus, vs):
                self.mu[p] = m + gam * c * v * innov
                self.var[p] = v - gam * gam * c * c * v * v / s_obs

    def _player_region(self, p: int, team: int, map_region: str) -> str:
        counts = self.player_regions.get(p)
        if counts:
            return max(sorted(counts), key=counts.get)
        return self.region_of(team, map_region)

    def _region_terms(self, five_a, five_b, team_a: int, team_b: int, map_region: str, now: float) -> list:
        """(region, coefficient in d, mean, variance) for every region offset in play on a map."""
        if not self.cfg.r_v0:
            return []
        coef: dict[str, float] = {}
        if self.cfg.region_by_player:
            for five, team, sign in ((five_a, team_a, 1.0), (five_b, team_b, -1.0)):
                for p in five:
                    r = self._player_region(p, team, map_region)
                    if r:
                        coef[r] = coef.get(r, 0.0) + sign / FIVE
            if any(not self._player_region(p, t, map_region) for five, t in ((five_a, team_a), (five_b, team_b)) for p in five):
                return []
        else:
            ra, rb = self.region_of(team_a, map_region), self.region_of(team_b, map_region)
            if not ra or not rb or ra == rb:
                return []
            coef = {ra: 1.0, rb: -1.0}
        return [(r, c, *self._region(r, now)) for r, c in sorted(coef.items()) if abs(c) > 1e-9]

    def update(self, five_a, five_b, team_a: int, team_b: int, rounds: float, rounds_a: float, src: int, div: str,
               now: float, rating_a=None, rating_b=None, map_region: str = "") -> None:
        cfg = self.cfg
        market = src == 2
        if market and not cfg.mkt_sd:
            return
        ma, va, _ = self.five(five_a, now, div)
        mb, vb, _ = self.five(five_b, now, div)
        ua, uva = self._team(team_a, now)
        ub, uvb = self._team(team_b, now)
        terms = self._region_terms(five_a, five_b, team_a, team_b, map_region, now)
        rd = sum(c * m for _, c, m, _ in terms)
        rv = sum(c * c * v for _, c, _, v in terms)
        scale = cfg.scale_faceit if src == 1 else 1.0
        w = cfg.w_faceit if src == 1 else cfg.w_hltv
        d = scale * ((sum(ma) - sum(mb)) / FIVE + ua - ub + rd)
        S = scale * scale * ((sum(va) + sum(vb)) / (FIVE * FIVE) + uva + uvb + rv)
        if market:
            h = 1.0 / (cfg.mkt_sd * cfg.mkt_sd)
            g = (rounds_a - d) * h
        else:
            pr = 1.0 / (1.0 + math.exp(-d))
            g = w * (rounds_a - rounds * pr)
            h = w * rounds * pr * (1 - pr)
        k = 1.0 / (1.0 + S * h)
        c = scale / FIVE
        for five, mus, vs, sign in ((five_a, ma, va, 1.0), (five_b, mb, vb, -1.0)):
            for p, m, v in zip(five, mus, vs):
                self.mu[p] = m + sign * c * v * g * k
                self.var[p] = v - c * c * v * v * h * k
                self.last[p] = now
                if not market:
                    self.n[p] = self.n.get(p, 0) + 1
        for t, u, uv, sign in ((team_a, ua, uva, 1.0), (team_b, ub, uvb, -1.0)):
            if t >= 0:
                self.tmu[t] = u + sign * scale * uv * g * k
                self.tvar[t] = uv - scale * scale * uv * uv * h * k
                self.tlast[t] = now
        for r, c, rm, rvar in terms:
            self.rmu[r] = rm + c * scale * rvar * g * k
            self.rvar[r] = rvar - c * c * scale * scale * rvar * rvar * h * k
            self.rlast[r] = now
        if market:
            return
        if map_region:
            for t in (team_a, team_b):
                if t >= 0:
                    c_t = self.team_regions.setdefault(t, {})
                    c_t[map_region] = c_t.get(map_region, 0) + 1
            if cfg.region_by_player:
                for p in (*five_a, *five_b):
                    c_p = self.player_regions.setdefault(p, {})
                    c_p[map_region] = c_p.get(map_region, 0) + 1
        w_r = cfg.w_rating_faceit if src else cfg.w_rating
        if cfg.rating_gamma and w_r:
            if rating_a is not None:
                self._ratings(five_a, rating_a, w_r)
            if rating_b is not None:
                self._ratings(five_b, rating_b, w_r)
        dm = self.div_mean.get(div, 0.0)
        self.div_mean[div] = dm + cfg.div_rate * ((sum(ma) + sum(mb)) / (2 * FIVE) - dm)

    def advance(self, maps: pd.DataFrame, cutoff: np.datetime64 | None, epoch: np.datetime64) -> None:
        """Consume maps known strictly before `cutoff` (all of them when None), from where the last call stopped."""
        known = maps.known.to_numpy()
        stop = len(maps) if cutoff is None else int(np.searchsorted(known, cutoff, side="left"))
        if stop <= self.cursor:
            return
        sl = slice(self.cursor, stop)
        now = ((maps.time.to_numpy()[sl] - epoch) / _DAY).astype(float)
        cols = [maps[c].to_numpy()[sl] for c in ("five_a", "five_b", "team_a", "team_b", "rounds", "rounds_a", "src", "div",
                                                 "rating_a", "rating_b", "region")]
        for fa, fb, ta, tb, n, ra, src, div, rta, rtb, reg, t in zip(*cols, now):
            self.update(fa, fb, int(ta), int(tb), float(n), float(ra), int(src), div, float(t), rta, rtb, reg)
        self.cursor = stop

    def side(self, five, team: int, now: float, div: str = "hltv") -> tuple[float, float, int, float]:
        """Mean and variance of a side's strength (without its region), how many of its players are unknown, and
        their mean map count."""
        mus, vs, unknown = self.five(five, now, div)
        u, uv = self._team(team, now)
        nmaps = float(np.mean([self.n.get(p, 0) for p in five]))
        return sum(mus) / FIVE + u, sum(vs) / (FIVE * FIVE) + uv, unknown, nmaps

    def region_gap(self, five_a, five_b, team_a: int, team_b: int, now: float) -> tuple[float, float]:
        """Mean and variance of the region term of the gap between two sides (zero within a region)."""
        terms = self._region_terms(five_a, five_b, team_a, team_b, "", now)
        return sum(c * m for _, c, m, _ in terms), sum(c * c * v for _, c, _, v in terms)
