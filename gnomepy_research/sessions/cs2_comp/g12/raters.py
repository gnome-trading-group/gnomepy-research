"""
Cheap, diverse online raters replayed week by week.

Every rater is an online update applied in a fixed order: the replay walks every Monday from the start of the data,
applies the rows that became known during the week before it, then snapshots features for the series of the week that
starts on that Monday. The order of updates depends only on the rows themselves, so a run that ends at an earlier
Monday reproduces the same numbers for every week it shares with a longer run.

Raters (each side's value; features are A minus B):
  team_elo      map results, team ids
  player_elo    regulation rounds, mean of the five's player ratings (HLTV maps only)
  joint_elo     the same on HLTV and FACEIT organised maps together, FACEIT maps at a smaller step
  hltv_rating   recency-weighted HLTV map rating of each of the five
  faceit_level  latest FACEIT skill level of each linked player of the five
  points        HLTV ranking points (log)
  market_elo    team ratings fitted online to past kickoff prices: what the market tends to say about these teams
plus availability context: unknown players, days idle, new players against the five last fielded, stand-ins.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.data import FIVE, Inputs

DAY = 86400.0


@dataclass(frozen=True)
class RaterConfig:
    team_k: float = 0.08
    team_k_new: float = 0.25
    team_init: float = -0.4
    player_k: float = 0.012
    player_k_new: float = 0.04
    player_init: float = -0.15
    faceit_step: float = 0.5
    rating_alpha: float = 0.08
    market_k: float = 0.25
    market_k_new: float = 0.6
    form_half_life: float = 90.0
    bias_k: float = 0.1
    r_gamma: float = 0.3
    r_sd: float = 0.6
    g_var0: float = 0.8
    extra: tuple = ()
    g_team_var0: float = 0.0
    map_var0: float = 0.05
    map_slope: float = 4.0
    fr_gamma: float = 0.3
    fr_sd: float = 0.4
    g_hltv_step: float = 1.0
    g_tau2: float = 0.00002
    g_init: float = -0.2
    g_fill_drop: float = 0.6
    g_faceit_step: float = 0.5
    g_faceit_scale: float = 1.2


def _days(t) -> float:
    return pd.Timestamp(t).value / 86400e9


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class PlayerElo:
    """Round-level player ratings; a five's strength is its mean, and a player's step shrinks as its maps add up."""

    def __init__(self, k: float, k_new: float, init: float):
        self.k, self.k_new, self.init = k, k_new, init
        self.s: dict[int, float] = {}
        self.n: dict[int, float] = {}

    def _five(self, five: tuple[int, ...]) -> np.ndarray:
        known = [self.s[p] for p in five if p in self.s]
        fill = (np.mean(known) - 0.05) if known else self.init
        return np.array([self.s.get(p, fill) for p in five])

    def update(self, five_a, five_b, rounds_a: float, rounds: float, step: float = 1.0) -> None:
        va, vb = self._five(five_a), self._five(five_b)
        d = va.mean() - vb.mean()
        g = (rounds_a - rounds / (1.0 + np.exp(-d))) / FIVE * step
        for five, v, sign in ((five_a, va, 1.0), (five_b, vb, -1.0)):
            for p, x in zip(five, v):
                n = self.n.get(p, 0.0)
                k = self.k + self.k_new / (1.0 + n / 10.0)
                self.s[p] = x + sign * k * g
                self.n[p] = n + step

    def price(self, five) -> tuple[float, int]:
        if not five:
            return self.init, FIVE
        known = [self.s[p] for p in five if p in self.s and self.n.get(p, 0) >= 3]
        unknown = FIVE - len(known)
        fill = (np.mean(known) - 0.1) if known else self.init
        return float((sum(known) + unknown * fill) / FIVE), unknown


class GaussPlayer:
    """Player skills as independent Gaussians updated by one Newton step per map on the round likelihood (a
    Glicko-style filter): steps shrink as a player's variance does, and variance grows again while a player is idle."""

    def __init__(self, var0: float, tau2: float, init: float, fill_drop: float, team_var0: float = 0.0):
        self.var0, self.tau2, self.init, self.fill_drop, self.team_var0 = var0, tau2, init, fill_drop, team_var0
        self.mu: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.t: dict[int, float] = {}
        self.tmu: dict[int, float] = {}
        self.tvar: dict[int, float] = {}
        self.tt: dict[int, float] = {}

    def _state(self, five, now: float) -> tuple[np.ndarray, np.ndarray]:
        known = [self.mu[p] for p in five if p in self.mu]
        fill = (np.mean(known) - self.fill_drop) if known else self.init
        mu = np.array([self.mu.get(p, fill) for p in five])
        var = np.array([min(self.var[p] + self.tau2 * max(now - self.t[p], 0.0), self.var0) if p in self.var else self.var0
                        for p in five])
        return mu, var

    def _team(self, team, now: float) -> tuple[float, float]:
        if team is None or not self.team_var0:
            return 0.0, 0.0
        if team not in self.tmu:
            return 0.0, self.team_var0
        return self.tmu[team], min(self.tvar[team] + self.tau2 * max(now - self.tt[team], 0.0), self.team_var0)

    def update(self, five_a, five_b, rounds_a: float, rounds: float, now: float, step: float = 1.0, scale: float = 1.0,
               team_a=None, team_b=None) -> None:
        ma, va = self._state(five_a, now)
        mb, vb = self._state(five_b, now)
        (ta, tva), (tb, tvb) = self._team(team_a, now), self._team(team_b, now)
        d = scale * (ma.mean() - mb.mean() + ta - tb)
        vd = scale * scale * ((va.sum() + vb.sum()) / (FIVE * FIVE) + tva + tvb)
        p = 1.0 / (1.0 + np.exp(-d))
        h = step * rounds * p * (1 - p)
        g = step * (rounds_a - rounds * p)
        k = scale / FIVE / (1.0 + vd * h)
        for five, m, v, sign in ((five_a, ma, va, 1.0), (five_b, mb, vb, -1.0)):
            for i, pl in enumerate(five):
                self.mu[pl] = m[i] + sign * v[i] * k * g
                self.var[pl] = max(v[i] - (v[i] * scale / FIVE) ** 2 * h / (1.0 + vd * h), 1e-4)
                self.t[pl] = now
        if self.team_var0:
            kt = scale / (1.0 + vd * h)
            for team, m, v, sign in ((team_a, ta, tva, 1.0), (team_b, tb, tvb, -1.0)):
                if team is not None:
                    self.tmu[team] = m + sign * v * kt * g
                    self.tvar[team] = max(v - (v * scale) ** 2 * h / (1.0 + vd * h), 1e-5)
                    self.tt[team] = now

    def rate(self, five, ratings, gamma: float, sd: float) -> None:
        """HLTV map ratings relative to teammates as a measurement of who carries the five:
        rating_p - mean(rating) ~ N(gamma * (s_p - mean(s)), sd^2), one player at a time."""
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

    def price(self, five, now: float, team=None) -> tuple[float, float]:
        tm, tv = self._team(team, now)
        if not five:
            return self.init + tm, self.var0
        mu, var = self._state(five, now)
        return float(mu.mean()) + tm, float(var.mean())


def _bo3(p1, p2, p3):
    return p1 * p2 + (p1 * (1 - p2) + (1 - p1) * p2) * p3


class MapOffsets:
    """Team-by-map offsets on top of the player model's gap, and a bo3 priced through a greedy veto: each side bans
    its worst map and picks its best, the decider is what is left. Returns the logit change against pricing every
    map at the plain gap."""

    def __init__(self, var0: float, slope: float):
        self.var0, self.slope = var0, slope
        self.mu: dict[tuple, float] = {}
        self.var: dict[tuple, float] = {}
        self.recent: list[tuple[float, str]] = []

    def update(self, team_a: int, team_b: int, name: str, gap: float, rounds_a: float, rounds: float, now: float) -> None:
        ka, kb = (team_a, name), (team_b, name)
        oa, ob = self.mu.get(ka, 0.0), self.mu.get(kb, 0.0)
        va, vb = self.var.get(ka, self.var0), self.var.get(kb, self.var0)
        p = 1.0 / (1.0 + np.exp(-(gap + oa - ob)))
        h = rounds * p * (1 - p)
        g = rounds_a - rounds * p
        den = 1.0 + (va + vb) * h
        self.mu[ka], self.mu[kb] = oa + va * g / den, ob - vb * g / den
        self.var[ka], self.var[kb] = va - va * va * h / den, vb - vb * vb * h / den
        self.recent.append((now, name))

    def pool(self, now: float) -> list[str]:
        counts: dict[str, int] = {}
        for t, n in reversed(self.recent):
            if now - t > 90:
                break
            counts[n] = counts.get(n, 0) + 1
        return sorted(counts, key=lambda n: (-counts[n], n))[:7]

    def adjust(self, team_a: int, team_b: int, gap: float, pool: list[str]) -> float:
        if len(pool) < 7:
            return 0.0
        base = 1.0 / (1.0 + np.exp(-self.slope * gap))
        p = {n: 1.0 / (1.0 + np.exp(-self.slope * (gap + self.mu.get((team_a, n), 0.0) - self.mu.get((team_b, n), 0.0))))
             for n in pool}
        left = list(pool)
        left.remove(min(left, key=lambda n: (p[n], n)))
        left.remove(max(left, key=lambda n: (p[n], n)))
        pick_a = max(left, key=lambda n: (p[n], n))
        left.remove(pick_a)
        pick_b = min(left, key=lambda n: (p[n], n))
        left.remove(pick_b)
        left.remove(min(left, key=lambda n: (p[n], n)))
        left.remove(max(left, key=lambda n: (p[n], n)))
        s = _bo3(p[pick_a], p[pick_b], p[left[0]])
        s0 = _bo3(base, base, base)
        return float(np.log(s / (1 - s)) - np.log(s0 / (1 - s0)))


class Replay:
    def __init__(self, inputs: Inputs, cfg: RaterConfig = RaterConfig()):
        self.inp, self.cfg = inputs, cfg
        self.team: dict[int, float] = {}
        self.team_n: dict[int, float] = {}
        self.team_form: dict[int, tuple[float, float]] = {}
        self.last_time: dict[int, pd.Timestamp] = {}
        self.last_five: dict[int, tuple] = {}
        self.player = PlayerElo(cfg.player_k, cfg.player_k_new, cfg.player_init)
        self.joint = PlayerElo(cfg.player_k, cfg.player_k_new, cfg.player_init)
        self.gauss = GaussPlayer(cfg.g_var0, cfg.g_tau2, cfg.g_init, cfg.g_fill_drop, cfg.g_team_var0)
        self.extra = []
        for name, over in cfg.extra:
            e = dataclasses.replace(cfg, **over)
            self.extra.append((name, e, GaussPlayer(e.g_var0, e.g_tau2, e.g_init, e.g_fill_drop)))
        self.maps = MapOffsets(cfg.map_var0, cfg.map_slope)
        self.gauss_h = GaussPlayer(cfg.g_var0, cfg.g_tau2, cfg.g_init, cfg.g_fill_drop)
        self.rating: dict[int, float] = {}
        self.rating_n: dict[int, int] = {}
        self.level: dict[int, float] = {}
        self.mkt: dict[int, float] = {}
        self.mkt_n: dict[int, int] = {}
        self.sub_hist: dict[int, list] = {}
        self.bias: dict[int, float] = {}
        self._m0 = {int(k): float(v) for k, v in zip(inputs.market.match_id, inputs.market.m0) if np.isfinite(v)}
        self._hi = self._fi = self._mi = self._li = self._si = self._ri = 0
        self._rank_now: dict[int, tuple[pd.Timestamp, float, float]] = {}
        self._maps = list(inputs.maps.itertuples(index=False))
        self._fmaps = list(inputs.faceit_maps.itertuples(index=False)) if inputs.faceit_maps is not None else []
        self._market = list(inputs.market.itertuples(index=False))
        lv = inputs.faceit_levels
        self._levels = ((lv.known.to_numpy("datetime64[ns]"), lv.player.to_numpy(), lv.level.to_numpy()) if lv is not None
                        else (np.array([], "datetime64[ns]"), np.array([]), np.array([])))
        rk = inputs.rankings
        self._ranks = (rk.date.to_numpy("datetime64[ns]"), rk.team_id.to_numpy(), rk.points.to_numpy(float), rk["rank"].to_numpy(float))
        self._series = list(inputs.series.sort_values(["known", "match_id"], kind="stable").itertuples(index=False))

    def _advance(self, cutoff: pd.Timestamp) -> None:
        c = self.cfg
        m = self._maps
        while self._hi < len(m) and m[self._hi].known < cutoff:
            self._apply_hltv(m[self._hi])
            self._hi += 1
        f = self._fmaps
        while self._fi < len(f) and f[self._fi].known < cutoff:
            r = f[self._fi]
            self.joint.update(r.five_a, r.five_b, r.rounds_a, r.rounds, c.faceit_step)
            for e, gm in [(c, self.gauss)] + [(e, gm) for _, e, gm in self.extra]:
                if e.g_faceit_step <= 0:
                    continue
                gm.update(r.five_a, r.five_b, r.rounds_a, r.rounds, _days(r.time), e.g_faceit_step, e.g_faceit_scale)
                if e.fr_gamma > 0:
                    gm.rate(r.five_a, r.rating_a, e.fr_gamma, e.fr_sd)
                    gm.rate(r.five_b, r.rating_b, e.fr_gamma, e.fr_sd)
            self._fi += 1
        lk, lp, lv = self._levels
        end = int(np.searchsorted(lk, np.datetime64(cutoff), side="left"))
        for i in range(self._li, end):
            self.level[int(lp[i])] = float(lv[i])
        self._li = max(self._li, end)
        mk = self._market
        while self._mi < len(mk) and mk[self._mi].kickoff < cutoff:
            r = mk[self._mi]
            if np.isfinite(r.m0):
                self._apply_market(int(r.team_a), int(r.team_b), float(r.m0))
            self._mi += 1
        s = self._series
        while self._si < len(s) and s[self._si].known < cutoff:
            r = s[self._si]
            for team, sub in ((int(r.team_a), r.sub_a), (int(r.team_b), r.sub_b)):
                self.sub_hist.setdefault(team, []).append(int(sub > 0))
            m0 = self._m0.get(int(r.match_id))
            if m0 is not None and np.isfinite(r.y):
                # Teams the market keeps mispricing: an online rating of the outcome's surprise against the kickoff price.
                a, b = int(r.team_a), int(r.team_b)
                e = c.bias_k * (r.y - m0)
                self.bias[a] = self.bias.get(a, 0.0) + e
                self.bias[b] = self.bias.get(b, 0.0) - e
            self._si += 1
        rd, rt, rp, rr = self._ranks
        end = int(np.searchsorted(rd, np.datetime64(cutoff), side="left"))
        for i in range(self._ri, end):
            self._rank_now[int(rt[i])] = (pd.Timestamp(rd[i]), float(rp[i]), float(rr[i]))
        self._ri = max(self._ri, end)

    def _apply_hltv(self, r) -> None:
        c = self.cfg
        a, b = int(r.team_a), int(r.team_b)
        now = _days(r.time)
        if c.map_var0 > 0 and isinstance(r.map_name, str):
            gap = self.gauss.price(r.five_a, now, a)[0] - self.gauss.price(r.five_b, now, b)[0]
            self.maps.update(a, b, r.map_name, gap, r.rounds_a, r.rounds, now)
        ra, rb = self.team.get(a, c.team_init), self.team.get(b, c.team_init)
        g = r.won_a - 1.0 / (1.0 + np.exp(-(ra - rb)))
        for t, v, sign in ((a, ra, 1.0), (b, rb, -1.0)):
            n = self.team_n.get(t, 0.0)
            self.team[t] = v + sign * (c.team_k + c.team_k_new / (1.0 + n / 10.0)) * g
            self.team_n[t] = n + 1
            val, last = self.team_form.get(t, (0.0, r.time))
            self.team_form[t] = (val * 0.5 ** (max((r.time - last).total_seconds(), 0.0) / DAY / c.form_half_life) + 1.0, r.time)
            self.last_time[t] = r.time
        self.last_five[a], self.last_five[b] = r.five_a, r.five_b
        self.player.update(r.five_a, r.five_b, r.rounds_a, r.rounds)
        self.joint.update(r.five_a, r.five_b, r.rounds_a, r.rounds)
        self.gauss.update(r.five_a, r.five_b, r.rounds_a, r.rounds, _days(r.time), c.g_hltv_step, 1.0, a, b)
        if c.r_gamma > 0:
            self.gauss.rate(r.five_a, r.rating_a, c.r_gamma, c.r_sd)
            self.gauss.rate(r.five_b, r.rating_b, c.r_gamma, c.r_sd)
        for _, e, gm in self.extra:
            gm.update(r.five_a, r.five_b, r.rounds_a, r.rounds, _days(r.time), e.g_hltv_step)
            if e.r_gamma > 0:
                gm.rate(r.five_a, r.rating_a, e.r_gamma, e.r_sd)
                gm.rate(r.five_b, r.rating_b, e.r_gamma, e.r_sd)
        self.gauss_h.update(r.five_a, r.five_b, r.rounds_a, r.rounds, _days(r.time))
        for five, rt in ((r.five_a, r.rating_a), (r.five_b, r.rating_b)):
            for p, x in zip(five, rt):
                if np.isfinite(x):
                    n = self.rating_n.get(p, 0)
                    alpha = max(c.rating_alpha, 1.0 / (n + 1))
                    self.rating[p] = (1 - alpha) * self.rating.get(p, x) + alpha * x
                    self.rating_n[p] = n + 1

    def _apply_market(self, a: int, b: int, m0: float) -> None:
        c = self.cfg
        x = float(_logit(m0))
        va, vb = self.mkt.get(a, 0.0), self.mkt.get(b, 0.0)
        na, nb = self.mkt_n.get(a, 0), self.mkt_n.get(b, 0)
        err = x - (va - vb)
        ka = c.market_k + c.market_k_new / (1.0 + na)
        kb = c.market_k + c.market_k_new / (1.0 + nb)
        # Split the error by how settled each side is, so a new team absorbs most of it.
        share = ka / (ka + kb)
        self.mkt[a] = va + err * share * min(1.0, ka + kb)
        self.mkt[b] = vb - err * (1 - share) * min(1.0, ka + kb)
        self.mkt_n[a], self.mkt_n[b] = na + 1, nb + 1

    def _side(self, team: int, five, cutoff: pd.Timestamp) -> dict:
        c = self.cfg
        pe, pu = self.player.price(five)
        je, ju = self.joint.price(five)
        ge, gv = self.gauss.price(five, _days(cutoff), team)
        he, hv = self.gauss_h.price(five, _days(cutoff))
        rts = [self.rating[p] for p in (five or ()) if p in self.rating]
        lvl = [self.level[p] for p in (five or ()) if p in self.level]
        last = self.last_time.get(team)
        form = self.team_form.get(team)
        rk = self._rank_now.get(team)
        fresh = rk is not None and (cutoff - rk[0]).days <= 60
        last_five = self.last_five.get(team)
        subs = self.sub_hist.get(team, [])[-5:]
        return {
            "team_elo": self.team.get(team, c.team_init), "team_n": self.team_n.get(team, 0.0),
            "form": form[0] * 0.5 ** ((cutoff - form[1]).total_seconds() / DAY / c.form_half_life) if form else 0.0,
            "player_elo": pe, "player_unknown": pu, "joint_elo": je, "joint_unknown": ju,
            "gauss": ge, "gauss_var": gv, "gauss_h": he, "gauss_h_var": hv,
            **{name: gm.price(five, _days(cutoff))[0] for name, _, gm in self.extra},
            "hltv_rating": float(np.mean(rts)) if rts else np.nan, "rating_n": len(rts),
            "faceit_level": float(np.mean(lvl)) if lvl else np.nan, "level_n": len(lvl),
            "log_points": np.log1p(rk[1]) if fresh else 0.0, "ranked": float(fresh),
            "rank": rk[2] if fresh else 300.0,
            "market_bias": self.bias.get(team, 0.0),
            "market_elo": self.mkt.get(team, 0.0), "market_n": self.mkt_n.get(team, 0),
            "idle_days": (cutoff - last).total_seconds() / DAY if last is not None else 365.0,
            "new_players": FIVE - len(set(five or ()) & set(last_five or ())) if five and last_five else FIVE,
            "standin_share": float(np.mean(subs)) if subs else 0.0,
        }

    def run(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: match_id, team_a, team_b, week_cutoff, five_a, five_b (+ anything else, passed through). Returns one
        feature row per input row, from the state at its week_cutoff."""
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
            pool = self.maps.pool(_days(c))
            for r in g.itertuples(index=False):
                rec = r._asdict()
                for side, team, five in (("a", int(r.team_a), r.five_a), ("b", int(r.team_b), r.five_b)):
                    for k, v in self._side(team, five, c).items():
                        rec[f"{k}_{side}"] = v
                rec["map_adj"] = self.maps.adjust(int(r.team_a), int(r.team_b), rec["gauss_a"] - rec["gauss_b"], pool)
                out.append(rec)
        return pd.DataFrame(out)
