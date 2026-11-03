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
    g_var0: float = 0.5
    g_tau2: float = 0.00002
    g_init: float = -0.2
    g_fill_drop: float = 0.5
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

    def __init__(self, var0: float, tau2: float, init: float, fill_drop: float):
        self.var0, self.tau2, self.init, self.fill_drop = var0, tau2, init, fill_drop
        self.mu: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.t: dict[int, float] = {}

    def _state(self, five, now: float) -> tuple[np.ndarray, np.ndarray]:
        known = [self.mu[p] for p in five if p in self.mu]
        fill = (np.mean(known) - self.fill_drop) if known else self.init
        mu = np.array([self.mu.get(p, fill) for p in five])
        var = np.array([min(self.var[p] + self.tau2 * max(now - self.t[p], 0.0), self.var0) if p in self.var else self.var0
                        for p in five])
        return mu, var

    def update(self, five_a, five_b, rounds_a: float, rounds: float, now: float, step: float = 1.0, scale: float = 1.0) -> None:
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

    def price(self, five, now: float) -> tuple[float, float]:
        if not five:
            return self.init, self.var0
        mu, var = self._state(five, now)
        return float(mu.mean()), float(var.mean())


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
        self.gauss = GaussPlayer(cfg.g_var0, cfg.g_tau2, cfg.g_init, cfg.g_fill_drop)
        self.gauss_h = GaussPlayer(cfg.g_var0, cfg.g_tau2, cfg.g_init, cfg.g_fill_drop)
        self.rating: dict[int, float] = {}
        self.rating_n: dict[int, int] = {}
        self.level: dict[int, float] = {}
        self.mkt: dict[int, float] = {}
        self.mkt_n: dict[int, int] = {}
        self.sub_hist: dict[int, list] = {}
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
            self.gauss.update(r.five_a, r.five_b, r.rounds_a, r.rounds, _days(r.time), c.g_faceit_step, c.g_faceit_scale)
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
            self._si += 1
        rd, rt, rp, rr = self._ranks
        end = int(np.searchsorted(rd, np.datetime64(cutoff), side="left"))
        for i in range(self._ri, end):
            self._rank_now[int(rt[i])] = (pd.Timestamp(rd[i]), float(rp[i]), float(rr[i]))
        self._ri = max(self._ri, end)

    def _apply_hltv(self, r) -> None:
        c = self.cfg
        a, b = int(r.team_a), int(r.team_b)
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
        self.gauss.update(r.five_a, r.five_b, r.rounds_a, r.rounds, _days(r.time))
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
        ge, gv = self.gauss.price(five, _days(cutoff))
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
            "hltv_rating": float(np.mean(rts)) if rts else np.nan, "rating_n": len(rts),
            "faceit_level": float(np.mean(lvl)) if lvl else np.nan, "level_n": len(lvl),
            "log_points": np.log1p(rk[1]) if fresh else 0.0, "ranked": float(fresh),
            "rank": rk[2] if fresh else 300.0,
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
            for r in g.itertuples(index=False):
                rec = r._asdict()
                for side, team, five in (("a", int(r.team_a), r.five_a), ("b", int(r.team_b), r.five_b)):
                    for k, v in self._side(team, five, c).items():
                        rec[f"{k}_{side}"] = v
                out.append(rec)
        return pd.DataFrame(out)
