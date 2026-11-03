"""
Weekly batch skill fit: a penalised binomial model of every map in a trailing window, refitted at each cutoff.

    rounds_A ~ Binomial(n, sigmoid(mean(s_A) - mean(s_B) + t_A - t_B)),  map weight 0.5^(age / half_life) * w_source
    s_p ~ N(x_p . gamma, sigma^2)          x_p: [1, played on HLTV, mean ranking strength of the player's teams]
    t_T ~ N(0, team_sigma^2)               the organisation beyond its players (coaching, map pool, chemistry)

A batch fit re-reads every map in the window once teammates' skills are known (a smoother), which a sequential
filter cannot, and it is what makes Massey-style ratings beat Elo on a season of results. The fit is a point estimate
by L-BFGS from zero every week, so a week's fit is the same whether it runs alone or in a long walk-forward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.g15.data import FIVE

UNRANKED = 300.0
LOG_UNRANKED = math.log(UNRANKED)


@dataclass(frozen=True)
class FitConfig:
    window_days: int = 365
    half_life_days: float = 300.0
    sigma: float = 0.5
    team_sigma: float = 0.0
    faceit_weight: float = 0.5
    # With FACEIT maps, use each player's latest FACEIT skill level in the skill prior.
    faceit_levels: bool = False
    # HLTV map ratings relative to teammates as a second measurement of who in a five carries it:
    # rating_p - mean(rating_five) ~ N(rating_gamma * (s_p - mean(s_five)), rating_sd^2). Zero turns it off.
    rating_gamma: float = 0.0
    rating_sd: float = 0.31
    # Per-player deviation on each map, s_p + d_pm ~ N(s_p, map_sigma^2), so a five has a strength per map and a bo3 can
    # be priced through the veto. Zero turns it off.
    map_sigma: float = 0.0
    # Per-player deviation on LAN, s_p + l_p ~ N(s_p, lan_sigma^2): some players and fives travel better than others.
    # Zero turns it off.
    lan_sigma: float = 0.0
    maxiter: int = 600


@dataclass
class MapSet:
    """Decided maps with unified player ids: frame has time, team_a, team_b, rounds, rounds_a, rank_a, rank_b, source."""

    frame: pd.DataFrame
    five_a: np.ndarray
    five_b: np.ndarray
    levels: pd.DataFrame | None = None     # uid, known, level: FACEIT skill levels, when they became known

    def between(self, start: pd.Timestamp, cutoff: pd.Timestamp) -> MapSet:
        """Maps played since `start` and known before `cutoff` (a FACEIT map is known once FACEIT finalised it)."""
        t = self.frame.time.to_numpy()
        keep = (t >= np.datetime64(start)) & (self.frame.known.to_numpy() < np.datetime64(cutoff))
        levels = self.levels[self.levels.known < cutoff] if self.levels is not None else None
        return MapSet(self.frame[keep].reset_index(drop=True), self.five_a[keep], self.five_b[keep], levels)

    @staticmethod
    def from_maps(maps: pd.DataFrame, source: str = "hltv") -> MapSet:
        f = pd.DataFrame({"time": maps.time.to_numpy(), "known": maps.known.to_numpy() if "known" in maps else maps.time.to_numpy(),
                          "team_a": maps.team_a.astype(np.int64).to_numpy(),
                          "team_b": maps.team_b.astype(np.int64).to_numpy(),
                          "rounds": (maps.rounds_a + maps.rounds_b).to_numpy(float), "rounds_a": maps.rounds_a.to_numpy(float),
                          "rank_a": maps.rank_a.fillna(UNRANKED).to_numpy(float) if "rank_a" in maps else UNRANKED,
                          "rank_b": maps.rank_b.fillna(UNRANKED).to_numpy(float) if "rank_b" in maps else UNRANKED,
                          "source": source,
                          "map_name": maps.map_name.to_numpy() if "map_name" in maps else "",
                          "is_lan": maps.is_lan.to_numpy(float) if "is_lan" in maps else 0.0})
        for side in ("a", "b"):
            for j in range(FIVE):
                c = f"rating_{side}{j}"
                f[c] = maps[c].to_numpy(float) if c in maps else np.nan
        return MapSet(f, np.array(maps.five_a.tolist(), dtype=np.int64), np.array(maps.five_b.tolist(), dtype=np.int64))

    @staticmethod
    def concat(parts: list[MapSet]) -> MapSet:
        frame = pd.concat([p.frame for p in parts], ignore_index=True)
        order = np.argsort(frame.time.to_numpy(), kind="stable")
        levels = [p.levels for p in parts if p.levels is not None]
        return MapSet(frame.iloc[order].reset_index(drop=True), np.concatenate([p.five_a for p in parts])[order],
                      np.concatenate([p.five_b for p in parts])[order], levels[0] if levels else None)


@dataclass
class Fit:
    cutoff: pd.Timestamp
    players: np.ndarray
    skill: np.ndarray
    maps_played: np.ndarray
    gamma: np.ndarray
    teams: np.ndarray
    team_effect: np.ndarray
    map_names: list[str] = field(default_factory=list)
    map_dev: np.ndarray | None = None
    active_maps: list[str] = field(default_factory=list)
    lan_dev: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    def player(self, uid: int) -> tuple[float, int] | None:
        i = np.searchsorted(self.players, uid)
        if i < len(self.players) and self.players[i] == uid:
            return float(self.skill[i]), int(self.maps_played[i])
        return None

    def unknown_prior(self, rank: float) -> float:
        return float(self.gamma[0] + self.gamma[1] + self.gamma[2] * (LOG_UNRANKED - math.log(max(rank, 1.0))))

    def team(self, team: int) -> float:
        i = np.searchsorted(self.teams, team)
        return float(self.team_effect[i]) if i < len(self.teams) and self.teams[i] == team else 0.0

    def lineup_on_maps(self, five: tuple[int, ...] | None, rank: float, lan: bool = False) -> np.ndarray | None:
        """Mean strength of a five on each active map (None without map deviations or a five), on LAN if asked."""
        if self.map_dev is None or not five or not self.active_maps:
            return None
        cols = [self.map_names.index(m) for m in self.active_maps]
        out = np.zeros(len(cols))
        for p in five:
            i = np.searchsorted(self.players, p)
            if i < len(self.players) and self.players[i] == p:
                out += self.skill[i] + self.map_dev[i, cols] + (self.lan_dev[i] if lan and self.lan_dev is not None else 0.0)
            else:
                out += self.unknown_prior(rank)
        return out / FIVE

    def lineup_lan(self, five: tuple[int, ...] | None) -> float:
        """Mean LAN deviation of a five (zero without LAN deviations, a five, or for players the window never saw)."""
        if self.lan_dev is None or not five:
            return 0.0
        tot = 0.0
        for p in five:
            i = np.searchsorted(self.players, p)
            if i < len(self.players) and self.players[i] == p:
                tot += self.lan_dev[i]
        return tot / FIVE

    def lineup(self, five: tuple[int, ...] | None, rank: float) -> tuple[float, int, int]:
        """Mean skill of a five, how many of its players the window never saw, and how many it barely saw (< 5 maps)."""
        if not five:
            return self.unknown_prior(rank), FIVE, FIVE
        vals, unseen, few = [], 0, 0
        for p in five:
            known = self.player(p)
            if known is None:
                vals.append(self.unknown_prior(rank))
                unseen += 1
                few += 1
            else:
                vals.append(known[0])
                few += int(known[1] < 5)
        return float(np.mean(vals)), unseen, few


def _prior_design(ms: MapSet, players: np.ndarray) -> np.ndarray:
    f = ms.frame
    pl = np.concatenate([ms.five_a.ravel(), ms.five_b.ravel()])
    pos = np.searchsorted(players, pl)
    strength = np.concatenate([np.repeat(LOG_UNRANKED - np.log(f.rank_a.to_numpy(float)), FIVE),
                               np.repeat(LOG_UNRANKED - np.log(f.rank_b.to_numpy(float)), FIVE)])
    is_hltv = np.tile(np.repeat((f.source == "hltv").to_numpy(), FIVE), 2)
    P = len(players)
    n_hltv = np.bincount(pos[is_hltv], minlength=P)
    sum_strength = np.bincount(pos[is_hltv], weights=strength[is_hltv], minlength=P)
    return np.column_stack([np.ones(P), (n_hltv > 0).astype(float), sum_strength / np.maximum(n_hltv, 1)])


def _rating_rows(f: pd.DataFrame, ia: np.ndarray, ib: np.ndarray, w: np.ndarray, P: int,
                 cfg: FitConfig) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """One row per rated player-map of a fully rated five: L @ s is the player's skill minus the five's mean, and
    the target is the rating minus the five's mean rating, divided by rating_gamma."""
    Ls, ys, ws = [], [], []
    for side, idx in (("a", ia), ("b", ib)):
        rt = f[[f"rating_{side}{j}" for j in range(FIVE)]].to_numpy(float)
        ok = ~np.isnan(rt).any(axis=1)
        if not ok.any():
            continue
        rt, ii, ww = rt[ok], idx[ok], w[ok]
        R = len(rt)
        row = np.arange(R * FIVE)
        rows_mean = np.repeat(row, FIVE)
        cols_mean = np.repeat(ii, FIVE, axis=0).ravel()
        Ls.append(sp.csr_matrix((np.ones(R * FIVE), (row, ii.ravel())), shape=(R * FIVE, P))
                  - sp.csr_matrix((np.full(R * FIVE * FIVE, 1.0 / FIVE), (rows_mean, cols_mean)), shape=(R * FIVE, P)))
        ys.append(((rt - rt.mean(axis=1, keepdims=True)) / cfg.rating_gamma).ravel())
        ws.append(np.repeat(ww, FIVE))
    if not Ls:
        return sp.csr_matrix((0, P)), np.zeros(0), np.zeros(0)
    return sp.vstack(Ls).tocsr(), np.concatenate(ys), np.concatenate(ws)


def fit_window(ms: MapSet, cutoff: pd.Timestamp, cfg: FitConfig) -> Fit:
    f = ms.frame
    players = np.unique(np.concatenate([ms.five_a.ravel(), ms.five_b.ravel()]))
    P, M = len(players), len(f)
    ia = np.searchsorted(players, ms.five_a)
    ib = np.searchsorted(players, ms.five_b)
    rows = np.repeat(np.arange(M), FIVE)
    A = (sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, ia.ravel())), shape=(M, P))
         - sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, ib.ravel())), shape=(M, P))).tocsr()
    At = A.T.tocsr()
    use_team = cfg.team_sigma > 0
    teams = np.unique(np.concatenate([f.team_a.to_numpy(), f.team_b.to_numpy()])) if use_team else np.zeros(0, np.int64)
    T = len(teams)
    if use_team:
        ta, tb = np.searchsorted(teams, f.team_a.to_numpy()), np.searchsorted(teams, f.team_b.to_numpy())
        B = (sp.csr_matrix((np.ones(M), (np.arange(M), ta)), shape=(M, T)) - sp.csr_matrix((np.ones(M), (np.arange(M), tb)), shape=(M, T))).tocsr()
        Bt = B.T.tocsr()
    age = (cutoff - f.time).dt.total_seconds().to_numpy() / 86400.0
    w = 0.5 ** (age / cfg.half_life_days) * np.where(f.source == "faceit", cfg.faceit_weight, 1.0)
    n, ra = f.rounds.to_numpy(float), f.rounds_a.to_numpy(float)
    X = _prior_design(ms, players)
    K = X.shape[1]
    inv_var = 1.0 / cfg.sigma ** 2
    inv_team = 1.0 / cfg.team_sigma ** 2 if use_team else 0.0
    kappa = (cfg.rating_gamma / cfg.rating_sd) ** 2 if cfg.rating_gamma else 0.0
    L, yr, wr = _rating_rows(f, ia, ib, w, P, cfg) if kappa else (None, None, None)
    Lt = L.T.tocsr() if kappa else None
    use_maps = cfg.map_sigma > 0
    map_names = sorted(set(f.map_name.astype(str))) if use_maps else []
    NM = len(map_names)
    if use_maps:
        mi = np.searchsorted(np.array(map_names), f.map_name.astype(str).to_numpy())
        col_a = (ia * NM + mi[:, None]).ravel()
        col_b = (ib * NM + mi[:, None]).ravel()
        D = (sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, col_a)), shape=(M, P * NM))
             - sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, col_b)), shape=(M, P * NM))).tocsr()
        Dt = D.T.tocsr()
        inv_map = 1.0 / cfg.map_sigma ** 2
        active = _active_maps(f, cutoff)
    ND = P * NM if use_maps else 0
    use_lan = cfg.lan_sigma > 0
    if use_lan:
        E = (sp.diags(f.is_lan.to_numpy(float)) @ A).tocsr()
        Et = E.T.tocsr()
        inv_lan = 1.0 / cfg.lan_sigma ** 2
    NL = P if use_lan else 0

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        s, t, g = theta[:P], theta[P:P + T], theta[P + T:P + T + K]
        dv, lv = theta[P + T + K:P + T + K + ND], theta[P + T + K + ND:]
        d = A @ s
        if use_team:
            d = d + B @ t
        if use_maps:
            d = d + D @ dv
        if use_lan:
            d = d + E @ lv
        r = s - X @ g
        nll = -(w * (ra * d - n * np.logaddexp(0.0, d))).sum()
        obj = nll + 0.5 * inv_var * (r @ r) + 0.5 * inv_team * (t @ t) + 0.5e-2 * (g @ g)
        resid = w * (ra - n * expit(d))
        grad_s = -(At @ resid) + inv_var * r
        if kappa:
            e = yr - L @ s
            obj += 0.5 * kappa * (wr * e * e).sum()
            grad_s -= kappa * (Lt @ (wr * e))
        grad_t = -(Bt @ resid) + inv_team * t if use_team else np.zeros(0)
        grad_g = -inv_var * (X.T @ r) + 1e-2 * g
        if use_maps:
            obj += 0.5 * inv_map * (dv @ dv)
            grad_d = -(Dt @ resid) + inv_map * dv
        else:
            grad_d = np.zeros(0)
        if use_lan:
            obj += 0.5 * inv_lan * (lv @ lv)
            grad_l = -(Et @ resid) + inv_lan * lv
        else:
            grad_l = np.zeros(0)
        return obj, np.concatenate([grad_s, grad_t, grad_g, grad_d, grad_l])

    theta0 = np.zeros(P + T + K + ND + NL)
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": cfg.maxiter, "maxcor": 20, "ftol": 1e-12, "gtol": 1e-6})
    played = np.bincount(np.concatenate([ia.ravel(), ib.ravel()]), minlength=P)
    x = res.x
    return Fit(cutoff, players, x[:P], played, x[P + T:P + T + K], teams, x[P:P + T],
               map_names, x[P + T + K:P + T + K + ND].reshape(P, NM) if use_maps else None, active if use_maps else [],
               x[P + T + K + ND:] if use_lan else None,
               {"maps": M, "players": P, "teams": T, "iters": int(res.nit), "converged": bool(res.success)})


ACTIVE_POOL = 7
ACTIVE_DAYS = 90


def _active_maps(f: pd.DataFrame, cutoff: pd.Timestamp) -> list[str]:
    """The seven maps most played in the last 90 days before the cutoff: the active duty pool, which rotates."""
    recent = f[f.time >= cutoff - pd.Timedelta(days=ACTIVE_DAYS)]
    counts = recent.map_name.astype(str).value_counts()
    return sorted(counts.index[:ACTIVE_POOL].tolist())


def veto_series(gaps: np.ndarray) -> float:
    """P(team A wins a bo3) from A's round-logit gap on each active map, through the standard veto: the first team
    bans, the second bans, each picks one, the first bans, the second bans, and the last map decides. Each side bans
    its worst map and picks its best; both teams start half the time."""
    p = regulation_map_win(gaps)
    order = np.argsort(gaps)
    out = []
    for a_first in (True, False):
        left = list(order) if a_first else list(order)
        pool = list(range(len(gaps)))
        def ban(side_a: bool) -> None:
            # A bans its worst gap (B's best), B bans A's best gap.
            idx = min(pool, key=lambda i: gaps[i]) if side_a else max(pool, key=lambda i: gaps[i])
            pool.remove(idx)
        def pick(side_a: bool) -> int:
            idx = max(pool, key=lambda i: gaps[i]) if side_a else min(pool, key=lambda i: gaps[i])
            pool.remove(idx)
            return idx
        first, second = (True, False) if a_first else (False, True)
        ban(first)
        ban(second)
        m1 = pick(first)
        m2 = pick(second)
        if len(pool) > 1:
            ban(first)
        if len(pool) > 1:
            ban(second)
        m3 = pool[0]
        p1, p2, p3 = p[m1], p[m2], p[m3]
        out.append(p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3)
    return float(np.mean(out))
