"""
e12's model: a round-level player-skill fit on HLTV and FACEIT maps, then a small logistic layer that adds how
prepared and settled each five is.

Skill fit (penalised maximum likelihood, one fit per weekly cutoff on a trailing window):

    rounds_A ~ Binomial(n, sigmoid(mean(s_A) - mean(s_B))),   each map weighted by 0.5^(age / half_life) * w_source
    s_p ~ N(x_p . gamma, sigma^2)

The prior mean x_p . gamma is learned with the skills: HLTV ranking of the teams a player played for, the player's
latest FACEIT skill level, and the share of the player's FACEIT maps in each ESEA division (Open is not the HLTV top
30). FACEIT maps carry a lower weight than HLTV maps: they are another measurement of the same skill with more noise
per map.

Lineups going into a series are not known before it (the target's own lineup rows are dated on the match day), so each
team plays the five it fielded last. With FACEIT on, a team's most recent FACEIT five wins when it is newer than its
last HLTV map: online teams play ESEA every week, and a roster change shows there first.

The logistic layer (fitted on Apr 2025 - Mar 2026 weekly walk-forward predictions, before every tune cutoff) takes
logit of the skill price and A-minus-B differences of availability features (recent stand-in use, days since the five
last played), and lets the skill gap's weight shrink for fives carrying players the fit barely knows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e12.data import DIVISIONS, FIVE, UNRANKED, Inputs, Maps

LOG_UNRANKED = np.log(UNRANKED)
PRIOR_DIVISIONS = DIVISIONS[:-1]


@dataclass(frozen=True)
class Config:
    window_days: int = 365
    half_life_days: float = 150.0
    sigma: float = 0.35
    faceit_weight: float = 0.5
    # Round-logit gap per unit of skill on FACEIT maps relative to HLTV maps: online rounds between weaker teams can
    # be more or less decisive than the same skill gap on HLTV.
    faceit_scale: float = 1.0


def _prior_design(maps: Maps, players: np.ndarray, inputs: Inputs, cutoff: pd.Timestamp, faceit: bool) -> np.ndarray:
    """Per player: [1, played on HLTV, mean HLTV rank strength, has FACEIT level, level - 8, division shares...]."""
    f = maps.frame
    pl = np.concatenate([maps.five_a.ravel(), maps.five_b.ravel()])
    pos = np.searchsorted(players, pl)
    strength = np.concatenate([np.repeat(LOG_UNRANKED - np.log(f.rank_a.to_numpy(float)), FIVE),
                               np.repeat(LOG_UNRANKED - np.log(f.rank_b.to_numpy(float)), FIVE)])
    is_hltv = np.tile(np.repeat((f.source == "hltv").to_numpy(), FIVE), 2)
    P = len(players)
    hltv_maps = np.bincount(pos[is_hltv], minlength=P)
    hltv_strength = np.bincount(pos[is_hltv], weights=strength[is_hltv], minlength=P)
    cols = [np.ones(P), (hltv_maps > 0).astype(float), np.divide(hltv_strength, np.maximum(hltv_maps, 1))]
    if faceit:
        lv = inputs.levels[(inputs.levels.known < cutoff) & inputs.levels.uid.isin(players)]
        last = lv.groupby("uid")["level"].last()
        level = np.full(P, np.nan)
        level[np.searchsorted(players, last.index.to_numpy())] = last.to_numpy()
        has = ~np.isnan(level)
        cols += [has.astype(float), np.where(has, level - 8.0, 0.0)]
        div = np.tile(np.repeat(f.division.to_numpy(), FIVE), 2)
        fmaps = np.bincount(pos[~is_hltv], minlength=P)
        for d in PRIOR_DIVISIONS:
            m = div == d
            cols.append(np.bincount(pos[m], minlength=P) / np.maximum(fmaps, 1))
    return np.column_stack(cols)


@dataclass
class SkillFit:
    cutoff: pd.Timestamp
    players: np.ndarray
    skill: np.ndarray
    maps_played: np.ndarray
    gamma: np.ndarray
    info: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, inputs: Inputs, cutoff: pd.Timestamp, cfg: Config, faceit: bool) -> SkillFit:
        maps = inputs.maps(cutoff, cutoff - pd.Timedelta(days=cfg.window_days), faceit)
        f = maps.frame
        players = np.unique(np.concatenate([maps.five_a.ravel(), maps.five_b.ravel()]))
        P, M = len(players), len(f)
        ia = np.searchsorted(players, maps.five_a)
        ib = np.searchsorted(players, maps.five_b)
        rows = np.repeat(np.arange(M), FIVE)
        A = (sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, ia.ravel())), shape=(M, P))
             - sp.csr_matrix((np.full(M * FIVE, 1.0 / FIVE), (rows, ib.ravel())), shape=(M, P))).tocsr()
        A = (sp.diags(np.where(f.source == "faceit", cfg.faceit_scale, 1.0)) @ A).tocsr()
        At = A.T.tocsr()
        age = (cutoff - f.time).dt.total_seconds().to_numpy() / 86400.0
        w = 0.5 ** (age / cfg.half_life_days) * np.where(f.source == "faceit", cfg.faceit_weight, 1.0)
        n = f.rounds.to_numpy(float)
        ra = f.rounds_a.to_numpy(float)
        X = _prior_design(maps, players, inputs, cutoff, faceit)
        K = X.shape[1]
        inv_var = 1.0 / cfg.sigma ** 2

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            s, g = theta[:P], theta[P:]
            d = A @ s
            r = s - X @ g
            nll = -(w * (ra * d - n * np.logaddexp(0.0, d))).sum()
            obj = nll + 0.5 * inv_var * (r @ r) + 0.5e-2 * (g @ g)
            resid = w * (ra - n * expit(d))
            grad_s = -(At @ resid) + inv_var * r
            grad_g = -inv_var * (X.T @ r) + 1e-2 * g
            return obj, np.concatenate([grad_s, grad_g])

        def hessp(theta: np.ndarray, v: np.ndarray) -> np.ndarray:
            pr = expit(A @ theta[:P])
            vs, vg = v[:P], v[P:]
            r = vs - X @ vg
            hs = At @ (w * n * pr * (1 - pr) * (A @ vs)) + inv_var * r
            hg = -inv_var * (X.T @ r) + 1e-2 * vg
            return np.concatenate([hs, hg])

        res = minimize(objective, np.zeros(P + K), jac=True, hessp=hessp, method="trust-krylov",
                       options={"maxiter": 200, "gtol": 1e-6})
        played = np.bincount(np.concatenate([ia.ravel(), ib.ravel()]), minlength=P)
        return cls(cutoff, players, res.x[:P], played, res.x[P:],
                   {"maps": M, "players": P, "iters": int(res.nit), "converged": bool(res.success)})

    def player(self, uid: int) -> tuple[float, int] | None:
        i = np.searchsorted(self.players, uid)
        if i < len(self.players) and self.players[i] == uid:
            return float(self.skill[i]), int(self.maps_played[i])
        return None

    def unknown_prior(self, team_rank: float) -> float:
        """A player the window never saw, on a team with this HLTV rank."""
        return float(self.gamma[0] + self.gamma[1] + self.gamma[2] * (LOG_UNRANKED - np.log(team_rank)))

    def lineup(self, five: tuple[int, ...] | None, team_rank: float) -> tuple[float, int]:
        """Mean skill of a five, and how many of its players the window barely knows."""
        if not five:
            return self.unknown_prior(team_rank), FIVE
        vals, thin = [], 0
        for p in five:
            known = self.player(p)
            if known is None:
                vals.append(self.unknown_prior(team_rank))
                thin += 1
            else:
                vals.append(known[0])
                thin += int(known[1] < 5)
        return float(np.mean(vals)), thin


def series_price(gap: np.ndarray) -> np.ndarray:
    p = regulation_map_win(np.asarray(gap, float))
    return p * p * (3 - 2 * p)


class Lineups:
    """The five each team last fielded before a cutoff, per source."""

    def __init__(self, inputs: Inputs, faceit: bool):
        parts = [self._long(inputs.hltv)]
        if faceit:
            parts.append(self._long(inputs.faceit_maps))
        self.long = pd.concat(parts, ignore_index=True).sort_values(["time", "match_id"], kind="stable").reset_index(drop=True)

    @staticmethod
    def _long(maps: Maps) -> pd.DataFrame:
        f = maps.frame
        a = pd.DataFrame({"team": f.team_a.to_numpy(), "time": f.time.to_numpy(), "known": f.known.to_numpy(),
                          "match_id": f.match_id.to_numpy(), "source": f.source.to_numpy(),
                          "five": [tuple(sorted(x)) for x in maps.five_a.tolist()]})
        b = pd.DataFrame({"team": f.team_b.to_numpy(), "time": f.time.to_numpy(), "known": f.known.to_numpy(),
                          "match_id": f.match_id.to_numpy(), "source": f.source.to_numpy(),
                          "five": [tuple(sorted(x)) for x in maps.five_b.tolist()]})
        out = pd.concat([a, b], ignore_index=True)
        return out[out.team >= 0]

    def at(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Team -> its rows known before the cutoff (time-sorted)."""
        return self.long[self.long.known < cutoff]


def team_state(rows: pd.DataFrame, team: int, cutoff: pd.Timestamp) -> dict:
    """The five a team last fielded and how long ago."""
    r = rows[rows.team == team]
    if r.empty:
        return {"five": None, "days_idle": 365.0, "five_source": "none"}
    last = r.iloc[-1]
    return {"five": last.five, "days_idle": float((cutoff - last.time).total_seconds() / 86400.0), "five_source": last.source}


def standin_share(inputs: Inputs, team: int, cutoff: pd.Timestamp, last_n: int = 5) -> float:
    """Share of the team's last HLTV series that it played with a stand-in."""
    s = inputs.standins
    s = s[(s.team_id == team) & (s.known < cutoff)].tail(last_n)
    return float((s.standins > 0).mean()) if len(s) else 0.0
