"""
e12-wide-mm, fitted at a cutoff (daily in production) and priced from announced fives.

    e12      = sigmoid(LAYER . [zh, zh*thin_h, standin, idle, zf, zf*thin_f])
    logit(q) = logit(e12) + GAMMA * (mm(five A) - mm(five B))

zh and zf are the series prices of two skill fits (HLTV maps; HLTV + FACEIT maps) as logits, from each side's mean
player skill; thin counts the players a fit barely knows; standin and idle are each team's recent stand-in use and
days since it last played (A minus B). mm is a five's mean FACEIT matchmaking skill. PRIOR_SD, LAYER and GAMMA were
chosen on 2025-01-06 .. 2026-03-01 HLTV bo3 outcomes only, before any market or later month was looked at; changing
any of them makes a different model from the one that was validated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.e12.data import FIVE, UNRANKED, Inputs
from gnomepy_research.models.cs2.e12.matchmaking import MatchmakingGames
from gnomepy_research.models.cs2.e12.skill import Config, SkillFit, last_fives, series_price, standin_shares
from gnomepy_research.models.cs2.e12.tables import Tables

PRIOR_SD = 2.0
CONFIG = Config(sigma=PRIOR_SD, half_life_days=300.0, rating_gamma=0.6, rating_sd=0.31)
LAYER = {"zh": 0.149912, "zh_thin": -0.043914, "standin": -0.099792, "idle": -0.026092, "zf": 0.646569, "zf_thin": 0.00154}
GAMMA = 0.406
IDLE_UNSEEN_DAYS = 365.0
FORMAT_VERSION = 1


def _logit(p: float, eps: float) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def _expit(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def _announced(five) -> tuple[int, ...] | None:
    if five is None or len(five) != FIVE or len(set(five)) != FIVE:
        return None
    return tuple(sorted(int(p) for p in five))


@dataclass
class TeamState:
    rank: float = UNRANKED
    standin: float = 0.0
    idle_days: float = IDLE_UNSEEN_DAYS
    five_hltv: tuple[int, ...] | None = None
    five_any: tuple[int, ...] | None = None


@dataclass
class E12Model:
    cutoff: pd.Timestamp
    hltv: SkillFit
    faceit: SkillFit
    teams: dict[int, TeamState]
    mm: dict[int, float]
    info: dict = field(default_factory=dict)

    @classmethod
    def build(cls, tables: Tables, cutoff: pd.Timestamp) -> E12Model:
        """The fit at `cutoff` (naive UTC, any time of day) from rows known before it."""
        cutoff = pd.Timestamp(cutoff)
        inputs = Inputs(tables, cutoff)
        hltv = SkillFit.fit(inputs, cutoff, CONFIG, False)
        faceit = SkillFit.fit(inputs, cutoff, CONFIG, True)
        fives_hltv = last_fives(inputs, cutoff, False)
        fives_any = last_fives(inputs, cutoff, True)
        standins = standin_shares(inputs, cutoff)
        teams: dict[int, TeamState] = {}
        for team in set(fives_hltv) | set(fives_any) | set(standins) | set(inputs.rankings):
            st = TeamState(rank=inputs.rank(team, cutoff), standin=standins.get(team, 0.0))
            if team in fives_hltv:
                five, last = fives_hltv[team]
                st.five_hltv = five
                st.idle_days = float((cutoff - last).total_seconds() / 86400.0)
            if team in fives_any:
                st.five_any = fives_any[team][0]
            teams[team] = st
        mm = MatchmakingGames(tables, cutoff).fit(cutoff)
        return cls(cutoff, hltv, faceit, teams, mm, {"hltv": hltv.info, "faceit": faceit.info, "mm_players": len(mm)})

    def _side(self, team: int, five: tuple[int, ...] | None) -> tuple[TeamState, float, int, float, int]:
        st = self.teams.get(int(team), TeamState())
        mean_h, thin_h = self.hltv.lineup(five or st.five_hltv, st.rank)
        mean_f, thin_f = self.faceit.lineup(five or st.five_any, st.rank)
        return st, mean_h, thin_h, mean_f, thin_f

    def e12(self, team_a: int, five_a, team_b: int, five_b) -> float:
        """e12's price that team A wins, with each side's announced five (a side without a full five plays the five
        it fielded last)."""
        a, ha, tha, fa, tfa = self._side(team_a, _announced(five_a))
        b, hb, thb, fb, tfb = self._side(team_b, _announced(five_b))
        zh = _logit(float(series_price(ha - hb)), 1e-4)
        zf = _logit(float(series_price(fa - fb)), 1e-4)
        x = {"zh": zh, "zh_thin": zh * (tha + thb), "standin": a.standin - b.standin,
             "idle": np.log1p(a.idle_days) - np.log1p(b.idle_days), "zf": zf, "zf_thin": zf * (tfa + tfb)}
        return _expit(sum(LAYER[k] * x[k] for k in LAYER))

    def mm_gap(self, five_a, five_b) -> float:
        """Mean matchmaking skill of five A minus five B; players without matchmaking games count as zero."""
        def mean(five) -> float:
            five = [] if five is None else list(five)
            return float(np.mean([self.mm.get(int(p), 0.0) for p in five])) if five else 0.0
        return mean(five_a) - mean(five_b)

    def price(self, team_a: int, five_a, team_b: int, five_b) -> float:
        """P(team A wins the bo3)."""
        return _expit(_logit(self.e12(team_a, five_a, team_b, five_b), 1e-3) + GAMMA * self.mm_gap(five_a, five_b))

    def save(self, path: str | Path) -> None:
        tids = np.array(sorted(self.teams), dtype=np.int64)
        st = [self.teams[t] for t in tids]

        def fives(attr: str) -> tuple[np.ndarray, np.ndarray]:
            vals = [getattr(s, attr) for s in st]
            arr = np.array([v if v is not None else (0,) * FIVE for v in vals], dtype=np.int64).reshape(len(vals), FIVE)
            return arr, np.array([v is not None for v in vals])

        five_h, has_h = fives("five_hltv")
        five_any, has_any = fives("five_any")
        mm_ids = np.array(sorted(self.mm), dtype=np.int64)
        meta = {"format_version": FORMAT_VERSION, "cutoff": self.cutoff.isoformat(), "info": self.info,
                "prior_sd": PRIOR_SD, "layer": LAYER, "gamma": GAMMA}
        arrays = {"meta": np.array(json.dumps(meta)), "team_ids": tids,
                  "team_rank": np.array([s.rank for s in st]), "team_standin": np.array([s.standin for s in st]),
                  "team_idle": np.array([s.idle_days for s in st]), "five_hltv": five_h, "has_hltv": has_h,
                  "five_any": five_any, "has_any": has_any, "mm_ids": mm_ids,
                  "mm_skill": np.array([self.mm[int(p)] for p in mm_ids])}
        for name, fit in (("hltv", self.hltv), ("faceit", self.faceit)):
            arrays.update({f"{name}_players": fit.players, f"{name}_skill": fit.skill,
                           f"{name}_maps": fit.maps_played, f"{name}_gamma": fit.gamma})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> E12Model:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported e12 artifact format {meta['format_version']}")
        if meta["prior_sd"] != PRIOR_SD or meta["layer"] != LAYER or meta["gamma"] != GAMMA:
            raise ValueError("this e12 artifact was fitted with different constants than this code prices with")
        fits = {name: SkillFit(z[f"{name}_players"], z[f"{name}_skill"], z[f"{name}_maps"], z[f"{name}_gamma"],
                               meta["info"].get(name, {})) for name in ("hltv", "faceit")}
        teams = {}
        for i, t in enumerate(z["team_ids"]):
            teams[int(t)] = TeamState(rank=float(z["team_rank"][i]), standin=float(z["team_standin"][i]),
                                      idle_days=float(z["team_idle"][i]),
                                      five_hltv=tuple(int(p) for p in z["five_hltv"][i]) if z["has_hltv"][i] else None,
                                      five_any=tuple(int(p) for p in z["five_any"][i]) if z["has_any"][i] else None)
        mm = {int(p): float(s) for p, s in zip(z["mm_ids"], z["mm_skill"])}
        return cls(pd.Timestamp(meta["cutoff"]), fits["hltv"], fits["faceit"], teams, mm, meta["info"])

