"""
Two filters, one per source, and the features a series gets from them.

The HLTV filter sees only HLTV maps. The FACEIT filter sees only FACEIT maps, so its skills sit on the FACEIT scale
(an ESEA division is not the HLTV top 30) and a linked player's HLTV and FACEIT skills are separate numbers. The stage
that combines them (stack.py) learns how far a FACEIT skill moves an HLTV match, and a lineup with no FACEIT history
contributes nothing instead of a level-shifted guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gnomepy_research.sessions.cs2_comp.e07.data import RankPoints, Streams
from gnomepy_research.sessions.cs2_comp.e07.filter import Filter, FilterConfig

FIVE_DAYS = 180.0
ACTIVITY_DAYS = 60.0


@dataclass(frozen=True)
class ModelConfig:
    hltv: FilterConfig = field(default_factory=lambda: FilterConfig(use_faceit=False))
    faceit: FilterConfig = field(default_factory=lambda: FilterConfig(use_hltv=False, rank_prior_faceit=False,
                                                                       faceit_prior=0.0))
    use_faceit: bool = True


class Model:
    def __init__(self, cfg: ModelConfig, streams: Streams, rank_points: RankPoints):
        self.cfg = cfg
        self.h = Filter(cfg.hltv, streams, rank_points)
        self.f = Filter(cfg.faceit, streams, rank_points) if cfg.use_faceit else None

    def advance(self, day: float) -> None:
        self.h.advance(day)
        if self.f is not None:
            self.f.advance(day)

    def features(self, team_a: int, team_b: int, t: float, cutoff_day: float) -> dict:
        q, mean, var = self.h.price(team_a, team_b, t, cutoff_day)
        row = {"q_h": q, "mean_h": mean, "var_h": var}
        for side, team in (("a", team_a), ("b", team_b)):
            known = self.h.last_five.get(team)
            ids = None if known is None else known[0]
            row[f"five_h_{side}"] = 0 if ids is None else self.h.five_maps(ids, t, FIVE_DAYS)
            row[f"act_h_{side}"] = 0.0 if ids is None else self.h.activity(ids, t, ACTIVITY_DAYS)
            row[f"days_h_{side}"] = 999.0 if known is None else t - known[1]
            row[f"tm90_{side}"] = self.h.team_maps(team, t, 90.0)
            if self.f is None:
                continue
            s, v, n = (0.0, 0.0, 0) if ids is None else self.f.profile(ids, t)
            row[f"f_s_{side}"], row[f"f_v_{side}"], row[f"f_n_{side}"] = s, v, n
            row[f"five_f_{side}"] = 0 if ids is None else self.f.five_maps(ids, t, FIVE_DAYS)
            row[f"act_f_{side}"] = 0.0 if ids is None else self.f.activity(ids, t, ACTIVITY_DAYS)
        return row
