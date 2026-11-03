"""
Weekly prices from the player-skill Kalman filter (`filter.py`).

One pass over the maps in the order they became known: at each weekly cutoff the filter has consumed every map known
before it, and every series of that week is priced from that state with its announced five. A later series never
touches an earlier week's price, so a one-week bundle gives the same numbers.

FACEIT maps are another measurement of the same skills: their rounds enter the update scaled by a weight, which
scales the likelihood's gradient and curvature alike.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import PlayerSkillConfig, map_win_prob
from gnomepy_research.sessions.cs2_comp.g16.data import Inputs
from gnomepy_research.sessions.cs2_comp.g16.filter import SkillFilter

SKILL_COLUMNS = ["gap", "gap_sd", "p_map", "p_series", "strength_a", "strength_b", "sd_a", "sd_b", "new_a", "new_b",
                 "chem_a", "chem_b", "games_a", "games_b", "known_a", "known_b"]


@dataclass(frozen=True)
class SkillConfig:
    filter: PlayerSkillConfig = PlayerSkillConfig()
    faceit_weight: float = 1.0
    rating_gamma: float = 0.0
    rating_sd: float = 0.31
    division_offsets: dict = field(default_factory=dict)


def bo3(p: np.ndarray) -> np.ndarray:
    return p * p * (3 - 2 * p)


def make_filter(inputs: Inputs, cfg: SkillConfig) -> SkillFilter:
    return SkillFilter(cfg.filter, inputs.rankings, cfg.rating_gamma, cfg.rating_sd, cfg.division_offsets)


def map_arrays(inputs: Inputs, cfg: SkillConfig) -> dict:
    maps = inputs.maps
    cols = {c: maps[c].to_numpy() for c in ("known", "day", "team_a", "team_b", "five_a", "five_b", "rounds_a", "rounds_b",
                                            "rating_a", "rating_b", "division", "source")}
    cols["weight"] = np.where(cols["source"] == "faceit", cfg.faceit_weight, 1.0)
    cols["division"] = np.where(cols["source"] == "faceit", cols["division"], None)
    return cols


def observe(f: SkillFilter, cols: dict, i: int, cfg: SkillConfig) -> None:
    w = cols["weight"][i]
    if w == 0.0:
        return
    day = pd.Timestamp(cols["day"][i])
    f.observe(day, int(cols["team_a"][i]), int(cols["team_b"][i]), cols["five_a"][i], cols["five_b"][i],
              w * int(cols["rounds_a"][i]), w * int(cols["rounds_b"][i]), cols["division"][i])
    if cfg.rating_gamma:
        for side in ("a", "b"):
            if cols[f"rating_{side}"][i] is not None:
                f.observe_ratings(day, int(cols[f"team_{side}"][i]), cols[f"five_{side}"][i], cols[f"rating_{side}"][i])


def _side(f: SkillFilter, lu, side: str) -> dict:
    games = [f.games[p] for p in lu.players]
    return {f"strength_{side}": lu.strength, f"sd_{side}": f.lineup_sd(lu), f"chem_{side}": lu.kappa,
            f"new_{side}": float(sum(g < f.cfg.new_player_maps for g in games)), f"games_{side}": float(min(games)),
            f"known_{side}": float(sum(p in f._mu for p in lu.players))}


def weekly_rows(inputs: Inputs, to_price: pd.DataFrame, cfg: SkillConfig | None = None, verbose: bool = False) -> pd.DataFrame:
    """to_price: match_id, week_cutoff, team_a, team_b, lineup_a, lineup_b, standins_a, standins_b. Returns the same
    keys with SKILL_COLUMNS; a side without a full announced five gets NaN."""
    cfg = cfg or SkillConfig()
    f = make_filter(inputs, cfg)
    cols = map_arrays(inputs, cfg)
    known = cols["known"]
    done = 0
    out = []
    for cutoff in sorted(pd.unique(to_price.week_cutoff)):
        cutoff = pd.Timestamp(cutoff)
        stop = int(np.searchsorted(known, np.datetime64(cutoff), side="left"))
        for i in range(done, stop):
            observe(f, cols, i, cfg)
        done = max(done, stop)
        week = to_price[to_price.week_cutoff == cutoff]
        for r in week.itertuples(index=False):
            row = {"match_id": int(r.match_id), "week_cutoff": cutoff}
            if r.lineup_a is None or r.lineup_b is None or set(r.lineup_a) & set(r.lineup_b):
                row.update({c: np.nan for c in SKILL_COLUMNS})
            else:
                a = f.lineup(r.lineup_a, int(r.team_a), float(r.standins_a), cutoff)
                b = f.lineup(r.lineup_b, int(r.team_b), float(r.standins_b), cutoff)
                d, s2, _, _ = f.gap(a, b)
                p_map = map_win_prob(d, s2)
                row.update({"gap": d, "gap_sd": math.sqrt(max(s2, 0.0)), "p_map": p_map, "p_series": float(bo3(p_map))})
                row.update(_side(f, a, "a"))
                row.update(_side(f, b, "b"))
            out.append(row)
        if verbose:
            print(f"{cutoff:%Y-%m-%d} maps seen {done} series priced {len(week)}", flush=True)
    return pd.DataFrame(out)
