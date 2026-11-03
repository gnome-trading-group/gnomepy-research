"""
A newcomer prior for e12's HLTV+FACEIT fit: a player the fit's window never saw starts from what e12's learned prior
says about a player with their latest FACEIT skill level (prior mean with the level terms on) instead of the level-blind
prior of an unknown player on a team of that rank. Players without a FACEIT level keep e12's prior.

`feature_rows` is e12's feature_rows with that one change in how a five is priced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g06.e12.data import FIVE, Inputs
from gnomepy_research.sessions.cs2_comp.g06.e12.features import _announced
from gnomepy_research.sessions.cs2_comp.g06.e12.model import Config, Lineups, SkillFit, standin_share, team_state

HAS_LEVEL, LEVEL = 3, 4


def latest_levels(inputs: Inputs, cutoff: pd.Timestamp) -> dict[int, float]:
    lv = inputs.levels[inputs.levels.known < cutoff]
    return lv.groupby("uid")["level"].last().to_dict()


def lineup(fit: SkillFit, five, rank: float, levels: dict[int, float]) -> tuple[float, int]:
    """e12's SkillFit.lineup, with unknown players priced from their FACEIT level when they have one."""
    if not five:
        return fit.unknown_prior(rank), FIVE
    vals, thin = [], 0
    for p in five:
        known = fit.player(p)
        if known is None:
            v = fit.unknown_prior(rank)
            if p in levels:
                v += fit.gamma[HAS_LEVEL] + fit.gamma[LEVEL] * (levels[p] - 8.0)
            vals.append(v)
            thin += 1
        else:
            vals.append(known[0])
            thin += int(known[1] < 5)
    return float(np.mean(vals)), thin


def feature_rows(inputs: Inputs, targets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """e12's HLTV+FACEIT feature rows with the newcomer prior."""
    lineups = Lineups(inputs, True)
    out = []
    for c in sorted(targets.week_cutoff.unique()):
        c = pd.Timestamp(c)
        fit = SkillFit.fit(inputs, c, cfg, True)
        levels = latest_levels(inputs, c)
        rows = lineups.at(c)
        by_team = {int(k): g for k, g in rows.groupby("team")}
        empty = rows.iloc[:0]
        for r in targets[targets.week_cutoff == c].itertuples(index=False):
            rec = r._asdict()
            for side, team in (("a", int(r.team_a)), ("b", int(r.team_b))):
                st = team_state(by_team.get(team, empty), team, c)
                rank = inputs.rank(team, c)
                last_mean, last_thin = lineup(fit, st["five"], rank, levels)
                five = _announced(rec, side)
                if five is None:
                    five, new = st["five"], 0
                else:
                    new = FIVE - len(set(five) & set(st["five"] or ()))
                mean, thin = lineup(fit, five, rank, levels)
                rec.update({f"skill_{side}": mean, f"thin_{side}": thin, f"skill_last_{side}": last_mean,
                            f"thin_last_{side}": last_thin, f"new_{side}": new, f"sub_{side}": int(rec.get(f"sub_{side}") or 0),
                            f"idle_{side}": st["days_idle"], f"src_{side}": st["five_source"], f"rank_{side}": rank,
                            f"standin_{side}": standin_share(inputs, team, c), f"linked_{side}": len(inputs.faceit_ids(five))})
            rec.pop("lineup_a", None)
            rec.pop("lineup_b", None)
            out.append(rec)
    df = pd.DataFrame(out)
    df["gap"] = df.skill_a - df.skill_b
    df["gap_last"] = df.skill_last_a - df.skill_last_b
    return df
