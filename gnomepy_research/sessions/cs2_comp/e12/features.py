"""One row per target series: the skill gap from the fit at its week_cutoff and each side's availability features."""
from __future__ import annotations

import time

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.e12.model import Config, Lineups, SkillFit, standin_share, team_state


def feature_rows(inputs: Inputs, targets: pd.DataFrame, cfg: Config, faceit: bool, verbose: bool = False) -> pd.DataFrame:
    """targets: match_id, team_a, team_b, week_cutoff (plus anything to carry through, such as y)."""
    lineups = Lineups(inputs, faceit)
    out = []
    for c in sorted(targets.week_cutoff.unique()):
        c = pd.Timestamp(c)
        t0 = time.time()
        fit = SkillFit.fit(inputs, c, cfg, faceit)
        rows = lineups.at(c)
        by_team = {int(k): g for k, g in rows.groupby("team")}
        empty = rows.iloc[:0]
        week = targets[targets.week_cutoff == c]
        for r in week.itertuples(index=False):
            rec = r._asdict()
            for side, team in (("a", int(r.team_a)), ("b", int(r.team_b))):
                st = team_state(by_team.get(team, empty), team, c)
                rank = inputs.rank(team, c)
                mean, thin = fit.lineup(st["five"], rank)
                rec.update({f"skill_{side}": mean, f"thin_{side}": thin, f"idle_{side}": st["days_idle"],
                            f"src_{side}": st["five_source"], f"rank_{side}": rank,
                            f"standin_{side}": standin_share(inputs, team, c)})
                if faceit:
                    rec[f"linked_{side}"] = len(inputs.faceit_ids(st["five"]))
            out.append(rec)
        if verbose:
            print(f"{c:%Y-%m-%d} fit {time.time() - t0:.1f}s {fit.info} series {len(week)}", flush=True)
    df = pd.DataFrame(out)
    df["gap"] = df.skill_a - df.skill_b
    return df
