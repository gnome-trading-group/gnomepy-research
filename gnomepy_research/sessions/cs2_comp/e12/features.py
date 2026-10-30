"""One row per target series: the skill gap from the fit at its week_cutoff and each side's availability features."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e12.data import Inputs, history_practice
from gnomepy_research.sessions.cs2_comp.e12.model import Config, Lineups, SkillFit, practice, standin_share, team_state


def feature_rows(inputs: Inputs, targets: pd.DataFrame, cfg: Config, faceit: bool, data_dir: str | Path,
                 practice_features: bool = False, verbose: bool = False) -> pd.DataFrame:
    """targets: match_id, team_a, team_b, week_cutoff (plus anything to carry through, such as y). FACEIT practice
    features read the matchmaking history, which only development uses: they did not lower log loss."""
    lineups = Lineups(inputs, faceit)
    out, fives = [], []
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
                st = team_state(by_team.get(team, empty), team, c, cfg)
                rank = inputs.rank(team, c)
                mean, thin, var = fit.lineup(st["five"], rank)
                rec.update({f"skill_{side}": mean, f"thin_{side}": thin, f"var_{side}": var, f"together_{side}": st["together"],
                            f"maps90_{side}": st["maps90"], f"idle_{side}": st["days_idle"],
                            f"src_{side}": st["five_source"], f"rank_{side}": rank, f"pool_{side}": st["pool"],
                            f"standin_{side}": standin_share(inputs, team, c)})
                if faceit:
                    rec[f"linked_{side}"] = len(inputs.faceit_ids(st["five"]))
                fives.append(st["five"])
            out.append(rec)
        if verbose:
            print(f"{c:%Y-%m-%d} fit {time.time() - t0:.1f}s {fit.info} series {len(week)}", flush=True)
    df = pd.DataFrame(out)
    df["gap"] = df.skill_a - df.skill_b
    if faceit and practice_features:
        fids = [inputs.faceit_ids(f) for f in fives]
        hist = history_practice(data_dir, pd.Timestamp(targets.week_cutoff.max()), {x for f in fids for x in f})
        hist_by = {k: g for k, g in hist.groupby("faceit_id")}
        empty = hist.iloc[:0]
        feats = []
        for i, f in enumerate(fids):
            c = pd.Timestamp(df.week_cutoff.iloc[i // 2])
            h = pd.concat([hist_by.get(x, empty) for x in f]) if f else empty
            feats.append(practice(h, f, c, cfg))
        for side, k in (("a", 0), ("b", 1)):
            sub = feats[k::2]
            df[f"pair_{side}"] = [x["pair_practice"] for x in sub]
            df[f"act_{side}"] = [x["activity"] for x in sub]
            df[f"linked_{side}"] = [x["linked"] for x in sub]
    return df
