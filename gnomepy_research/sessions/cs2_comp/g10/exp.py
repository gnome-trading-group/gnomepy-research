"""Development experiments: HLTV-only skill and map-pool prices for the targets, scored against the market and the
e12 pool. python -m ...g10.exp --data D --targets T --rows ROWS [--out P]"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g10.e12_data import Inputs
from gnomepy_research.sessions.cs2_comp.g10.e12_model import Config, Lineups, SkillFit, series_price, team_state
from gnomepy_research.sessions.cs2_comp.g10.evaluate import Scorer
from gnomepy_research.sessions.cs2_comp.g10.mapveto import MapConfig, MapModel, load_maps

BASE = Config(sigma=0.5, half_life_days=300.0, rating_gamma=0.6, rating_sd=0.31)
SKILLS = {"base": BASE, "hl90": Config(sigma=0.5, half_life_days=90.0, rating_gamma=0.6, rating_sd=0.31)}
MAPS = {"m120": MapConfig(), "m60": MapConfig(half_life_days=60.0, prior_rounds=30.0),
        "m240": MapConfig(half_life_days=240.0, prior_rounds=120.0), "m120w": MapConfig(prior_rounds=20.0)}


def _five(v):
    if v is None or len(v) != 5 or len(set(v)) != 5:
        return None
    return tuple(int(p) for p in v)


def run(data_dir, targets: pd.DataFrame) -> pd.DataFrame:
    until = pd.Timestamp(targets.week_cutoff.max())
    inputs = Inputs(data_dir, until, False)
    lineups = Lineups(inputs, False)
    maps = load_maps(data_dir, until)
    out = []
    for c in sorted(targets.week_cutoff.unique()):
        c = pd.Timestamp(c)
        t0 = time.time()
        week = targets[targets.week_cutoff == c]
        rows = lineups.at(c)
        by_team = {int(k): g for k, g in rows.groupby("team")}
        rec = {"match_id": week.match_id.to_numpy()}
        for sname, cfg in SKILLS.items():
            fit = SkillFit.fit(inputs, c, cfg, False)
            gaps = []
            for r in week.itertuples(index=False):
                g = []
                for team, lu in ((int(r.team_a_id), r.team_a_lineup_ids), (int(r.team_b_id), r.team_b_lineup_ids)):
                    five = _five(lu) or team_state(by_team.get(team, rows.iloc[:0]), team, c)["five"]
                    g.append(fit.lineup(five, inputs.rank(team, c))[0])
                gaps.append(g[0] - g[1])
            gaps = np.array(gaps)
            rec[f"q_{sname}"] = series_price(gaps)
            if sname == "base":
                for mname, mcfg in MAPS.items():
                    mm = MapModel(maps, fit, c, mcfg)
                    rec[f"q_{mname}"] = [mm.series_price(g, int(a), int(b)) for g, a, b in
                                         zip(gaps, week.team_a_id, week.team_b_id)]
        out.append(pd.DataFrame(rec))
        print(f"{c:%Y-%m-%d} {len(week)} {time.time() - t0:.1f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    threadpool_limits(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = pd.to_datetime(targets.week_cutoff)
    res = run(a.data, targets)
    out = a.out or str(Path(os.environ["OUTPUT_DIR"]) / "exp.parquet")
    res.to_parquet(out, index=False)
    rows = pd.read_parquet(a.rows)
    rows = rows[rows.is_target & rows.y_true.notna() & rows.m0_true.notna()] if "y_true" in rows else rows
    f = rows[["match_id", "y", "m0", "q_e12", "month"]].merge(res, on="match_id")
    sc = Scorer(f)
    for c in [c for c in res.columns if c.startswith("q_")]:
        print(sc(f[c], c), flush=True)


if __name__ == "__main__":
    main()
