"""Walk-forward check on pre-April 2026 HLTV series, scored against their HLTV outcomes with no market."""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e01.model import Config, build_maps, fit, naive, predict

CONFIGS = {
    "hltv": Config(faceit=False),
    "w03": Config(faceit_weight=0.3),
    "w02": Config(faceit_weight=0.2),
    "w05": Config(faceit_weight=0.5),
    "w01": Config(faceit_weight=0.1),
    "w03_r": Config(faceit_weight=0.3, round_weighted=True),
    "w03_hl1000": Config(faceit_weight=0.3, half_life_days=1000.0, window_days=1000),
    "w03_hl730": Config(faceit_weight=0.3, half_life_days=730.0),
    "w03_hl1000_r": Config(faceit_weight=0.3, half_life_days=1000.0, window_days=1000, round_weighted=True),
    "w03_C07": Config(faceit_weight=0.3, C=0.7),
    "w03_C15": Config(faceit_weight=0.3, C=1.5),
    "w03_dev05": Config(faceit_weight=0.3, dev_scale=0.5),
    "w03_t05": Config(faceit_weight=0.3, team_scale=0.5),
}


def outcomes(data: str, start: str, end: str) -> pd.DataFrame:
    h = load(data, "hltv/cs2_match_history", pd.Timestamp(end),
             ["match_id", "match_date", "team_a_id", "team_b_id", "team_a_won", "event_name", "format", "bo_type"])
    h = h[~showmatch_mask(h)]
    h = h[(h.match_date >= pd.Timestamp(start)) & (h.match_date < pd.Timestamp(end)) & (h.bo_type == 3)]
    g = h.groupby("match_id").agg(date=("match_date", "min"), team_a_id=("team_a_id", "first"),
                                  team_b_id=("team_b_id", "first"), bo_type=("bo_type", "first"),
                                  a=("team_a_won", "sum"), n=("team_a_won", "size")).reset_index()
    g = g[(g.n >= 2) & (g.n <= 3) & (np.maximum(g.a, g.n - g.a) == 2)]
    g["y"] = (g.a == 2).astype(int)
    g["week_cutoff"] = g.date - pd.to_timedelta(g.date.dt.weekday, unit="D")
    return g.dropna(subset=["team_a_id", "team_b_id"]).reset_index(drop=True)


def run_grid(data: str, targets: pd.DataFrame, configs: dict[str, Config]) -> pd.DataFrame:
    rows = []
    for cut in sorted(targets.week_cutoff.unique()):
        grp = targets[targets.week_cutoff == cut]
        base = Config()
        maps_by_faceit = {True: build_maps(data, naive(cut), base),
                          False: build_maps(data, naive(cut), replace(base, faceit=False))}
        for name, cfg in configs.items():
            fitted = fit(maps_by_faceit[cfg.faceit], naive(cut), cfg)
            rows.append(pd.DataFrame({"config": name, "match_id": grp.match_id.to_numpy(), "y": grp.y.to_numpy(),
                                      "q": predict(fitted, grp), "cut": cut}))
    return pd.concat(rows, ignore_index=True)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    q = res.q.clip(1e-4, 1 - 1e-4)
    res = res.assign(ll=-(res.y * np.log(q) + (1 - res.y) * np.log(1 - q)))
    return res.groupby("config").agg(ll=("ll", "mean"), n=("ll", "size"), se=("ll", lambda x: x.std() / np.sqrt(len(x))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", default="2025-12-01")
    ap.add_argument("--end", default="2026-04-01")
    ap.add_argument("--configs", required=True, help="comma-separated names from CONFIGS")
    ap.add_argument("--tag", default="diag")
    a = ap.parse_args()
    names = a.configs.split(",")
    targets = outcomes(a.data, a.start, a.end)
    t0 = time.time()
    res = run_grid(a.data, targets, {n: CONFIGS[n] for n in names})
    print(summarize(res).to_string(), file=sys.stdout)
    print(f"series {len(targets)} weeks {targets.week_cutoff.nunique()} seconds {time.time() - t0:.0f}", file=sys.stdout)
    out_dir = Path(os.environ.get("OUTPUT_DIR", "/Users/mprey/gnome-research-work/cs2/competition/scratch_e01"))
    res.to_parquet(out_dir / f"{a.tag}.parquet", index=False)


if __name__ == "__main__":
    main()
