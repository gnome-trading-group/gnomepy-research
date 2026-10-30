"""Walk-forward check on pre-April 2026 HLTV series, scored against their HLTV outcomes with no market."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e01.model import Config, build_maps, fit, naive, predict


def outcomes(data: str, start: str, end: str) -> pd.DataFrame:
    h = load(data, "hltv/cs2_match_history", pd.Timestamp(end),
             ["match_id", "match_date", "team_a_id", "team_b_id", "team_a_won", "event_name", "format", "bo_type"])
    h = h[~showmatch_mask(h)]
    h = h[(h.match_date >= pd.Timestamp(start)) & (h.match_date < pd.Timestamp(end)) & (h.bo_type == 3)]
    g = h.groupby("match_id").agg(date=("match_date", "min"), team_a_id=("team_a_id", "first"),
                                  team_b_id=("team_b_id", "first"), bo_type=("bo_type", "first"),
                                  a=("team_a_won", "sum"), n=("team_a_won", "size")).reset_index()
    g = g[(g.n >= 2) & (g.n <= 3) & (g.a != g.n / 2) & (np.maximum(g.a, g.n - g.a) == 2)]
    g["y"] = (g.a == 2).astype(int)
    g["week_cutoff"] = g.date - pd.to_timedelta(g.date.dt.weekday, unit="D")
    return g.dropna(subset=["team_a_id", "team_b_id"]).reset_index(drop=True)


def run_grid(data: str, targets: pd.DataFrame, configs: dict[str, Config]) -> pd.DataFrame:
    rows = []
    cuts = sorted(targets.week_cutoff.unique())
    for cut in cuts:
        grp = targets[targets.week_cutoff == cut]
        base = Config()
        maps_by_faceit = {True: build_maps(data, naive(cut), base),
                          False: build_maps(data, naive(cut), replace(base, faceit=False))}
        for name, cfg in configs.items():
            maps = maps_by_faceit[cfg.faceit]
            fitted = fit(maps, naive(cut), cfg)
            q = predict(fitted, grp)
            rows.append(pd.DataFrame({"config": name, "match_id": grp.match_id.to_numpy(), "y": grp.y.to_numpy(),
                                      "q": q, "cut": cut}))
    return pd.concat(rows, ignore_index=True)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    res = res.assign(ll=-(res.y * np.log(res.q.clip(1e-4, 1 - 1e-4)) + (1 - res.y) * np.log(1 - res.q.clip(1e-4, 1 - 1e-4))))
    return res.groupby("config").agg(ll=("ll", "mean"), n=("ll", "size"), se=("ll", lambda x: x.std() / np.sqrt(len(x))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-04-01")
    ap.add_argument("--grid", default="smoke")
    ap.add_argument("--out")
    a = ap.parse_args()
    targets = outcomes(a.data, a.start, a.end)
    grids = {
        "smoke": {"hltv": Config(faceit=False), "faceit_w04": Config()},
        "weights": {"hltv": Config(faceit=False), **{f"faceit_w{w}": Config(faceit_weight=w) for w in (0.2, 0.4, 0.7, 1.0)}},
        "decay": {f"hl{h}_w04": Config(half_life_days=h) for h in (60.0, 120.0, 365.0)},
        "ridge": {f"C{c}_w04": Config(C=c) for c in (0.3, 1.0, 3.0)},
        "scan5": {"w03": Config(faceit_weight=0.3),
                  "w03_dev05": Config(faceit_weight=0.3, dev_scale=0.5),
                  "w03_dev02": Config(faceit_weight=0.3, dev_scale=0.2),
                  "w05_dev05": Config(faceit_weight=0.5, dev_scale=0.5),
                  "w01_dev05": Config(faceit_weight=0.1, dev_scale=0.5)},
        "team": {"w03_t05": Config(faceit_weight=0.3, team_scale=0.5),
                 "w03_t1": Config(faceit_weight=0.3, team_scale=1.0),
                 "w03_t2": Config(faceit_weight=0.3, team_scale=2.0),
                 "hltv_t1": Config(faceit=False, team_scale=1.0)},
        "rounds": {"w03": Config(faceit_weight=0.3, round_weighted=True),
                   "w03_hl1000": Config(faceit_weight=0.3, round_weighted=True, half_life_days=1000.0),
                   "w03_hl730": Config(faceit_weight=0.3, half_life_days=730.0),
                   "w03_hl1000_nr": Config(faceit_weight=0.3, half_life_days=1000.0, window_days=1000)},
        "scan4": {"hltv_hl365_win730": Config(faceit=False, half_life_days=365.0, window_days=730),
                  "hl365_w01_win730": Config(faceit_weight=0.1, half_life_days=365.0, window_days=730),
                  "hl365_w03_win730": Config(faceit_weight=0.3, half_life_days=365.0, window_days=730),
                  "hl365_w02_win730_C05": Config(faceit_weight=0.2, half_life_days=365.0, window_days=730, C=0.5),
                  "hl365_w02_win730_C2": Config(faceit_weight=0.2, half_life_days=365.0, window_days=730, C=2.0)},
        "scan3": {"hltv_hl365": Config(faceit=False, half_life_days=365.0),
                  "hl365_w02_win730": Config(faceit_weight=0.2, half_life_days=365.0, window_days=730),
                  "hl730_w02_win730": Config(faceit_weight=0.2, half_life_days=730.0, window_days=730),
                  "hl365_w05": Config(faceit_weight=0.5, half_life_days=365.0),
                  "hl365_w10": Config(faceit_weight=1.0, half_life_days=365.0)},
        "scan2": {**{f"hl{h}_w02": Config(faceit_weight=0.2, half_life_days=h) for h in (60.0, 365.0)},
                  **{f"C{c}_w02": Config(faceit_weight=0.2, C=c) for c in (0.3, 3.0)}},
    }
    t0 = time.time()
    res = run_grid(a.data, targets, grids[a.grid])
    print(summarize(res).to_string(), file=sys.stdout)
    print(f"series {len(targets)} weeks {targets.week_cutoff.nunique()} seconds {time.time() - t0:.0f}", file=sys.stdout)
    if a.out:
        res.to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
