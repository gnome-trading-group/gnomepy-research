"""Tune rater settings on bo3 series from before the dev period (outcome log loss of a one-rater logistic fit)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.data import load_inputs
from gnomepy_research.sessions.cs2_comp.g12.raters import RaterConfig, Replay
from gnomepy_research.sessions.cs2_comp.g12.stack import fit_logistic, sigmoid, stage1_design
from gnomepy_research.sessions.cs2_comp.g12.predict import HIST_COLS

DATA = "/Users/mprey/gnome-research-work/cs2/groundup/comp/data"


def score(p: pd.DataFrame, cols: list[str]) -> float:
    X = stage1_design(p)[cols].to_numpy()
    y = p.y.to_numpy(float)
    q = sigmoid(X @ fit_logistic(X, y, 1e-3))
    q = np.clip(q, 0.01, 0.99)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--grid", required=True, help="json list of RaterConfig overrides")
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2026-03-30")
    a = ap.parse_args()
    inp = load_inputs(a.data, pd.Timestamp(a.end), True)
    s = inp.series
    rows = s[(s.bo_type == 3) & s.y.notna() & (s.week_cutoff >= a.start) & (s.week_cutoff < a.end)][HIST_COLS].assign(target=False)
    for over in json.loads(a.grid):
        t0 = time.time()
        if "extra" in over:
            over["extra"] = tuple((n, o) for n, o in over["extra"])
        cfg = dataclasses.replace(RaterConfig(), **over)
        p = Replay(inp, cfg).run(rows)
        names = [n for n, _ in cfg.extra]
        res = {k: round(score(p, c), 5) for k, c in
               {**{n: [n] for n in names}, "all_extra": ["gauss"] + names, "gauss": ["gauss"], "gauss_map": ["gauss", "map_adj"], "gauss_u": ["gauss", "gauss_unc"], "gauss_h": ["gauss_h"], "joint": ["joint_elo"],
                "player": ["player_elo"], "team": ["team_elo"]}.items()}
        print(json.dumps(over), res, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
