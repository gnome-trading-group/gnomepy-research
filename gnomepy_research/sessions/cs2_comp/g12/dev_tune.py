"""
Tune rater settings on bo3 series from before the dev period: log loss of a one-coefficient logistic fit of the outcome
on each setting's skill gap (Apr 2025 - Mar 2026 by default). Each grid entry is a list of (name, RaterConfig
overrides) raters replayed together; the stack of all of them is scored too.

    python -m gnomepy_research.sessions.cs2_comp.g12.dev_tune --data <bundle> --grid '[[["a", {"tau2": 1e-4}]]]'
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.data import load_inputs
from gnomepy_research.sessions.cs2_comp.g12.predict import HIST_COLS
from gnomepy_research.sessions.cs2_comp.g12.raters import Replay
from gnomepy_research.sessions.cs2_comp.g12.stack import fit_logistic, sigmoid


def score(p: pd.DataFrame, cols: list[str]) -> float:
    X = p[cols].to_numpy(float)
    y = p.y.to_numpy(float)
    q = np.clip(sigmoid(X @ fit_logistic(X, y, 1e-3)), 0.01, 0.99)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2026-03-30")
    ap.add_argument("--pugs", type=int, default=0, help="load FACEIT pugs with at least this many linked players")
    a = ap.parse_args()
    inp = load_inputs(a.data, pd.Timestamp(a.end), True, a.pugs)
    s = inp.series
    rows = s[(s.bo_type == 3) & s.y.notna() & (s.week_cutoff >= a.start) & (s.week_cutoff < a.end)][HIST_COLS]
    for raters in json.loads(a.grid):
        t0 = time.time()
        raters = tuple((n, o) for n, o in raters)
        p = Replay(inp, raters).run(rows.assign(target=False))
        names = [n for n, _ in raters]
        res = {n: round(score(p, [n]), 5) for n in names}
        res["stack"] = round(score(p, names), 5)
        print(json.dumps(raters), res, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
