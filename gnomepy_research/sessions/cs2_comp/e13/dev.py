"""
Development walk-forward on data before Apr 2026: weekly prices of every HLTV bo3 series from Oct 2025 to Mar 2026,
scored against their outcomes (no market), for a list of RatingConfig variants.

    python -m gnomepy_research.sessions.cs2_comp.e13.dev --data <bundle> --out <dir> [--no-faceit] '<json>' ...

Each variant is a JSON object of RatingConfig overrides, plus optional run() keywords (use_moves, sigma_map,
sigma_series, actual_fives,
core_maps, rich_recal). Writes one line per variant to summary.txt and its weekly prices to hist_<i>.parquet.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e13.data import build
from gnomepy_research.sessions.cs2_comp.e13.predict import run
from gnomepy_research.sessions.cs2_comp.e13.rating import RatingConfig

DEV_END = pd.Timestamp("2026-03-30")
DEV_FROM = pd.Timestamp("2025-10-06")
HISTORY_FROM = pd.Timestamp("2025-04-07")
RUN_KEYS = ("use_moves", "sigma_map", "sigma_series", "actual_fives", "core_maps", "rich_recal")


def log_loss(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("variants", nargs="*", default=["{}"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    targets = pd.DataFrame({"match_id": [0], "team_a_id": [0], "team_b_id": [0], "week_cutoff": [DEV_END]})
    b = build(a.data, DEV_END, a.faceit)
    lines = []
    for i, text in enumerate(a.variants):
        v = json.loads(text)
        extra = {k: v.pop(k) for k in RUN_KEYS if k in v}
        t0 = time.time()
        _, h = run(b, targets, replace(RatingConfig(), **v), history_from=HISTORY_FROM, **extra)
        d = h[(h.week >= DEV_FROM) & (h.week < DEV_END) & h.q.notna()].copy()
        y = d.y.to_numpy(float)
        loss = log_loss(d.q, y)
        on = (d.is_lan == 0).to_numpy()
        thin = (d.games_min < 40).to_numpy()
        line = (f"faceit={int(a.faceit)} {json.dumps({**v, **extra}):70s} n={len(d)} LL={loss.mean():.4f} "
                f"raw={log_loss(d.p_raw, y).mean():.4f} online={loss[on].mean():.4f} lan={loss[~on].mean():.4f} "
                f"thin={loss[thin].mean():.4f}({thin.sum()}) {time.time() - t0:.0f}s")
        print(line, flush=True)
        lines.append(line)
        d.assign(variant=i).to_parquet(a.out / f"hist_{i}.parquet", index=False)
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
