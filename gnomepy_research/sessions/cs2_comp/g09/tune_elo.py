"""
Tune the player power rating on HLTV bo3 series before the first dev cutoff (never on dev months): coordinate search
over EloConfig, each step's candidates scored in parallel by series log loss.

    python -m gnomepy_research.sessions.cs2_comp.g09.tune_elo [--data /work/in/comp] [--cutoff 2026-03-30]
        [--start 2025-06-01] [--faceit] [--rounds 3]
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g09.core import (EloConfig, _ids, combined_maps, elo_pass, load_faceit_maps, load_maps,
                                                         player_stats, series_outcomes, week_start)

MULT = ("k", "k_new", "n_settle", "k_faceit", "decay_days", "ind_alpha", "fast_decay", "k_faceit_low", "k_lan")
STEPS = {"k": 1.3, "k_new": 1.4, "n_settle": 1.6, "new_gap": 0.15, "rank_scale": 0.3, "margin": 0.1, "k_faceit": 1.5,
         "faceit_offset": 0.4, "decay_days": 2.0, "ind_weight": 0.5, "ind_alpha": 1.5, "swing_weight": 0.5, "k_fast": 0.05, "fast_decay": 1.5, "k_faceit_low": 1.4, "k_lan": 1.3, "rank_pull": 0.003}
STATE = {}


def _score(cfg: EloConfig) -> tuple[float, float]:
    cm, req, y = STATE["cm"], STATE["req"], STATE["y"]
    e, _ = elo_pass(cm, req, cfg, STATE["stats"])
    gap = e.set_index("match_id").elo_gap.reindex(req.match_id).to_numpy()
    best = (np.inf, 0.0)
    for s in np.arange(0.4, 1.31, 0.05):
        p = 1 / (1 + np.exp(-s * gap))
        p = np.clip(p * p * (3 - 2 * p), 0.01, 0.99)
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        best = min(best, (ll, float(s)))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--cutoff", default="2026-03-30")
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--faceit", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ind", action="store_true", help="also tune the five-carry terms")
    ap.add_argument("--fast", action="store_true", help="also tune the fast component")
    ap.add_argument("--init", default="{}", help="JSON overrides of the starting EloConfig")
    ap.add_argument("--only", default="", help="comma-separated parameters to tune (default: all enabled)")
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff)
    maps = load_maps(a.data, cutoff)
    cm = combined_maps(maps, load_faceit_maps(a.data, cutoff) if a.faceit else None)
    o = series_outcomes(maps)
    first = maps.drop_duplicates("match_id").set_index("match_id")
    ev = o[(o.bo == 3) & (o.time >= a.start) & o.y.notna()].join(first[["team_a_lineup_ids", "team_b_lineup_ids", "team_a_rank", "team_b_rank"]])
    req = pd.DataFrame({"match_id": ev.index, "snap": week_start(ev.time).to_numpy(),
                        "lineup_a": [_ids(v) for v in ev.team_a_lineup_ids], "lineup_b": [_ids(v) for v in ev.team_b_lineup_ids],
                        "rank_a": ev.team_a_rank.to_numpy(), "rank_b": ev.team_b_rank.to_numpy()})
    cur = replace(EloConfig(), **json.loads(a.init))
    if a.faceit and cur.k_faceit <= 0:
        cur = replace(cur, k_faceit=0.5)
    if not a.ind:
        cur = replace(cur, ind_weight=0.0, swing_weight=0.0)
    STATE.update(cm=cm, req=req, y=ev.y.reindex(req.match_id).to_numpy(), stats=player_stats(a.data, cutoff, cur))
    params = [p for p in STEPS if (a.faceit or p not in ("k_faceit", "faceit_offset", "k_faceit_low")) and (a.ind or p not in ("ind_weight", "ind_alpha", "swing_weight"))
              and (a.fast or p not in ("k_fast", "fast_decay"))]
    if a.only:
        params = [p for p in params if p in a.only.split(",")]
    log = []
    out = Path(os.environ.get("OUTPUT_DIR", "."))
    with Pool(min(os.cpu_count() or 1, 2 * len(params))) as pool:
        best, scale = pool.apply(_score, (cur,))
        print("start", best, scale, cur, flush=True)
        for rnd in range(a.rounds):
            improved = False
            while True:
                cands = []
                for p in params:
                    for d in (1, -1):
                        v = getattr(cur, p)
                        nv = v * STEPS[p] ** d if p in MULT else v + STEPS[p] * d
                        if p == "margin" and not 0 <= nv <= 1:
                            continue
                        cands.append(replace(cur, **{p: nv}))
                scores = pool.map(_score, cands)
                i = int(np.argmin([s[0] for s in scores]))
                if scores[i][0] < best - 1e-5:
                    best, scale, cur = scores[i][0], scores[i][1], cands[i]
                    improved = True
                    log.append({"round": rnd, "ll": best, "scale": scale, "cfg": asdict(cur)})
                    print(rnd, round(best, 5), scale, cur, flush=True)
                    (out / "tune_log.json").write_text(json.dumps(log, indent=1))
                else:
                    break
            if not improved:
                break
    (out / "tune_best.json").write_text(json.dumps({"ll": best, "scale": scale, "cfg": asdict(cur)}, indent=1))
    print("best", best, scale, cur)


if __name__ == "__main__":
    main()
