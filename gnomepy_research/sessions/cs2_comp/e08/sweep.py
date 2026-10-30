"""
Parameter sweep on the development walk-forward: each config is a comma-separated list of key=value overrides, run in
a worker pool over one shared set of maps, one JSON line of scores per config.

    python -m gnomepy_research.sessions.cs2_comp.e08.sweep --data <bundle> --workers 8 --run "kappa_f=0.5" "kappa_f=0.7,rho_f=0.05"
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e08.data import RankPoints, build_maps
from gnomepy_research.sessions.cs2_comp.e08.dev import DEV_END, TRACK_FROM, apply_overrides, dev_targets, ll
from gnomepy_research.sessions.cs2_comp.e08.series import Moves
from gnomepy_research.sessions.cs2_comp.e08.walk import walk

STATE: dict = {}


def score(out: pd.DataFrame) -> dict:
    d = ll(out.q, out.y)
    cov = np.minimum(out.n_face_a, out.n_face_b).to_numpy()
    thin = out.thin.to_numpy(bool)
    seg = {"all": d.mean(), "thin": d[thin].mean(), "rest": d[~thin].mean(), "n_thin": int(thin.sum()),
           "cov0": d[cov == 0].mean() if (cov == 0).any() else None,
           "cov1_3": d[(cov >= 1) & (cov <= 3)].mean() if ((cov >= 1) & (cov <= 3)).any() else None,
           "cov4_5": d[cov >= 4].mean() if (cov >= 4).any() else None}
    return {k: (float(v) if v is not None else None) for k, v in seg.items()}


def run(spec: str) -> dict:
    s = STATE
    params, sp = apply_overrides([x for x in spec.split(",") if x])
    try:
        out, f, ff = walk(s["maps"], s["targets"], s["moves"], s["ranks"], params, sp, track_from=TRACK_FROM)
    except (OverflowError, FloatingPointError, np.linalg.LinAlgError) as e:
        return {"spec": spec, "error": repr(e)}
    out = out.merge(s["targets"][["match_id", "y", "thin"]], on="match_id")
    res = {"spec": spec, **score(out), "round_ll_hltv": float(f.nll[0] / max(f.nmaps[0], 1)),
           "round_ll_faceit": float(f.nll[1] / max(f.nmaps[1], 1))}
    if ff is not None and ff is not f:
        res["round_ll_linked"] = float(f.linked_nll / max(f.nmaps[0], 1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else DEV_END
    STATE["targets"] = dev_targets(args.data, pd.Timestamp("2025-10-06"), end)
    STATE["maps"] = build_maps(args.data, end, use_faceit=True)
    STATE["ranks"] = RankPoints(load(args.data, "hltv/cs2_team_rankings", end))
    STATE["moves"] = Moves(load(args.data, "hltv/cs2_roster_moves", end))
    out = args.out or Path(os.environ.get("OUTPUT_DIR", ".")) / "sweep.jsonl"
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(args.workers) as pool, open(out, "a") as fh:
        for res in pool.imap_unordered(run, args.run):
            print(json.dumps(res), flush=True)
            fh.write(json.dumps(res) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
