"""
Series prices for the competition targets: one fit per distinct week_cutoff, each on data dated before it.

    python -m gnomepy_research.sessions.cs2_comp.e14.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--no-faceit] [--weeks 2026-04-06,2026-04-13]

--no-faceit fits the same model on HLTV maps alone (no FACEIT maps, rating rows or newcomer features).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e14.data import load_bundle
from gnomepy_research.sessions.cs2_comp.e14.model import Config, fit


def predict(data: str | Path, targets: pd.DataFrame, faceit: bool = True, cfg: Config | None = None,
            log: bool = True) -> pd.DataFrame:
    cfg = cfg or Config(faceit_priors=faceit)
    cutoffs = sorted(pd.to_datetime(targets.week_cutoff).unique())
    bundle = load_bundle(data, pd.Timestamp(cutoffs[-1]), faceit=faceit)
    out = []
    for c in cutoffs:
        t0 = time.time()
        c = pd.Timestamp(c)
        f = fit(bundle, c, cfg)
        wk = targets[pd.to_datetime(targets.week_cutoff) == c]
        q = [f.price(a, b, int(bo)) for a, b, bo in zip(wk.team_a_id, wk.team_b_id, wk.bo_type)]
        out.append(pd.DataFrame({"match_id": wk.match_id.to_numpy(), "q": np.clip(q, 0.01, 0.99), "fit_cutoff": c}))
        if log:
            print(f"{c:%Y-%m-%d} {len(wk)} targets, {len(f.player_ids)} players, kappa_f {f.kappa_f:.3f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to price (default: all)")
    args = ap.parse_args()
    targets = pd.read_parquet(args.targets)
    if args.weeks:
        keep = pd.to_datetime(args.weeks.split(","))
        targets = targets[pd.to_datetime(targets.week_cutoff).isin(keep)]
    p = predict(args.data, targets, faceit=args.faceit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(args.out, index=False)
    print(f"wrote {len(p)} predictions to {args.out}")


if __name__ == "__main__":
    main()
