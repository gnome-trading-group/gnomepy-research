"""Pre-match series probabilities from the joint HLTV + FACEIT skill filter.

    python -m gnomepy_research.sessions.cs2_comp.e09.predict --data <bundle dir> --targets <targets parquet> --out <parquet> [--no-faceit]

One chronological pass prices every target from the filter's state at its week cutoff, which holds exactly the maps a
bundle cut at that cutoff holds, so a run on a truncated bundle with only that week's targets gives the same prices.
`--no-faceit` runs the same model on HLTV maps alone."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e09.skill import Params
from gnomepy_research.sessions.cs2_comp.e09.walk import walk

Q_BOUNDS = (0.01, 0.99)


def final_params(use_faceit: bool) -> Params:
    return dataclasses.replace(Params(), use_faceit=use_faceit, **TUNED)


TUNED = dict(
    faceit_rho=0.12,
    hltv_perf_gamma=1.3, hltv_perf_sigma=0.4,
    faceit_perf_gamma=0.4, faceit_perf_sigma=0.25,
    hltv_dev_gamma=0.6, hltv_dev_sigma=0.4,
    newcomer_anchor=0.5, newcomer_sigma=0.4,
    lam=0.2,
    known_anchor=0.5, known_sigma=0.4,
)


def predict(data_dir: Path, targets: pd.DataFrame, use_faceit: bool) -> pd.DataFrame:
    prices, _ = walk(data_dir, targets, final_params(use_faceit))
    prices["q"] = np.clip(prices.q, *Q_BOUNDS)
    return prices[["match_id", "q", "fit_cutoff"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-faceit", dest="faceit", action="store_false")
    args = parser.parse_args()
    out = predict(args.data, pd.read_parquet(args.targets), args.faceit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out)} series, mean q {out.q.mean():.4f}")


if __name__ == "__main__":
    main()
