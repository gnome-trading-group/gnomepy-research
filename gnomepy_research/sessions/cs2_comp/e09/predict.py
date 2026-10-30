"""Pre-match series probabilities from the joint HLTV + FACEIT skill filter.

    python -m gnomepy_research.sessions.cs2_comp.e09.predict --data <bundle dir> --targets <targets parquet> --out <parquet> [--no-faceit]

One chronological pass prices every target from the filter's state at its week cutoff, which holds exactly the maps a
bundle cut at that cutoff holds, so a run on a truncated bundle with only that week's targets gives the same prices.
A second pass on HLTV maps alone prices the targets whose lineups have too little FACEIT history. `--no-faceit` runs the
same model on HLTV maps alone."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e09.maps import build_maps
from gnomepy_research.sessions.cs2_comp.e09.skill import Params, RankPoints
from gnomepy_research.sessions.cs2_comp.e09.walk import walk

Q_BOUNDS = (0.01, 0.99)
MIN_COVERED = 3


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
    last = pd.to_datetime(targets.week_cutoff).max()
    maps = build_maps(data_dir, last, use_faceit)
    ranks = RankPoints(load(data_dir, "hltv/cs2_team_rankings", last))
    prices, _ = walk(data_dir, targets, final_params(use_faceit), maps, ranks)
    if use_faceit:
        hltv_only, _ = walk(data_dir, targets, final_params(False), maps, ranks)
        # Where either side has fewer than MIN_COVERED players with FACEIT history, the FACEIT games on record are mostly played
        # in other fives, and the HLTV-only price scored better than the joint one on outcomes before the tune period.
        thin = np.minimum(prices.cov_a, prices.cov_b).to_numpy() < MIN_COVERED
        prices["q"] = np.where(thin, prices.match_id.map(hltv_only.set_index("match_id").q), prices.q)
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
