"""g07 refreshed for the Aug-Oct confirmation: the same model as `predict`, with its filter configurations re-selected on
all data before 2026-08-01.

    python -m gnomepy_research.sessions.cs2_comp.g07.predict_refit --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-08-03,2026-08-10]

Selection was the procedure that chose `predict.CONFIGS`, with a longer window: a random sweep of filter
configurations (`dev.py --history-end 2026-08-01`), each scored by the recalibrated log loss on every HLTV bo3 from
Jan 2025 to Jul 2026 plus pool - market on the Mar-Jul 2026 market series. The ensemble is the two best by that sum
and the best by pool - market. Every read in the sweep was cut at 2026-08-01, so these constants know nothing of
later months; the layer is refit weekly exactly as in `predict`."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g07.data import naive
from gnomepy_research.sessions.cs2_comp.g07.model import Config
from gnomepy_research.sessions.cs2_comp.g07.predict import predict

_COMMON = dict(ttau2=0.0, scale_faceit=1.0, rating_gamma=0.45, rating_sd=0.4, region_by_player=1.0, new_team_var=0.6)
REFIT_CONFIGS = [
    Config(**_COMMON, v0=0.3, v_cap=2.0, tau2=4.5e-05, newcomer=-0.6, tv0=0.0025, w_hltv=0.6, w_faceit=0.75,
           w_rating=0.25, w_rating_faceit=0.15, r_v0=0.04, r_tau2=3e-05, new_team_offset=-0.6, mkt_sd=0.3),
    Config(**_COMMON, v0=0.3, v_cap=3.0, tau2=4.5e-05, newcomer=-0.4, newcomer_faceit=-0.5, tv0=0.001, w_hltv=0.6,
           w_faceit=0.9, w_rating=0.15, w_rating_faceit=0.25, r_v0=0.02, r_tau2=1e-04, new_team_offset=-0.6, mkt_sd=0.45),
    Config(**_COMMON, v0=0.15, v_cap=1.4, tau2=3e-05, newcomer=-0.6, tv0=0.001, w_hltv=0.6, w_faceit=0.75,
           w_rating=0.1, w_rating_faceit=0.25, r_v0=0.04, r_tau2=3e-05, new_team_offset=-0.3, mkt_sd=0.3),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = naive(targets.week_cutoff)
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    res, _, _ = predict(a.data, targets, REFIT_CONFIGS)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out, index=False)
    print(f"wrote {len(res)} predictions to {a.out}")


if __name__ == "__main__":
    main()
