"""g02: player-level ridge-logistic ratings over HLTV and FACEIT rounds, blended with the kickoff market price."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.g02 import rating

C_RIDGE = 0.5
HALF_LIFE_DAYS = 500.0
WINDOW_DAYS = 900
FACEIT_WEIGHT = 0.6
MAP_SCALE = 1.3
MARKET_WINDOW = pd.Timedelta(minutes=30)
# Blend weights are the dev logistic fit of the outcome on logit(m0) and logit(rating price); the rating-only slope
# (without a market) is larger because the rating price alone is under-confident.
W_MARKET, W_RATING, INTERCEPT = 0.67, 0.97, 0.03
RATING_ONLY_SLOPE = 1.8


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def kickoff_prices(data_dir: str, targets: pd.DataFrame) -> pd.Series:
    paths = bundle.target_paths(data_dir, targets.match_id)
    paths = paths.assign(t=pd.to_datetime(paths.t), kickoff=pd.to_datetime(paths.kickoff)).sort_values("t")
    near = paths[paths.t >= paths.kickoff - MARKET_WINDOW]
    return near.groupby("match_id").p_a.last()


def predict_week(data_dir: str, week: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    maps = pd.concat([rating.hltv_maps(data_dir, cutoff, WINDOW_DAYS), rating.faceit_maps(data_dir, cutoff, WINDOW_DAYS)],
                     ignore_index=True)
    ratings, rank_coef = rating.fit_ratings(maps, cutoff, C_RIDGE, HALF_LIFE_DAYS, FACEIT_WEIGHT)
    ranks = rating.current_ranks(data_dir, cutoff)
    m0 = kickoff_prices(data_dir, week)
    q = []
    for x in week.itertuples():
        p = rating.price(ratings, rank_coef, list(x.team_a_lineup_ids), list(x.team_b_lineup_ids),
                         rating.rank_diff(ranks, x.team_a_id, x.team_b_id), MAP_SCALE)
        market = m0.get(x.match_id)
        z = (INTERCEPT + W_MARKET * logit(market) + W_RATING * logit(p)) if market is not None and np.isfinite(market) \
            else RATING_ONLY_SLOPE * logit(p)
        q.append(1 / (1 + np.exp(-z)))
    return pd.DataFrame({"match_id": week.match_id.to_numpy(), "q": q, "fit_cutoff": cutoff})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", type=int, nargs="*", help="indices of the distinct week cutoffs to run (default all)")
    args = ap.parse_args()
    targets = pd.read_parquet(args.targets)
    cutoffs = sorted(targets.week_cutoff.unique())
    if args.weeks:
        cutoffs = [cutoffs[i] for i in args.weeks]
    parts = [predict_week(args.data, targets[targets.week_cutoff == c], pd.Timestamp(c)) for c in cutoffs]
    pd.concat(parts, ignore_index=True).to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
