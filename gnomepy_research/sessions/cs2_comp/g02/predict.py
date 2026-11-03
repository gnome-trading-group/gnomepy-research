"""g02: player-level ridge-logistic ratings over HLTV and FACEIT rounds, blended with the kickoff market price."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gnomepy_research.models.cs2.hltv.config import showmatch_mask

from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.g02 import rating

C_RIDGE = 0.5
HALF_LIFE_DAYS = 500.0
WINDOW_DAYS = 900
FACEIT_WEIGHT = 0.6
MAP_SCALE = 1.3
MARKET_WINDOW = pd.Timedelta(minutes=30)
# The blend is refit before every cutoff on earlier series, so no dev months leak into it. The rating-only slope is
# larger than one because the rating price alone is under-confident.
PRIOR_BLEND = (0.0, 1.0, 0.5)
PRIOR_STRENGTH = 100.0
TRAIN_WEEKS = 12
RATING_ONLY_SLOPE = 1.8


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def kickoff_prices(data_dir: str, targets: pd.DataFrame) -> pd.Series:
    paths = bundle.target_paths(data_dir, targets.match_id)
    paths = paths.assign(t=pd.to_datetime(paths.t), kickoff=pd.to_datetime(paths.kickoff)).sort_values("t")
    near = paths[paths.t >= paths.kickoff - MARKET_WINDOW]
    return near.groupby("match_id").p_a.last()


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def past_series(data_dir: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Finished bo3 series before the cutoff with their played fives, outcome and kickoff market price."""
    h = bundle.load(data_dir, "hltv/cs2_match_history", cutoff)
    h = h[(h.bo_type == 3) & ~showmatch_mask(h)].dropna(subset=["team_a_won"])
    h = h.sort_values(["match_id", "map_position_in_series"])
    first = h.groupby("match_id").head(1).set_index("match_id")
    last = h.groupby("match_id").tail(1).set_index("match_id")
    out = pd.DataFrame({"team_a_id": first.team_a_id, "team_b_id": first.team_b_id, "team_a_lineup_ids": first.team_a_player_ids,
                        "team_b_lineup_ids": first.team_b_player_ids, "y": last.team_a_won,
                        "match_time": _naive(first.match_time.fillna(first.match_date))}).reset_index()
    out["week_cutoff"] = out.match_time.dt.normalize() - pd.to_timedelta(out.match_time.dt.weekday, unit="D")
    paths = bundle.load(data_dir, "market/series_paths", cutoff)
    paths = paths.assign(t=_naive(paths.t), kickoff=_naive(paths.kickoff)).sort_values("t")
    near = paths[(paths.t >= paths.kickoff - MARKET_WINDOW) & (paths.t < paths.kickoff)]
    out["m0"] = out.match_id.map(near.groupby("match_id").p_a.last())
    out = out.dropna(subset=["m0"])
    return out[out.team_a_lineup_ids.map(len).eq(5) & out.team_b_lineup_ids.map(len).eq(5)]


class Fitter:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._cache: dict[pd.Timestamp, tuple] = {}

    def at(self, cutoff: pd.Timestamp):
        if cutoff not in self._cache:
            maps = pd.concat([rating.hltv_maps(self.data_dir, cutoff, WINDOW_DAYS),
                              rating.faceit_maps(self.data_dir, cutoff, WINDOW_DAYS)], ignore_index=True)
            ratings, rank_coef = rating.fit_ratings(maps, cutoff, C_RIDGE, HALF_LIFE_DAYS, FACEIT_WEIGHT)
            self._cache[cutoff] = (ratings, rank_coef, rating.current_ranks(self.data_dir, cutoff))
        return self._cache[cutoff]

    def price(self, cutoff: pd.Timestamp, row) -> float:
        ratings, rank_coef, ranks = self.at(cutoff)
        return rating.price(ratings, rank_coef, list(row.team_a_lineup_ids), list(row.team_b_lineup_ids),
                            rating.rank_diff(ranks, row.team_a_id, row.team_b_id), MAP_SCALE)


def fit_blend(lm: np.ndarray, lq: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logistic blend on (1, logit m0, logit rating), ridge-shrunk towards the prior weights."""
    prior = np.array(PRIOR_BLEND)
    X = np.c_[np.ones(len(y)), lm, lq]

    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * PRIOR_STRENGTH * np.sum((w - prior) ** 2)

    return minimize(nll, prior, method="BFGS").x


def predict_week(fitter: Fitter, week: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    train = past_series(fitter.data_dir, cutoff)
    weeks = sorted(train.week_cutoff.unique())[-TRAIN_WEEKS:]
    train = train[train.week_cutoff.isin(weeks)]
    lq_train = np.array([logit(fitter.price(pd.Timestamp(r.week_cutoff), r)) for r in train.itertuples()])
    w = fit_blend(logit(train.m0.to_numpy()), lq_train, train.y.to_numpy(float))
    m0 = kickoff_prices(fitter.data_dir, week)
    q = []
    for x in week.itertuples():
        lq = logit(fitter.price(cutoff, x))
        market = m0.get(x.match_id)
        z = w[0] + w[1] * logit(market) + w[2] * lq if market is not None and np.isfinite(market) else RATING_ONLY_SLOPE * lq
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
    fitter = Fitter(args.data)
    parts = [predict_week(fitter, targets[targets.week_cutoff == c], pd.Timestamp(c)) for c in cutoffs]
    pd.concat(parts, ignore_index=True).to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
