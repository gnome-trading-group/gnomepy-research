"""
Walk-forward of the Bayesian series model, and scoring of its blend with the pre-map model.

    fit:   for each month, fit on the window before it and price that month's bo3 series pre-veto;
           writes posterior_<month>.npz and prices_<month>.parquet (match_id, q, y) to <out dir>.
           With --step week, each fit ends on a Monday and prices that week's series (prices_<Monday>.parquet),
           as production refits weekly; first and last are then Mondays (YYYY-MM-DD). --step day fits through
           the day before and prices that day's series, skipping days without any.
    score: blend weights fitted on the dev months only, then the blend, the pre-map model and the market
           compared on the confirm months

The data dir holds parquet snapshots of cs2_match_history, cs2_match_lineups, cs2_player_map_stats and
cs2_team_rankings. The pre-map prices file has match_id and xgb (series price from the pre-map walk-forward).

    python -m gnomepy_research.sessions.cs2_win_probability.bayes.walk_forward fit <data dir> <out dir> 2025-09 2026-09
    python -m gnomepy_research.sessions.cs2_win_probability.bayes.walk_forward score <out dir> <pre-map prices> 2026-03

Months are fitted independently, so on AWS Batch each month is one task of a research job (the data dir uploaded
as input "data"; tasks.json is a list of ["fit", "/work/in/data", "/work/out", "<month>", "<month>"]):

    research jobs submit gnomepy_research.sessions.cs2_win_probability.bayes.walk_forward --commit <pushed branch> \
        --input data=<data dir> --array-file tasks.json
    research jobs download <run id> <out dir> --flat      # then score <out dir> as above
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.bayes.data import STAT_NAMES, prepare_maps
from gnomepy_research.models.cs2.bayes.model import BayesSeriesConfig, fit_month
from gnomepy_research.models.cs2.hltv.bayes_series import RankStrength, SeriesBlend

MARKET_EVAL = Path.home() / ".cache/gnomepy/cs2_research/backtest_6mo_eval.parquet"


STEPS = {"month": ("MS", pd.DateOffset(months=1)), "week": ("W-MON", pd.Timedelta(days=7)),
         "day": ("D", pd.Timedelta(days=1))}


def series_in(maps: pd.DataFrame, start: pd.Timestamp, length) -> pd.DataFrame:
    return maps[(maps.day >= start) & (maps.day < start + length) & (maps.bo_type == 3)].drop_duplicates("match_id")


def fit(data_dir: Path, out_dir: Path, first: str, last: str, cfg: BayesSeriesConfig, step: str = "month") -> None:
    history = pd.read_parquet(data_dir / "cs2_match_history.parquet")
    maps = prepare_maps(history, pd.read_parquet(data_dir / "cs2_match_lineups.parquet"))
    stats = pd.read_parquet(data_dir / "cs2_player_map_stats.parquet",
                            columns=["match_id", "map_name", "player_id", "team_id", "side", "rating", "eadr", "ekast",
                                     "ek", "ed"])
    ranks = RankStrength(pd.read_parquet(data_dir / "cs2_team_rankings.parquet"))
    won = maps.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() * 2 > len(w)))
    out_dir.mkdir(parents=True, exist_ok=True)
    freq, length = STEPS[step]
    for start in pd.date_range(first, last, freq=freq):
        t0 = time.time()
        label = f"{start:%Y-%m}" if step == "month" else f"{start:%Y-%m-%d}"
        series = series_in(maps, start, length)
        if series.empty:
            print(f"{label} no bo3 series, skipped", flush=True)
            continue
        posterior, diag = fit_month(maps, stats, ranks, start, cfg)
        if step == "month":
            # A weekly or daily run's dozens of posteriors at ~120 MB each aren't worth keeping; its prices are the result.
            posterior.save(out_dir / f"posterior_{label}.npz")
        q = [posterior.price(r.team_a_id, r.lineup_a, ranks(r.team_a_id, start),
                             r.team_b_id, r.lineup_b, ranks(r.team_b_id, start)) for r in series.itertuples()]
        prices = pd.DataFrame({"match_id": series.match_id.to_numpy(), "q": np.clip(q, 0.01, 0.99),
                               "y": series.match_id.map(won).to_numpy()})
        prices.to_parquet(out_dir / f"prices_{label}.parquet", index=False)
        print(f"{label} {time.time() - t0:.0f}s series {len(prices)} LL {_ll(prices.q, prices.y).mean():.4f} "
              + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in diag.items()}), flush=True)


def score(out_dir: Path, premap_path: Path, confirm_from: str, seed: int = 0) -> None:
    prices = pd.concat([pd.read_parquet(f).assign(month=f.stem.split("_")[1]) for f in sorted(out_dir.glob("prices_*.parquet"))])
    a = prices.set_index("match_id").join(pd.read_parquet(premap_path).set_index("match_id").xgb, how="inner")
    dev = (a.month < confirm_from).to_numpy()
    blend = SeriesBlend.fit(a.q[dev], a.xgb[dev], a.y[dev])
    a["blend"] = blend.predict(a.q, a.xgb)
    c = a[~dev]
    rng = np.random.default_rng(seed)
    print(f"blend weights fitted on {dev.sum()} dev series: {blend.to_dict()}")
    print(f"confirm {len(c)} series: pre-map {_ll(c.xgb, c.y).mean():.4f}  bayes {_ll(c.q, c.y).mean():.4f}  "
          f"blend {_ll(c.blend, c.y).mean():.4f}  blend - pre-map {_ci(_ll(c.blend, c.y) - _ll(c.xgb, c.y), rng)}")
    if MARKET_EVAL.exists():
        m = pd.read_parquet(MARKET_EVAL)
        m = m[m.market == "series"].drop_duplicates("match_id").set_index("match_id")[["k12", "k0"]]
        k = c.join(m, how="inner")
        for col in ("k0", "k12"):
            s = k[k[col].between(0.005, 0.995)]
            print(f"  gap to {col} ({len(s)}): pre-map {_ci(_ll(s.xgb, s.y) - _ll(s[col], s.y), rng)}  "
                  f"blend {_ci(_ll(s.blend, s.y) - _ll(s[col], s.y), rng)}")
    (out_dir / "blend.json").write_text(json.dumps(blend.to_dict()))


def _ll(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.01, 0.99)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ci(x: np.ndarray, rng: np.random.Generator) -> str:
    idx = rng.integers(0, len(x), (2000, len(x)))
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return f"{x.mean():+.4f} [{lo:+.4f},{hi:+.4f}]{'*' if lo > 0 or hi < 0 else ''}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("data_dir", type=Path)
    f.add_argument("out_dir", type=Path)
    f.add_argument("first")
    f.add_argument("last")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--no-debut-prior", dest="debut_prior", action="store_false",
                   help="give debutants the regular prior (the model before Oct 2026)")
    f.add_argument("--step", choices=sorted(STEPS), default="month", help="refit every month, week or day")
    f.add_argument("--stats", default="rating", help=f"comma-separated player stats in the skill likelihood, of {STAT_NAMES}")
    f.add_argument("--roles", action="store_true", help="per-stat AWPer and in-game-leader offsets")
    s = sub.add_parser("score")
    s.add_argument("out_dir", type=Path)
    s.add_argument("premap_prices", type=Path)
    s.add_argument("confirm_from")
    args = parser.parse_args()
    if args.command == "fit":
        cfg = BayesSeriesConfig(seed=args.seed, debut_prior=args.debut_prior, stats=tuple(args.stats.split(",")),
                                roles=args.roles)
        fit(args.data_dir, args.out_dir, args.first, args.last, cfg, args.step)
    else:
        score(args.out_dir, args.premap_prices, args.confirm_from)


if __name__ == "__main__":
    main()
