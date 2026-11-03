"""Dev: rows of every market series and target, each priced by the graph fit at its own week cutoff, for walk-forward
and pooled comparisons against the market and e12.

    python -m gnomepy_research.sessions.cs2_comp.g11.dev_rows [--data <bundle>] [--targets <parquet>] [--out <dir>]
        [--weeks a,b] [--history-weeks N] [--no-faceit] [--faceit-maps] [--laplace] [--communities 32[,k2..]]
        [--tau T] [--hl H] [--sigma S] [--window D] [--seed N] [--tag name]
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g11.graph import Config
from gnomepy_research.sessions.cs2_comp.g11.pipeline import build

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/work/in/comp")
ap.add_argument("--targets")
ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR"))
ap.add_argument("--weeks")
ap.add_argument("--history-weeks", type=int, default=0)
ap.add_argument("--no-faceit", dest="faceit", action="store_false")
ap.add_argument("--faceit-maps", action="store_true")
ap.add_argument("--laplace", action="store_true")
ap.add_argument("--communities", default="32")
ap.add_argument("--tau", type=float, default=0.3)
ap.add_argument("--hl", type=float, default=300.0)
ap.add_argument("--sigma", type=float, default=0.5)
ap.add_argument("--window", type=int, default=365)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--graph-lambda", type=float, default=0.0)
ap.add_argument("--tag", default="")
a = ap.parse_args()
t = pd.read_parquet(a.targets or Path(a.data) / "targets_tune.parquet")
wc = pd.to_datetime(t.week_cutoff)
t["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
if a.weeks:
    t = t[t.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
cfg = Config(tau=a.tau, communities=tuple(int(k) for k in a.communities.split(",")), half_life_days=a.hl, sigma=a.sigma,
             window_days=a.window, laplace=a.laplace, faceit_maps=a.faceit_maps, community_seed=a.seed, graph_lambda=a.graph_lambda)
train, tgt = build(a.data, t, cfg, a.faceit, a.history_weeks, verbose=True)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = a.tag or f"{t.week_cutoff.min():%Y%m%d}"
train.to_parquet(out / f"train_{tag}.parquet", index=False)
tgt.to_parquet(out / f"targets_{tag}.parquet", index=False)
print(len(train), len(tgt))
