"""Dev: feature rows of every market series and target, priced at their own week, saved for layer experiments.

    python -m gnomepy_research.sessions.cs2_comp.g11.dev_rows --data <bundle> --targets <parquet> --out <dir>
        [--weeks a,b] [--history-weeks N] [--faceit] [--tau T] [--communities K]
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
ap.add_argument("--faceit", action="store_true")
ap.add_argument("--tau", type=float, default=0.3)
ap.add_argument("--communities", type=int, default=8)
ap.add_argument("--tag", default="")
ap.add_argument("--short-hl", type=float, default=0.0)
a = ap.parse_args()
t = pd.read_parquet(a.targets or Path(a.data) / "targets_tune.parquet")
wc = pd.to_datetime(t.week_cutoff)
t["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
if a.weeks:
    t = t[t.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
cfg = Config(tau=a.tau, communities=a.communities)
short = Config(tau=a.tau, communities=a.communities, half_life_days=a.short_hl, window_days=int(3 * a.short_hl)) if a.short_hl else None
train, tgt = build(a.data, t, cfg, a.faceit, a.history_weeks, short, verbose=True)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = a.tag or f"{t.week_cutoff.min():%Y%m%d}"
train.to_parquet(out / f"train_{tag}.parquet", index=False)
tgt.to_parquet(out / f"targets_{tag}.parquet", index=False)
print(len(train), len(tgt))
