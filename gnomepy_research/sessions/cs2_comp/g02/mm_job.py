"""Batch task: fit matchmaking skills at each requested week cutoff and save them."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g02.mm import MatchmakingTable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--targets", default="/work/in/comp/targets_tune.parquet")
    ap.add_argument("--weeks", type=int, nargs="*")
    ap.add_argument("--c", type=float, default=0.3)
    ap.add_argument("--half-life", type=float, default=300.0)
    ap.add_argument("--window", type=int, default=730)
    ap.add_argument("--rounds", action="store_true")
    ap.add_argument("--types", nargs="*")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "."))
    args = ap.parse_args()
    cutoffs = sorted(pd.read_parquet(args.targets).week_cutoff.unique())
    table = MatchmakingTable(args.data)
    for i in args.weeks if args.weeks else range(len(cutoffs)):
        ratings, level_coef = table.fit(pd.Timestamp(cutoffs[i]), args.c, args.half_life, args.window, args.rounds,
                                        tuple(args.types) if args.types else None)
        df = pd.DataFrame({"player_id": list(ratings), "mm": list(ratings.values())})
        df["level_coef"] = level_coef
        act = table.activity(pd.Timestamp(cutoffs[i]))
        df["n30"] = df.player_id.map(act).fillna(0)
        df.to_parquet(Path(args.out) / f"mm_{args.tag}_{pd.Timestamp(cutoffs[i]).date()}.parquet", index=False)
        print(i, len(df), level_coef, flush=True)


if __name__ == "__main__":
    main()
