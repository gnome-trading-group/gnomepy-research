"""Competition entry point: predictions from data before each target's week cutoff.

    python -m gnomepy_research.sessions.cs2_comp.e01.predict --data <bundle> --targets <parquet> --out <parquet> [--no-faceit]
"""
from __future__ import annotations

import argparse
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e01.model import (
    Config,
    build_maps,
    build_performance,
    fit,
    naive,
    predict,
)


def run(data: str, targets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t = targets.assign(cut=targets.week_cutoff.map(naive))
    parts = []
    for cut, grp in t.groupby("cut"):
        fitted = fit(build_maps(data, cut, cfg), build_performance(data, cut, cfg), cut, cfg)
        parts.append(pd.DataFrame({"match_id": grp.match_id.to_numpy(), "q": predict(fitted, grp),
                                   "fit_cutoff": cut}))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", action="store_true")
    a = ap.parse_args()
    cfg = Config(faceit=not a.no_faceit)
    out = run(a.data, pd.read_parquet(a.targets), cfg)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
