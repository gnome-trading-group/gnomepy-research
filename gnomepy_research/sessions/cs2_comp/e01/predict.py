"""Competition entry point: predictions from data before each target's week cutoff.

    python -m gnomepy_research.sessions.cs2_comp.e01.predict --data <bundle> --targets <parquet> --out <parquet> [--no-faceit]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e01.model import Config, build_maps, fit, naive, predict


def run(data: str, targets: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t = targets.assign(cut=targets.week_cutoff.map(naive))
    parts = []
    for cut, grp in t.groupby("cut"):
        fitted = fit(build_maps(data, cut, cfg), cut, cfg)
        parts.append(pd.DataFrame({"match_id": grp.match_id.to_numpy(), "q": predict(fitted, grp),
                                   "fit_cutoff": cut}))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", action="store_true")
    ap.add_argument("--faceit-weight", type=float)
    ap.add_argument("--half-life", type=float)
    ap.add_argument("--C", type=float)
    a = ap.parse_args()
    cfg = Config(faceit=not a.no_faceit)
    overrides = {k: v for k, v in (("faceit_weight", a.faceit_weight), ("half_life_days", a.half_life), ("C", a.C))
                 if v is not None}
    cfg = replace(cfg, **overrides)
    out = run(a.data, pd.read_parquet(a.targets), cfg)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
