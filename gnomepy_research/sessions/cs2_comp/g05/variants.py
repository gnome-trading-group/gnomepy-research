"""Skill-fit feature rows for one configuration over a range of weekly cutoffs, for comparing configurations.

    python -m gnomepy_research.sessions.cs2_comp.g05.variants --data <bundle> --name <variant> --mode dev|pre|both
        [--first 2025-09-29 --last 2026-03-23] [--set half_life_days=150 --set faceit_weight=1.0 ...] [--out <dir>]

`dev` builds rows for the target series of the bundle (priced with their announced fives); `pre` builds them for the
HLTV bo3 series of each week from --first to --last, with their outcome, which is what a layer is fitted on before
the first target week. Rows land in --out (default $OUTPUT_DIR) as <name>_<mode>_hltv.parquet and
<name>_<mode>_faceit.parquet.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g05.data import Inputs
from gnomepy_research.sessions.cs2_comp.g05.features import feature_rows
from gnomepy_research.sessions.cs2_comp.g05.model import Config

BASE = Config(sigma=0.5, half_life_days=300.0, rating_gamma=0.6, rating_sd=0.31)


def config(sets: list[str]) -> Config:
    kw = {}
    for s in sets:
        k, v = s.split("=")
        kw[k] = type(getattr(BASE, k))(v)
    return dataclasses.replace(BASE, **kw)


def prepare_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """Target series in the shape feature_rows takes: naive week_cutoff, announced fives and stand-in counts."""
    wc = pd.to_datetime(targets.week_cutoff)
    t = targets.rename(columns={"team_a_id": "team_a", "team_b_id": "team_b"})[["match_id", "team_a", "team_b"]].copy()
    t["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    for side in ("a", "b"):
        t[f"lineup_{side}"] = targets[f"team_{side}_lineup_ids"].to_numpy()
        t[f"sub_{side}"] = [0 if v is None else len(v) for v in targets[f"team_{side}_standin_ids"]]
    return t


def pre_targets(inputs: Inputs, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    s = inputs.series[inputs.series.bo_type == 3]
    return pd.concat([s[(s.time >= c) & (s.time < c + pd.Timedelta(days=7))].assign(week_cutoff=c) for c in cutoffs],
                     ignore_index=True)[["match_id", "team_a", "team_b", "week_cutoff", "y", "lineup_a", "lineup_b", "sub_a", "sub_b"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["dev", "pre", "both"], required=True)
    ap.add_argument("--first")
    ap.add_argument("--last")
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "."))
    a = ap.parse_args()
    # BLAS sees every core of a Batch host, not the task's vCPUs; oversubscribed threads made one fit take minutes.
    threadpool_limits(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    cfg = config(a.sets)
    print(a.name, cfg, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode in (("dev", "pre") if a.mode == "both" else (a.mode,)):
        if mode == "dev":
            targets = prepare_targets(pd.read_parquet(Path(a.data) / "targets_tune.parquet"))
            inputs = Inputs(a.data, targets.week_cutoff.max(), faceit=True)
        else:
            cutoffs = list(pd.date_range(a.first, a.last, freq="7D"))
            inputs = Inputs(a.data, cutoffs[-1] + pd.Timedelta(days=7), faceit=True)
            targets = pre_targets(inputs, cutoffs)
        for faceit in (False, True):
            rows = feature_rows(inputs, targets, cfg, faceit, verbose=True)
            rows.to_parquet(out / f"{a.name}_{mode}_{'faceit' if faceit else 'hltv'}.parquet", index=False)
        del inputs


if __name__ == "__main__":
    main()
