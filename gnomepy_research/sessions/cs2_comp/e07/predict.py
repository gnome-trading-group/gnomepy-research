"""
Predict P(team A wins the bo3) for every target series with the player-skill filter, one pass over the data.

    python -m gnomepy_research.sessions.cs2_comp.e07.predict --data <bundle dir> --targets <targets parquet> \
        --out <parquet> [--no-faceit] [--cutoff YYYY-MM-DD] [--weeks YYYY-MM-DD ...]

The filter is causal, so its state when it reaches a week's cutoff is exactly what a fit on the data before that
cutoff would hold; one chronological pass therefore serves every week. Targets carry no announced lineup (those rows
are after the cutoff), so each team plays its last known five.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.e07 import data
from gnomepy_research.sessions.cs2_comp.e07 import stack
from gnomepy_research.sessions.cs2_comp.e07.final_config import FINAL, STACK_FACEIT, STACK_HLTV
from gnomepy_research.sessions.cs2_comp.e07.lineup import RosterMoves
from gnomepy_research.sessions.cs2_comp.e07.model import Model, ModelConfig

Q_FLOOR = 0.02


def run(data_dir: Path, targets: pd.DataFrame, cfg: ModelConfig, features_out: Path | None = None) -> pd.DataFrame:
    cutoffs = sorted(targets.week_cutoff.unique())
    last = pd.Timestamp(cutoffs[-1])
    streams = data.build(data_dir, last, cfg.use_faceit)
    weights = STACK_FACEIT if cfg.use_faceit else STACK_HLTV
    model = Model(cfg, streams, data.RankPoints(bundle.load(data_dir, "hltv/cs2_team_rankings", last)),
                  RosterMoves(bundle.load(data_dir, "hltv/cs2_roster_moves", last)),
                  data.practice_times(data_dir, last, streams.index) if any(n.startswith("practice") for n in weights) else None)
    out, tables = [], []
    for cutoff in cutoffs:
        cutoff_day = pd.Timestamp(cutoff).to_datetime64().astype("datetime64[ns]").astype(np.int64) / data.NS_PER_DAY
        model.advance(cutoff_day)
        wk = targets[targets.week_cutoff == cutoff]
        t_match = pd.to_datetime(wk.match_time)
        t_match = t_match.dt.tz_convert(None) if t_match.dt.tz is not None else t_match
        t_days = t_match.to_numpy("datetime64[ns]").astype(np.int64) / data.NS_PER_DAY
        feats = pd.DataFrame([model.features(int(r.team_a_id), int(r.team_b_id), float(td), cutoff_day)
                              for r, td in zip(wk.itertuples(), t_days)])
        feats["is_lan"] = wk.is_lan.to_numpy(float)
        q = np.clip(stack.apply(feats, weights), Q_FLOOR, 1 - Q_FLOOR)
        out.append(pd.DataFrame({"match_id": wk.match_id.to_numpy(), "q": q, "fit_cutoff": pd.Timestamp(cutoff)}))
        tables.append(feats.assign(match_id=wk.match_id.to_numpy(), fit_cutoff=pd.Timestamp(cutoff)))
    if features_out is not None:
        pd.concat(tables, ignore_index=True).to_parquet(features_out, index=False)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-faceit", action="store_true")
    parser.add_argument("--features-out", type=Path)
    parser.add_argument("--cutoff")
    parser.add_argument("--weeks", nargs="*")
    args = parser.parse_args()
    targets = pd.read_parquet(args.targets)
    if args.cutoff:
        targets = targets.assign(week_cutoff=pd.Timestamp(args.cutoff))
    if args.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(args.weeks))]
    cfg = replace(FINAL, use_faceit=not args.no_faceit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args.data, targets, cfg, args.features_out).to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
