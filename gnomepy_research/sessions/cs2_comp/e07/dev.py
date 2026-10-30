"""
Development scoring on series before the tune period, against HLTV outcomes only (no market).

Builds the event streams once and runs each named configuration over the weekly walk-forward, pricing every HLTV bo3
series of the week from the state at that Monday 00:00.

    python -m gnomepy_research.sessions.cs2_comp.e07.dev --data <bundle> --out <dir> --sets-file sets.json \
        [--first 2025-10-06] [--before 2026-04-01]

sets.json is a list of {"name": ..., "set": {"key": value, ...}}.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.e07 import analysis, data
from gnomepy_research.sessions.cs2_comp.e07.filter import FilterConfig
from gnomepy_research.sessions.cs2_comp.e07.final_config import FINAL
from gnomepy_research.sessions.cs2_comp.e07.lineup import RosterMoves
from gnomepy_research.sessions.cs2_comp.e07.model import Model, ModelConfig

Q_FLOOR = 0.02


def _ll(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.clip(q, Q_FLOOR, 1 - Q_FLOOR)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def walk(streams: data.Streams, rankings: pd.DataFrame, moves: RosterMoves, practice: dict, cfg: ModelConfig, first: str, before: str) -> pd.DataFrame:
    model = Model(cfg, streams, data.RankPoints(rankings), moves, practice)
    series = streams.series[streams.series.bo_type == 3]
    rows = []
    for cutoff in pd.date_range(first, before, freq="7D", inclusive="left"):
        c = cutoff.to_datetime64().astype("datetime64[ns]").astype(np.int64) / data.NS_PER_DAY
        model.advance(c)
        wk = series[(series.t >= c) & (series.t < c + 7)]
        for r in wk.itertuples():
            f = model.features(int(r.team_a_id), int(r.team_b_id), float(r.t), c)
            ia = np.array([streams.index[int(k)] for k in r.a_players])
            ib = np.array([streams.index[int(k)] for k in r.b_players])
            if np.isfinite(model.h.seen[ia]).all() and np.isfinite(model.h.seen[ib]).all():
                f["q_oracle"] = model.h.price(int(r.team_a_id), int(r.team_b_id), float(r.t), c, ia, ib)[0]
            known_a, known_b = model.h.last_five.get(int(r.team_a_id)), model.h.last_five.get(int(r.team_b_id))
            f["same_a"] = np.nan if known_a is None else len(set(known_a[0].tolist()) & set(ia.tolist()))
            f["same_b"] = np.nan if known_b is None else len(set(known_b[0].tolist()) & set(ib.tolist()))
            rows.append({"match_id": r.match_id, "y": r.y, "cutoff": cutoff, "t": r.t, "is_lan": r.is_lan, **f})
    return pd.DataFrame(rows)


def summarize(p: pd.DataFrame) -> dict:
    ll = _ll(p.q_h.to_numpy(), p.y.to_numpy())
    return {"n": len(p), "ll_h": float(ll.mean()), "acc_h": float(((p.q_h > 0.5) == (p.y == 1)).mean())}


def build_config(overrides: dict) -> ModelConfig:
    """Keys prefixed h_ or f_ go to the HLTV or FACEIT filter; use_faceit goes to the model."""
    h = {k[2:]: v for k, v in overrides.items() if k.startswith("h_")}
    f = {k[2:]: v for k, v in overrides.items() if k.startswith("f_")}
    base = FINAL
    model = {k: v for k, v in overrides.items() if k in ("use_faceit", "roster_moves", "faceit_lineup_weight", "faceit_lineup_days", "newcomer_prob")}
    return replace(base, hltv=replace(base.hltv, **h), faceit=replace(base.faceit, **f), **model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "dev_out")))
    parser.add_argument("--sets-file", type=Path)
    parser.add_argument("--sets-json")
    parser.add_argument("--first", default="2025-10-06")
    parser.add_argument("--before", default="2026-04-01")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sets = json.loads(args.sets_json if args.sets_json else args.sets_file.read_text())
    before = pd.Timestamp(args.before)
    rankings = bundle.load(args.data, "hltv/cs2_team_rankings", before)
    moves = RosterMoves(bundle.load(args.data, "hltv/cs2_roster_moves", before))
    built: dict[bool, data.Streams] = {}
    for item in sets:
        cfg = build_config(item["set"])
        t0 = time.time()
        if cfg.use_faceit not in built:
            built[cfg.use_faceit] = data.build(args.data, before, cfg.use_faceit)
        t1 = time.time()
        practice = data.practice_times(args.data, before, built[cfg.use_faceit].index) if cfg.use_faceit else {}
        p = walk(built[cfg.use_faceit], rankings, moves, practice, cfg, args.first, args.before)
        p.to_parquet(args.out / f"{item['name']}.parquet", index=False)
        res = {"name": item["name"], "set": item["set"], **summarize(p), "report": analysis.report(p), "build_s": round(t1 - t0), "walk_s": round(time.time() - t1)}
        (args.out / f"{item['name']}.json").write_text(json.dumps(res))
        print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
