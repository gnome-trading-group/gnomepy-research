"""
Walk-forward diagnostic on weeks before the tune period: fit at each Monday, price that week's HLTV bo3 series
from the teams' latest fives, and score against the HLTV outcome (no market involved).

    python -m gnomepy_research.sessions.cs2_comp.e20.diag --data <bundle> --first 2026-01-05 --last 2026-03-23 [--no-faceit] [config overrides]

On Batch (OUTPUT_DIR set) the result goes to $OUTPUT_DIR/<name>.json with the per-series prices beside it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e20.predict import add_config_args, config_from, run

# Share of the less-covered lineup with a FACEIT link: none, 1-3 of five, 4-5 of five.
COVERAGE = ("0", "1-3", "4-5")


def week_series(data_dir: str, first: str, last: str) -> pd.DataFrame:
    end = pd.Timestamp(last) + pd.Timedelta(days=7)
    h = load(data_dir, "hltv/cs2_match_history", end,
             ["match_id", "match_date", "match_time", "event_name", "format", "team_a_id", "team_b_id", "team_a_won", "bo_type",
              "team_a_rank", "team_b_rank", "is_lan", "team_a_player_ids", "team_b_player_ids"])
    h = h[~showmatch_mask(h) & (h.bo_type == 3)]
    h["t"] = pd.to_datetime(h.match_time).dt.tz_convert(None).fillna(pd.to_datetime(h.match_date))
    h = h[h.t >= pd.Timestamp(first)]
    # Reporting only: how many of the five that played have a FACEIT link (never a model input).
    linked = set(load(data_dir, "faceit/hltv_faceit_map").hltv_id.astype(int))
    for side in ("a", "b"):
        h[f"linked_{side}"] = [sum(int(p) in linked for p in ids) if isinstance(ids, (list, np.ndarray)) else 0
                               for ids in h[f"team_{side}_player_ids"]]
    s = h.groupby("match_id").agg(t=("t", "first"), team_a_id=("team_a_id", "first"), team_b_id=("team_b_id", "first"),
                                  wins=("team_a_won", "sum"), maps=("team_a_won", "size"), rank_a=("team_a_rank", "first"),
                                  rank_b=("team_b_rank", "first"), is_lan=("is_lan", "first"),
                                  linked_a=("linked_a", "first"), linked_b=("linked_b", "first")).reset_index()
    s["y"] = (s.wins * 2 > s.maps).astype(int)
    s["week_cutoff"] = s.t.dt.normalize() - pd.to_timedelta(s.t.dt.weekday, unit="D")
    best = s[["rank_a", "rank_b"]].min(axis=1).fillna(999)
    s["thin"] = (s.is_lan == 0) & (best > 100)
    s["coverage"] = pd.cut(s[["linked_a", "linked_b"]].min(axis=1), [-1, 0, 3, 5], labels=COVERAGE).astype(str)
    return s[(s.week_cutoff >= pd.Timestamp(first)) & (s.week_cutoff <= pd.Timestamp(last))]


def ll(q, y):
    q = np.clip(q, 0.01, 0.99)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def slope(q, y):
    x = np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    k = 1.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-k * x))
        k += (x @ (y - p)) / ((x * x) @ (p * (1 - p)))
    return float(k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--first", default="2026-01-05")
    ap.add_argument("--last", default="2026-03-23")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--out")
    ap.add_argument("--name", default="diag")
    add_config_args(ap)
    a = ap.parse_args()
    s = week_series(a.data, a.first, a.last)
    pred = run(a.data, s, config_from(a), a.faceit)
    d = s.merge(pred, on="match_id")
    y = d.y.to_numpy(float)
    res = {"name": a.name, "faceit": a.faceit, "config": dataclasses.asdict(config_from(a)), "n": len(d),
           "ll": float(ll(d.q, y).mean()), "slope": slope(d.q.to_numpy(), y),
           "ll_thin": float(ll(d.q[d.thin], y[d.thin]).mean()), "n_thin": int(d.thin.sum()),
           "ll_rest": float(ll(d.q[~d.thin], y[~d.thin]).mean())}
    for c in COVERAGE:
        m = (d.coverage == c).to_numpy()
        res[f"ll_cov_{c}"] = float(ll(d.q[m], y[m]).mean()) if m.any() else None
        res[f"n_cov_{c}"] = int(m.sum())
    print(json.dumps(res))
    out_dir = os.environ.get("OUTPUT_DIR")
    if out_dir:
        Path(out_dir, f"{a.name}.json").write_text(json.dumps(res))
        d.to_parquet(Path(out_dir, f"{a.name}.parquet"), index=False)
    if a.out:
        d.to_parquet(a.out, index=False)


if __name__ == "__main__":
    sys.exit(main())
