"""
Walk-forward diagnostics before the tune period, against HLTV outcomes only (no market).

    python -m gnomepy_research.sessions.cs2_comp.e15.dev --data <bundle> --start 2026-01-05 --end 2026-03-30 \
        --configs '[{}, {"credibility_faceit": 0.5}]' [--every 2] [--out <parquet>]

Targets are the bo3 HLTV series played in [start, end), priced from their week's Monday cutoff exactly as predict.py
prices the competition targets. Each config is run with FACEIT and without; --every k fits every k-th week only and
prices the following k weeks from it (a cheaper, slightly staler approximation for comparing configs).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e15.data import load_raw
from gnomepy_research.sessions.cs2_comp.e15.model import Config
from gnomepy_research.sessions.cs2_comp.e15.predict import career_first, predict_week


def dev_targets(data: Path, start: str, end: str) -> pd.DataFrame:
    h = load(data, "hltv/cs2_match_history")
    h = h[~showmatch_mask(h) & (h.bo_type == 3)]
    h["t"] = pd.to_datetime(h.match_time.fillna(h.match_date))
    h["t"] = h.t.dt.tz_convert(None) if h.t.dt.tz is not None else h.t
    s = h.groupby("match_id").agg(t=("t", "first"), team_a_id=("team_a_id", "first"), team_b_id=("team_b_id", "first"),
                                  wins=("team_a_won", "sum"), maps=("team_a_won", "size"), is_lan=("is_lan", "first"),
                                  tier=("event_tier", "first")).reset_index()
    s = s[(s.t >= start) & (s.t < end) & (s.maps >= 2)]
    s["y"] = (s.wins * 2 > s.maps).astype(float)
    s["week_cutoff"] = s.t.dt.normalize() - pd.to_timedelta(s.t.dt.weekday, unit="D")
    return s.reset_index(drop=True)


def ll(q, y):
    q = np.clip(q, 0.01, 0.99)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def slope_ll(q, y):
    """Log loss after the best single logit slope: discrimination with calibration scale factored out."""
    x = np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    a = 1.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-a * x))
        a += np.sum(x * (y - p)) / max(np.sum(x * x * p * (1 - p)), 1e-9)
    return ll(1 / (1 + np.exp(-a * x)), y).mean(), a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--start", default="2026-01-05")
    ap.add_argument("--end", default="2026-03-30")
    ap.add_argument("--configs", default="[{}]")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--sources", default="faceit,hltv")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    targets = dev_targets(a.data, a.start, a.end)
    weeks = sorted(targets.week_cutoff.unique())
    fit_week = {w: weeks[(i // a.every) * a.every] for i, w in enumerate(weeks)}
    targets["fit_week"] = targets.week_cutoff.map(fit_week)
    rows, preds = [], []
    for source in a.sources.split(","):
        raw = load_raw(a.data, source == "faceit")
        first = career_first(raw)
        for ci, overrides in enumerate(json.loads(a.configs)):
            cfg = dataclasses.replace(Config(), **overrides)
            p = pd.concat([predict_week(raw, first, g, pd.Timestamp(w), cfg) for w, g in targets.groupby("fit_week")])
            d = targets.merge(p, on="match_id")
            sl, slope = slope_ll(d.q.to_numpy(), d.y.to_numpy())
            online = d.is_lan == 0
            rows.append({"source": source, "config": ci, **overrides, "n": len(d), "ll": ll(d.q, d.y).mean(),
                         "ll_slope": sl, "slope": slope, "ll_online": ll(d.q[online], d.y[online]).mean(),
                         "ll_lan": ll(d.q[~online], d.y[~online]).mean()})
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}), flush=True)
            preds.append(d.assign(source=source, config=ci))
    if a.out:
        pd.concat(preds).to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
