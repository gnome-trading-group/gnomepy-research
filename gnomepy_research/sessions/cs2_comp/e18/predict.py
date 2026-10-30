"""Price the target series: one fit per distinct week_cutoff, on bundle rows dated before it.

    python -m gnomepy_research.sessions.cs2_comp.e18.predict --data <bundle dir> --targets <targets parquet> --out <parquet>
        [--no-faceit] [--weeks 2026-04-06,2026-04-13] [--config '{"half_life_days": 90}']

Writes match_id, q (P(team A wins the series)) and fit_cutoff for every target. Lineups are the latest five each team
fielded before the cutoff; a team with no maps in the window is priced from its rank prior alone.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e18.data import build
from gnomepy_research.sessions.cs2_comp.e18.model import Config, Fit, fit_model, price, team_key


def predict_week(data_dir, cutoff: pd.Timestamp, targets: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, Fit]:
    obs = build(data_dir, cutoff, cfg.faceit, cfg.window_days)
    fit = fit_model(obs, cutoff, cfg)
    rows = []
    for r in targets.itertuples():
        a, b = team_key(r.team_a_id), team_key(r.team_b_id)
        q = price(fit, a, obs.latest_lineup.get(a, []), b, obs.latest_lineup.get(b, []), cutoff)
        rows.append((r.match_id, q, cutoff))
    return pd.DataFrame(rows, columns=["match_id", "q", "fit_cutoff"]), fit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--weeks", help="comma-separated week cutoffs to restrict to (ISO dates)")
    ap.add_argument("--config", help="JSON overrides for Config")
    a = ap.parse_args()
    cfg = replace(Config(**(json.loads(a.config) if a.config else {})), faceit=a.faceit)
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = pd.to_datetime(targets.week_cutoff)
    if a.weeks:
        keep = {pd.Timestamp(w) for w in a.weeks.split(",")}
        targets = targets[targets.week_cutoff.isin(keep)]
    parts = []
    for cutoff, week in targets.groupby("week_cutoff", sort=True):
        preds, fit = predict_week(a.data, pd.Timestamp(cutoff), week, cfg)
        parts.append(preds)
        print(f"{cutoff:%Y-%m-%d} targets {len(week)} {json.dumps(fit.info)}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out_path = Path(os.environ["OUTPUT_DIR"]) / a.out if "OUTPUT_DIR" in os.environ and not Path(a.out).is_absolute() else Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out)} predictions to {out_path}")


if __name__ == "__main__":
    main()
