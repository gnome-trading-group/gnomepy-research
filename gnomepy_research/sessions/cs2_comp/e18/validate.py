"""Walk-forward diagnostics before the tune period: weekly fits, scored on HLTV bo3 outcomes (no market).

    python -m gnomepy_research.sessions.cs2_comp.e18.validate --data <bundle dir> --first 2026-01-05 --last 2026-03-23 \
        --configs '[{"name": "base"}, {"name": "hltv", "faceit": false}]' --out <dir>

For each Monday cutoff in [first, last] and each config: fit on data before the cutoff, price every HLTV bo3 series
of that week (teams and lineups as predict.py sees them: latest pre-cutoff five), and compare with the outcome.
Prints the mean log loss per config, overall and by how many of the inferred lineup's players have FACEIT maps.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e18.data import build
from gnomepy_research.sessions.cs2_comp.e18.model import Config, fit_model, price, team_key


def week_series(data_dir, cutoff: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history")
    h = h[~showmatch_mask(h) & (h.bo_type == 3)]
    t = pd.to_datetime(h.match_time).dt.tz_convert(None).fillna(pd.to_datetime(h.match_date))
    h = h[(t >= cutoff) & (t < cutoff + pd.Timedelta(days=7))]
    won = h.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() * 2 > len(w)))
    s = h.drop_duplicates("match_id")[["match_id", "team_a_id", "team_b_id", "is_lan", "team_a_rank", "team_b_rank"]].copy()
    s["y"] = s.match_id.map(won).to_numpy()
    return s.reset_index(drop=True)


def log_loss(q, y) -> np.ndarray:
    q = np.clip(np.asarray(q, float), 0.01, 0.99)
    y = np.asarray(y, float)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--configs", required=True, help="JSON list of Config overrides, each with a name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_dir = Path(os.environ["OUTPUT_DIR"]) / a.out if "OUTPUT_DIR" in os.environ and not Path(a.out).is_absolute() else Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = json.loads(a.configs)
    rows = []
    for cutoff in pd.date_range(a.first, a.last, freq="7D"):
        series = week_series(a.data, cutoff)
        cache: dict[tuple, object] = {}
        for c in configs:
            cfg = Config(**{k: v for k, v in c.items() if k != "name"})
            key = (cfg.faceit, cfg.window_days, cfg.roster_moves)
            if key not in cache:
                cache[key] = build(a.data, cutoff, cfg.faceit, cfg.window_days, cfg.roster_moves)
            obs = cache[key]
            fit = fit_model(obs, cutoff, cfg)
            faceit_players = {p for five in obs.maps.lineup_a[obs.maps.source == "faceit"] for p in five} \
                | {p for five in obs.maps.lineup_b[obs.maps.source == "faceit"] for p in five}
            for r in series.itertuples():
                ka, kb = team_key(r.team_a_id), team_key(r.team_b_id)
                la, lb = obs.latest_lineup.get(ka, []), obs.latest_lineup.get(kb, [])
                q = price(fit, ka, la, kb, lb, cutoff)
                cov = min(sum(p in faceit_players for p in la), sum(p in faceit_players for p in lb))
                rows.append({"config": c["name"], "cutoff": cutoff, "match_id": r.match_id, "q": q, "y": r.y,
                             "has_lineups": bool(la and lb), "faceit_cov": cov, "is_lan": r.is_lan})
            print(f"{cutoff:%Y-%m-%d} {c['name']:12s} series {len(series)} {json.dumps(fit.info)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(out_dir / "walk_forward.parquet", index=False)
    summary = summarize(d)
    print(summary.round(4).to_string())
    (out_dir / "summary.json").write_text(summary.to_json(orient="index"))


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["ll"] = log_loss(d.q, d.y)
    d["cov_bin"] = np.where(d.faceit_cov == 0, "cov0", np.where(d.faceit_cov <= 3, "cov1-3", "cov4-5"))
    summary = d.groupby("config").ll.agg(["mean", "count"])
    for name, g in d.groupby("cov_bin"):
        summary[str(name)] = g.groupby("config").ll.mean()
    summary["no_lineup"] = d[~d.has_lineups].groupby("config").ll.mean()
    return summary


if __name__ == "__main__":
    main()
