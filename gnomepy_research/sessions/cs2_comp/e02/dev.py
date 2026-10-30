"""Walk-forward diagnostics on Jan-Mar 2026 series, scored on outcomes only (no market).

    python -m gnomepy_research.sessions.cs2_comp.e02.dev --data <bundle> --out <dir>

Features are cut at 2026-03-30 so the Apr-Jul tune targets never enter a fit or a feature. Coverage buckets use the
fewer of the two lineups' FACEIT-linked players with FACEIT history (0, 1-3, 4-5), as the harness does.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e02.features import build_queries
from gnomepy_research.sessions.cs2_comp.e02.predict import Config, fit, predict_week

DEV_CUT = pd.Timestamp("2026-03-30")
DEV_WEEKS = pd.date_range("2026-01-05", "2026-03-23", freq="7D")
BASE = Config()
VARIANTS = {
    "faceit_gbm": (True, BASE),
    "hltv_gbm": (False, BASE),
    "faceit_lr": (True, replace(BASE, models=("lr",))),
    "faceit_ens": (True, replace(BASE, models=("gbm", "lr"))),
    "faceit_gbm_deep": (True, replace(BASE, gbm={**BASE.gbm, "max_depth": 4, "n_estimators": 400, "min_child_weight": 80})),
    "faceit_gbm_shallow": (True, replace(BASE, gbm={**BASE.gbm, "max_depth": 2, "n_estimators": 600, "min_child_weight": 20})),
    "faceit_gbm_365d": (True, replace(BASE, window_days=365)),
    "hltv_gbm_365d": (False, replace(BASE, window_days=365)),
}


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def coverage(fcov_a: np.ndarray, fcov_b: np.ndarray) -> np.ndarray:
    low = np.minimum(fcov_a, fcov_b)
    return np.where(low == 0, "0", np.where(low <= 3, "1-3", "4-5"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    queries = build_queries(args.data, DEV_CUT, None, faceit=True)
    cov = coverage(queries.ab.a_fcov.to_numpy(), queries.ab.b_fcov.to_numpy())
    rows = []
    for name, (faceit, config) in VARIANTS.items():
        for week in DEV_WEEKS:
            mask = ((queries.meta.week == week) & ~queries.meta.is_target & (queries.meta.bo == 3)).to_numpy()
            if not mask.any():
                continue
            models = fit(queries, week, faceit, config)
            p = predict_week(models, queries, mask, faceit)
            y = queries.meta.y.to_numpy()[mask]
            rows.append(pd.DataFrame({"variant": name, "week": week, "match_id": queries.meta.match_id.to_numpy()[mask],
                                      "p": p, "y": y, "ll": logloss(p, y), "cov": cov[mask]}))
    d = pd.concat(rows, ignore_index=True)
    summary = {}
    for name, g in d.groupby("variant"):
        summary[name] = {"n": int(len(g)), "logloss": float(g.ll.mean()),
                         "by_cov": {c: [int(len(h)), float(h.ll.mean())] for c, h in g.groupby("cov")},
                         "by_week": [round(float(x), 4) for x in g.groupby("week").ll.mean()]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / "dev_predictions.parquet", index=False)
    (out / "dev_summary.json").write_text(json.dumps(summary, indent=1))
    for name, s in summary.items():
        print(f"{name}: n={s['n']} logloss={s['logloss']:.5f} by_cov={s['by_cov']}")


if __name__ == "__main__":
    main()
