"""Walk-forward diagnostics on Jan-Mar 2026 series, scored on outcomes only (no market).

    python -m gnomepy_research.sessions.cs2_comp.e02.dev --data <bundle> --out <dir>

Features are cut at 2026-03-30 so the Apr-Jul tune targets never enter a fit or a feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e02.features import build_queries, feature_columns
from gnomepy_research.sessions.cs2_comp.e02.predict import fit, predict_week

DEV_CUT = pd.Timestamp("2026-03-30")
DEV_WEEKS = pd.date_range("2026-01-05", "2026-03-23", freq="7D")


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    queries = build_queries(args.data, DEV_CUT, None, faceit=True)
    rows = []
    for faceit in (True, False):
        for week in DEV_WEEKS:
            model = fit(queries, week, faceit)
            mask = ((queries.meta.week == week) & ~queries.meta.is_target & (queries.meta.bo == 3)).to_numpy()
            if not mask.any():
                continue
            p = predict_week(model, queries, mask, faceit)
            y = queries.meta.y.to_numpy()[mask]
            rows.append(pd.DataFrame({"variant": "faceit" if faceit else "hltv_only", "week": week,
                                      "match_id": queries.meta.match_id.to_numpy()[mask], "p": p, "y": y,
                                      "ll": logloss(p, y)}))
    d = pd.concat(rows, ignore_index=True)
    summary = {v: {"n": int(len(g)), "logloss": float(g.ll.mean()),
                   "by_week": g.groupby("week").ll.mean().round(4).astype(float).tolist()}
               for v, g in d.groupby("variant")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / "dev_predictions.parquet", index=False)
    (out / "dev_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    for v, s in summary.items():
        print(f"{v}: n={s['n']} logloss={s['logloss']:.5f}")
    print("feature columns faceit:", feature_columns(True))


if __name__ == "__main__":
    main()
