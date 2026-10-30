"""Dev walk-forward without the market: log loss of weekly-calibrated prices against HLTV outcomes, split by FACEIT coverage.

    python -m gnomepy_research.sessions.cs2_comp.e10.evaluate --data <bundle> --out <dir> --name <label> [--config JSON] [--no-faceit]

Dev weeks are Monday cutoffs before the tune period; every bo3 series in them is a target.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e10.predict import build_config, calibrate, logit
from gnomepy_research.sessions.cs2_comp.e10.run import TRAIN_START, Pass, monday_range

DEV_FIRST, DEV_LAST = pd.Timestamp("2025-10-06"), pd.Timestamp("2026-03-23")


def ll(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summarize(feats: pd.DataFrame, specs=("", "f", "fg", "f+new")) -> dict:
    weeks = [w for w in monday_range(DEV_FIRST, DEV_LAST)]
    dev = feats[(feats.week_cutoff >= DEV_FIRST) & (feats.week_cutoff <= DEV_LAST) & feats.y.notna()].copy()
    cov = np.minimum(dev.nf_a, dev.nf_b)
    dev["cov"] = np.where(cov == 0, "0", np.where(cov <= 3, "1-3", "4-5"))
    out = {"n": len(dev), "raw": float(ll(dev.p_series, dev.y).mean())}
    for spec in specs:
        q = calibrate(feats, weeks, spec).reindex(dev.match_id).to_numpy()
        out[f"cal[{spec}]"] = float(ll(q, dev.y.to_numpy()).mean())
        if spec == "":
            for c, g in dev.assign(l=ll(q, dev.y.to_numpy())).groupby("cov"):
                out[f"cal cov={c}"] = (int(len(g)), float(g.l.mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="run")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--no-faceit", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cfg = build_config(not a.no_faceit, json.loads(a.config))
    p = Pass(a.data, DEV_LAST + pd.Timedelta(days=7), cfg)
    t1 = time.time()
    feats = p.run(monday_range(TRAIN_START, DEV_LAST), None)
    t2 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(a.out / f"{a.name}.parquet", index=False)
    res = summarize(feats)
    res.update({"name": a.name, "config": a.config, "no_faceit": a.no_faceit, "events": {k: v for k, v in p.info.items() if k != "rating_fit"},
                "rating_fit": p.info.get("rating_fit"), "load_s": round(t1 - t0), "pass_s": round(t2 - t1)})
    (a.out / f"{a.name}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
