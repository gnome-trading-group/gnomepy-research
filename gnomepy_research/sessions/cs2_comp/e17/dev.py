"""Walk-forward check on HLTV outcomes before the tune period, with no market: the honest place to pick settings.

    python -m gnomepy_research.sessions.cs2_comp.e17.dev --data <bundle dir> --start 2025-10-01 --end 2026-04-01
        [--no-faceit] [--faceit-weight W ...] [--name NAME] [--out-dir DIR]

Every bo3 HLTV series in [start, end) is priced at its week's Monday, exactly as the predict module prices a target,
and scored by log loss against its outcome, overall and by how many of each five have FACEIT history. Both
calibration designs, with and without the extra base features, are scored from the same filter snapshot. With --out-dir (default: $OUTPUT_DIR on Batch) the
report, the snapshot and the coverage-calibrated predictions are written there under NAME; an existing snapshot is
reused, so calibration variants cost seconds.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e17.data import hltv_maps, hltv_series, monday
from gnomepy_research.sessions.cs2_comp.e17.predict import calibrate, snapshot
from gnomepy_research.sessions.cs2_comp.e17.skill import SourceConfig


def ll(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def dev_targets(data_dir: str, start: str, end: str) -> pd.DataFrame:
    s = hltv_series(hltv_maps(data_dir, pd.Timestamp(end)))
    s = s[(s.bo_type == 3) & (s.time >= pd.Timestamp(start))]
    return pd.DataFrame({"match_id": s.match_id, "match_date": s.time.dt.normalize(), "team_a_id": s.team_a, "team_b_id": s.team_b,
                         "week_cutoff": monday(s.time), "y": s.y}).reset_index(drop=True)


def report(pred: pd.DataFrame, y: np.ndarray, label: str) -> list[str]:
    pred = pred.assign(y=y, l_q=ll(pred.q, y), l_raw=ll(pred.q_u if "q_u" in pred else pred.q_h, y), l_h=ll(pred.q_h, y))
    lines = [f"{label}: n={len(pred)} calibrated {pred.l_q.mean():.4f}  raw {pred.l_raw.mean():.4f}  raw hltv-only {pred.l_h.mean():.4f}"]
    if "cov_a_u" in pred:
        cov = np.minimum(pred.cov_a_u, pred.cov_b_u)
        seg = np.where(cov == 0, "0", np.where(cov <= 3, "1-3", "4-5"))
        for s in ("0", "1-3", "4-5"):
            g = pred[seg == s]
            lines.append(f"   faceit_cov={s:4s} n={len(g):5d} calibrated {g.l_q.mean():.4f}  raw unified {g.l_raw.mean():.4f}  raw hltv-only {g.l_h.mean():.4f}")
    known = pred.known_a_h & pred.known_b_h
    g = pred[~known]
    lines.append(f"   no HLTV lineup for a side: n={len(g)} calibrated {g.l_q.mean():.4f}  raw hltv-only {g.l_h.mean():.4f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-04-01")
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--faceit-weight", type=float, default=SourceConfig.faceit_weight)
    ap.add_argument("--faceit-prior-offset", type=float, default=SourceConfig.faceit_prior_offset)
    ap.add_argument("--faceit-sigma0", type=float, default=SourceConfig.faceit_sigma0)
    ap.add_argument("--calibration-months", type=int, nargs="+", default=[12])
    ap.add_argument("--name", default="dev")
    ap.add_argument("--out-dir", default=os.environ.get("OUTPUT_DIR"))
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else None
    targets = dev_targets(a.data, a.start, a.end)
    src = SourceConfig(a.faceit_weight, a.faceit_prior_offset, a.faceit_sigma0)
    snap_path = out_dir / f"{a.name}_snapshot.parquet" if out_dir else None
    if snap_path and snap_path.exists():
        snap = pd.read_parquet(snap_path)
    else:
        snap = snapshot(a.data, targets, a.faceit, src)
        if snap_path:
            out_dir.mkdir(parents=True, exist_ok=True)
            snap.to_parquet(snap_path, index=False)
    lines = []
    for months in a.calibration_months:
        label = f"faceit={a.faceit} w={a.faceit_weight} a={a.faceit_prior_offset} s0={a.faceit_sigma0} months={months}"
        for extra in (False, True):
            for calibration in ("pooled", "coverage") if a.faceit else ("pooled",):
                pred = calibrate(snap, a.faceit, months, calibration, extra, log=lambda *_: None)
                pred = targets[["match_id", "y"]].merge(pred, on="match_id", how="left")
                lines += report(pred, pred.y.to_numpy(float), f"{label} calibration={calibration} extra={extra}")
                if out_dir and extra and months == a.calibration_months[0] and calibration == ("coverage" if a.faceit else "pooled"):
                    pred.to_parquet(out_dir / f"{a.name}.parquet", index=False)
    print("\n".join(lines))
    if out_dir:
        (out_dir / f"{a.name}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
