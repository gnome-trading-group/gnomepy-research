"""Predictions for the competition targets.

    python -m gnomepy_research.sessions.cs2_comp.e10.predict --data <bundle> --targets <targets.parquet> --out <parquet> [--no-faceit]

One causal pass of the skill filter prices every historical bo3 series at its week's Monday, exactly as it prices the
targets. A weekly logistic calibration, fitted on the series that finished before the target week, turns the
filter's price (plus coverage features) into q. Nothing of a later week reaches an earlier one.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e10.run import TRAIN_START, Pass, RunConfig, monday_range

CALIBRATION_START = pd.Timestamp("2024-07-01")
L2 = 5.0
DEFAULT_SPEC = "s+new+c"


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(f: pd.DataFrame, spec: str) -> np.ndarray:
    """Columns antisymmetric under swapping the teams, so q(A, B) = 1 - q(B, A) with no intercept."""
    z = logit(f.p_series)
    cols = [z]
    tokens = set(spec.split("+"))
    if "f" in tokens:
        cols.append(3.0 * (f.fres_lo + f.fres_hi).to_numpy())
    if "s" in tokens:
        cols.append(3.0 * f.fres_lo.to_numpy())
        cols.append(3.0 * f.fres_hi.to_numpy())
    if "c" in tokens:
        cols.append(np.log1p(f.coh_a.to_numpy()) - np.log1p(f.coh_b.to_numpy()))
    if "a" in tokens:
        cols.append(np.log1p(f.act_a.to_numpy()) - np.log1p(f.act_b.to_numpy()))
    if "elo" in tokens:
        cols.append(f.elo.to_numpy())
    if "lan" in tokens:
        cols.append(z * f.lan.fillna(0).to_numpy())
    if "tier" in tokens:
        cols.append(z * f.tier.fillna(0).to_numpy())
    if "sd" in tokens:
        cols.append(f.d.to_numpy() * f.sd.to_numpy())
    if "new" in tokens:
        cols.append(z * (f.nnew_a + f.nnew_b).to_numpy() / 10.0)
    return np.column_stack(cols)


def fit_logistic(x: np.ndarray, y: np.ndarray, prior: np.ndarray) -> np.ndarray:
    w = prior.copy()
    for _ in range(50):
        p = 1 / (1 + np.exp(-x @ w))
        g = x.T @ (y - p) - L2 * (w - prior)
        h = x.T @ (x * (p * (1 - p))[:, None]) + L2 * np.eye(len(w))
        step = np.linalg.solve(h, g)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w


def calibrate(feats: pd.DataFrame, weeks: list[pd.Timestamp], spec: str = "") -> pd.Series:
    """q for each row of the given weeks, from a logistic fitted on the labelled rows of earlier weeks."""
    out = {}
    x_all = design(feats, spec)
    prior = np.zeros(x_all.shape[1])
    prior[0] = 1.0
    labelled = feats.y.notna().to_numpy()
    for w in weeks:
        train = labelled & (feats.week_cutoff < w).to_numpy() & (feats.week_cutoff >= CALIBRATION_START).to_numpy()
        coef = fit_logistic(x_all[train], feats.y.to_numpy()[train], prior) if train.sum() >= 200 else prior
        sel = (feats.week_cutoff == w).to_numpy()
        z = x_all[sel] @ coef
        out.update(dict(zip(feats.match_id[sel], 1 / (1 + np.exp(-z)))))
    return pd.Series(out)


def build_config(use_faceit: bool, overrides: dict) -> RunConfig:
    """Overrides: h_<field> and f_<field> set the HLTV and FACEIT skill filters, any other key a RunConfig field."""
    base = RunConfig()
    hltv = replace(base.hltv, **{k[2:]: v for k, v in overrides.items() if k.startswith("h_")})
    faceit = replace(base.faceit_skill, **{k[2:]: v for k, v in overrides.items() if k.startswith("f_")})
    run = {k: v for k, v in overrides.items() if k in RunConfig.__dataclass_fields__ and k not in ("hltv", "faceit_skill")}
    run["faceit_rating"] = use_faceit and run.get("faceit_rating", True)
    return replace(base, hltv=hltv, faceit_skill=faceit, use_faceit=use_faceit, **run)


def predict(data_dir: Path, targets: pd.DataFrame, use_faceit: bool, overrides: dict | None = None, spec: str = DEFAULT_SPEC) -> pd.DataFrame:
    cfg = build_config(use_faceit, overrides or {})
    weeks = sorted(targets.week_cutoff.unique())
    p = Pass(data_dir, pd.Timestamp(weeks[-1]), cfg)
    feats = p.run(monday_range(TRAIN_START, pd.Timestamp(weeks[-1])), targets)
    q = calibrate(feats, [pd.Timestamp(w) for w in weeks], spec)
    out = targets[["match_id", "week_cutoff"]].rename(columns={"week_cutoff": "fit_cutoff"}).copy()
    out["q"] = out.match_id.map(q).to_numpy()
    return out[["match_id", "q", "fit_cutoff"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("OUTPUT_DIR", ".")) / "predictions.parquet")
    ap.add_argument("--no-faceit", action="store_true")
    ap.add_argument("--config", default="{}", help="JSON overrides of RunConfig / SkillConfig fields")
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    out = predict(a.data, targets, not a.no_faceit, json.loads(a.config), a.spec)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
