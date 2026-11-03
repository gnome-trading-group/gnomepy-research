"""
g16 predictions: P(team A wins the bo3) for every target series.

    python -m gnomepy_research.sessions.cs2_comp.g16.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--rows <dir>]

Per distinct week_cutoff: the player-skill filter has read every HLTV map before the cutoff; each target is priced
with its announced five from that state. The market path up to kickoff is read as a noisy measurement (`market.py`),
and a small logistic layer, refitted at each cutoff on every market-joined series played before it, combines the two.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths
from gnomepy_research.sessions.cs2_comp.g16.data import Inputs, naive, target_rows
from gnomepy_research.sessions.cs2_comp.g16.layer import Spec, design, fit
from gnomepy_research.sessions.cs2_comp.g16.layer import predict as layer_predict
from gnomepy_research.sessions.cs2_comp.g16.market import path_features
from gnomepy_research.sessions.cs2_comp.g16.skill import SkillConfig, weekly_rows

SKILL = SkillConfig(faceit_weight=1.0)
SPEC = Spec(columns=("zm", "dz", "zs"), prior_mean={"zm": 1.0}, prior_precision=4.0)
SPEC_NO_MARKET = Spec(columns=("zs",), prior_mean={"zs": 1.0}, prior_precision=4.0)
# Rows of a market-joined series only train the layer once the market has had enough prints to mean anything.
MIN_ACTIVE_PRINTS = 5


def training_rows(inputs: Inputs, skill: SkillConfig = SKILL, verbose: bool = False) -> pd.DataFrame:
    """Every bo3 series in the history with a market path, priced at its own week's cutoff."""
    paths = load(inputs.data_dir, "market/series_paths", inputs.until)
    feats = path_features(paths)
    s = inputs.series[(inputs.series.bo_type == 3) & inputs.series.match_id.isin(feats.match_id)]
    rows = weekly_rows(inputs, s, skill, verbose=verbose)
    return s.merge(rows, on=["match_id", "week_cutoff"]).merge(feats, on="match_id")


def price(data_dir: str | Path, targets: pd.DataFrame, spec: Spec = SPEC, skill: SkillConfig = SKILL, verbose: bool = False,
          rows_dir: str | Path | None = None) -> pd.DataFrame:
    t = target_rows(targets)
    inputs = Inputs(data_dir, t.week_cutoff.max())
    train = training_rows(inputs, skill, verbose)
    tr = t.merge(weekly_rows(inputs, t, skill, verbose=verbose), on=["match_id", "week_cutoff"])
    tr = tr.merge(path_features(target_paths(data_dir, t.match_id)), on="match_id", how="left")
    if rows_dir is not None:
        Path(rows_dir).mkdir(parents=True, exist_ok=True)
        train.drop(columns=["lineup_a", "lineup_b"]).to_parquet(Path(rows_dir) / "train.parquet", index=False)
        tr.drop(columns=["lineup_a", "lineup_b"]).to_parquet(Path(rows_dir) / "targets.parquet", index=False)
    return combine(train, tr, spec, verbose)


def combine(train: pd.DataFrame, tr: pd.DataFrame, spec: Spec, verbose: bool = False) -> pd.DataFrame:
    """Fit the layer at each target cutoff on the series played before it and price that week's targets."""
    train = train[train.p_series.notna() & (train.n_active >= MIN_ACTIVE_PRINTS)]
    out = []
    for cutoff, week in tr.groupby("week_cutoff"):
        cutoff = pd.Timestamp(cutoff)
        past = train[train.time < cutoff]
        y = past.y.to_numpy(float)
        w = fit(design(past, spec), y, spec)
        w0 = fit(design(past, SPEC_NO_MARKET), y, SPEC_NO_MARKET)
        has_skill = week.p_series.notna()
        has_market = week.twap.notna() & (week.n_active >= MIN_ACTIVE_PRINTS)
        q = np.full(len(week), 0.5)
        both = (has_skill & has_market).to_numpy()
        skill_only = (has_skill & ~has_market).to_numpy()
        market_only = (~has_skill & has_market).to_numpy()
        if both.any():
            q[both] = layer_predict(design(week[both], spec), w)
        if skill_only.any():
            q[skill_only] = layer_predict(design(week[skill_only], SPEC_NO_MARKET), w0)
        if market_only.any():
            q[market_only] = week.twap.to_numpy(float)[market_only]
        out.append(pd.DataFrame({"match_id": week.match_id.to_numpy(), "q": q, "fit_cutoff": cutoff}))
        if verbose:
            print(f"{cutoff:%Y-%m-%d} layer on {len(past)} series: {np.round(w, 3)}  targets {len(week)} "
                  f"(skill-only {int(skill_only.sum())}, market-only {int(market_only.sum())}, neither {int((~has_skill & ~has_market).sum())})",
                  flush=True)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--rows", help="directory to also save the training and target rows in")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = naive(targets.week_cutoff)
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    out = price(a.data, targets, verbose=True, rows_dir=a.rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
