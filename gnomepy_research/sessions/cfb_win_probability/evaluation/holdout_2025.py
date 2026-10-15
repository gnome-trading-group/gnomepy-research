"""One-time 2025 holdout: build every model's 2025 snapshots (hyperparameters 2019-2024) and score each model once.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.holdout_2025 --confirm-preregistered

The rules are fixed beforehand in results/backtest/holdout_2025_preregistration.md, which must exist. Each model's
outcome layer trains on 2020-2024 (backtest's rolling origin); Kc's calibrator is fitted on 2023-24 out-of-sample
K predictions; the QB sensitivity re-scores the listed models with expected starters.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import DRAWS_DIR, draws_name, snapshot_season
from gnomepy_research.models.cfb.efficiency.projection import project_seasons, projection_table
from gnomepy_research.models.cfb.features.drives import GARBAGE_WEIGHT
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import HOLDOUT, RESULTS, resolve_config

MODELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I0", "I", "J0", "J", "K", "L", "Kc", "M", "N1", "N2", "N", "Kp", "Np"]
QB_SENSITIVITY = ["K", "N", "Kp", "Np"]
PREREGISTRATION = RESULTS / "holdout_2025_preregistration.md"
HEAVY_TAILS_MCMC_HOLDOUT = {"draws": 500, "tune": 500, "chains": 2}


def snapshot_kwargs(model: str) -> dict:
    c = resolve_config(model)
    if c.get("learner") in ("catboost", "residual"):
        c = resolve_config("C")  # their game features read the split snapshots
    return dict(units=c["units"], conference=c.get("conference", False), projection=bool(c.get("projection", False)),
                drop_features=tuple(c.get("drop_features", ())), drives=c.get("drives", False),
                independents_separate=c.get("independents_separate", False), tau_phases=c.get("tau_phases", False),
                heavy_tails=c.get("heavy_tails", False), garbage_weight=c.get("garbage_weight", GARBAGE_WEIGHT),
                extensions=tuple(c.get("extensions", ())))


def ensure_2025_snapshots(inputs: Inputs, model: str) -> None:
    kw = snapshot_kwargs(model)
    if (DRAWS_DIR / f"{draws_name(HOLDOUT, **kw)}.npz").exists():
        return
    if kw["projection"] and inputs.projection is None:
        seasons = list(range(2019, HOLDOUT + 1))
        inputs.projection = project_seasons(projection_table(inputs.preseason, SourceData.load(), inputs.games, seasons),
                                            seasons)
    print(f"{model}: building {HOLDOUT} snapshots {draws_name(HOLDOUT, **kw)}", flush=True)
    snapshot_season(inputs, HOLDOUT, "normal", **kw, mcmc=HEAVY_TAILS_MCMC_HOLDOUT if kw["heavy_tails"] else None)


def score(model: str, extra: list[str]) -> int:
    cmd = [sys.executable, "-m", "gnomepy_research.sessions.cfb_win_probability.evaluation.backtest", str(HOLDOUT),
           "--holdout", "--model", model, *extra]
    print("running:", " ".join(cmd[2:]), flush=True)
    return subprocess.run(cmd).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-preregistered", action="store_true", required=True)
    args = ap.parse_args()
    if not PREREGISTRATION.exists():
        raise SystemExit(f"{PREREGISTRATION} is missing: write the rules before scoring the holdout")
    inputs = Inputs.load(with_form=True, with_drives=True)
    for m in MODELS:
        ensure_2025_snapshots(inputs, m)
    failed = []
    for m in MODELS:
        extra = ["--calibrator-seasons", "2023", "2024"] if resolve_config(m).get("calibrator") else []
        if score(m, extra) != 0:
            failed.append(m)
    for m in QB_SENSITIVITY:
        if score(m, ["--qb-expected"]) != 0:
            failed.append(f"{m} (qb-expected)")
    print("failed:", failed or "none")


if __name__ == "__main__":
    main()
