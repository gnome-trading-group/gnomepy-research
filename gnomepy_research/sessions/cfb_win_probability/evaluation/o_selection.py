"""Phase 16 ladder on Np, chosen on the 2022 validation season only (train 2020-2021).

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.o_selection --validate 2022

Steps in order: results above efficiency (outcome term), QB churn (outcome term), Group-of-5 prior width (g5
extension), coach transitions (coach extension). Each is kept unless it worsens the 2022 log loss by more than
KEEP_TOLERANCE against the last kept config. The chosen config is frozen in O_selection.json (model "O").
2023+ is never scored.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.models.cfb.efficiency.data import canonical_extensions
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.features.qb_status import (
    apply_overrides, historical_starters, load_qb_status, qb_changes, qb_deltas)
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import (
    FIRST_OUTCOME_SEASON, MODEL_CONFIGS, O_SELECTION, QB_STATUS_CSV, RESULTS, VALIDATION_SEASON)
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.n_selection import (
    KEEP_TOLERANCE, ensure_snapshots, hyper_summary, keep, score)

BASE_MODEL = "Np"
REPORT_TXT = RESULTS / f"o_selection_{VALIDATION_SEASON}.txt"


def add_step(cfg: dict, step: str) -> dict:
    if step in ("results", "qb_change"):
        return {**cfg, step: True}
    return {**cfg, "extensions": canonical_extensions((*cfg.get("extensions", ()), step), cfg["drives"])}


STEPS = ("results", "qb_change", "g5", "coach")


def label(cfg: dict) -> str:
    on = [s for s in STEPS if cfg.get(s) or s in cfg.get("extensions", ())]
    return BASE_MODEL + ("+" + "+".join(on) if on else "")


def serializable(cfg: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, required=True)
    args = ap.parse_args()
    if args.validate != VALIDATION_SEASON:
        raise SystemExit(f"model O is only selected on the {VALIDATION_SEASON} validation season")
    ds = DatasetStore()
    games = ds.load("cfb_games")
    p_open = market_probs(ds.load("cfb_lines")).set_index("game_id")["p_open"]
    src = SourceData.load(ds)
    starters = apply_overrides(historical_starters(src), load_qb_status(QB_STATUS_CSV, src))
    window = games[games["season"].between(FIRST_OUTCOME_SEASON, VALIDATION_SEASON)]
    qb = qb_deltas(src, window, starters)
    changes = qb_changes(starters, window)
    inputs = Inputs.load(ds, with_form=True, with_drives=True)
    results = {}

    def run(cfg: dict) -> float:
        name = label(cfg)
        if name not in results:
            print(f"{name}: ensuring snapshots", flush=True)
            ensure_snapshots(inputs, cfg)
            extra = {"qb_change": changes} if cfg.get("qb_change") else None
            results[name] = score(games, qb, cfg, p_open, extra) | {"hypers": hyper_summary(cfg)}
            print(f"{name}: {json.dumps(results[name])}", flush=True)
        return results[name]["overall"]

    kept = dict(MODEL_CONFIGS[BASE_MODEL])
    kept_score = run(kept)
    steps = []
    for step in STEPS:
        cand = add_step(kept, step)
        s = run(cand)
        ok = keep(s, kept_score)
        steps.append({"candidate": label(cand), "log_loss": s, "vs_kept": s - kept_score, "kept": ok})
        if ok:
            kept, kept_score = cand, s
    table = pd.DataFrame({k: {m: v[m] for m in ("overall", "big_favorites", "toss_ups")} for k, v in results.items()}).T
    text = "\n\n".join([
        f"model O ladder on {BASE_MODEL}: train {FIRST_OUTCOME_SEASON}-{VALIDATION_SEASON - 1}, score {VALIDATION_SEASON}; "
        f"keep a step unless it worsens log loss by more than {KEEP_TOLERANCE} vs the last kept config",
        f"{VALIDATION_SEASON} log loss:\n" + table.round(5).to_string(),
        "ladder:\n" + pd.DataFrame(steps).round(5).to_string(index=False),
        f"chosen: {label(kept)}"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(text + "\n")
    O_SELECTION.write_text(json.dumps({"label": label(kept), "config": serializable(kept), "steps": steps,
                                       "scores": results}, indent=2, default=float) + "\n")
    print(text)


if __name__ == "__main__":
    main()
