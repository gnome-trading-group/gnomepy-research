"""Phase 15: keep the tight-prior extension on Np unless it worsens the 2022 validation log loss by more than
KEEP_TOLERANCE (train 2020-2021, score 2022; 2023+ never scored).

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.tight_selection --validate 2022
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.features.qb_status import apply_overrides, historical_starters, load_qb_status, qb_deltas
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import (
    FIRST_OUTCOME_SEASON, MODEL_CONFIGS, QB_STATUS_CSV, RESULTS, VALIDATION_SEASON)
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.n_selection import (
    KEEP_TOLERANCE, ensure_snapshots, hyper_summary, keep, score)

PAIRS = (("Np", "Npt"),)
P_SELECTION = RESULTS / "T_selection.json"
REPORT_TXT = RESULTS / f"tight_selection_{VALIDATION_SEASON}.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, required=True)
    args = ap.parse_args()
    if args.validate != VALIDATION_SEASON:
        raise SystemExit(f"the tight prior is only selected on the {VALIDATION_SEASON} validation season")
    ds = DatasetStore()
    games = ds.load("cfb_games")
    p_open = market_probs(ds.load("cfb_lines")).set_index("game_id")["p_open"]
    src = SourceData.load(ds)
    starters = apply_overrides(historical_starters(src), load_qb_status(QB_STATUS_CSV, src))
    qb = qb_deltas(src, games[games["season"].between(FIRST_OUTCOME_SEASON, VALIDATION_SEASON)], starters)
    inputs = Inputs.load(ds, with_form=True, with_drives=True)
    results, decisions = {}, {}
    for base, cand in PAIRS:
        for m in (base, cand):
            if m not in results:
                print(f"{m}: ensuring snapshots", flush=True)
                ensure_snapshots(inputs, MODEL_CONFIGS[m])
                results[m] = score(games, qb, MODEL_CONFIGS[m], p_open) | {"hypers": hyper_summary(MODEL_CONFIGS[m])}
                print(f"{m}: {json.dumps(results[m])}", flush=True)
        delta = results[cand]["overall"] - results[base]["overall"]
        decisions[cand] = {"base": base, "delta": delta, "kept": keep(results[cand]["overall"], results[base]["overall"])}
    table = pd.DataFrame({k: {c: v[c] for c in ("overall", "big_favorites", "toss_ups")} for k, v in results.items()}).T
    text = "\n\n".join([
        f"tight prior: train {FIRST_OUTCOME_SEASON}-{VALIDATION_SEASON - 1}, score {VALIDATION_SEASON}; keep unless "
        f"worse than the base by more than {KEEP_TOLERANCE}",
        f"{VALIDATION_SEASON} log loss:\n" + table.round(5).to_string(),
        "decisions:\n" + pd.DataFrame(decisions).T.to_string()])
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(text + "\n")
    P_SELECTION.write_text(json.dumps({"decisions": decisions, "scores": results}, indent=2, default=float) + "\n")
    print(text)


if __name__ == "__main__":
    main()
