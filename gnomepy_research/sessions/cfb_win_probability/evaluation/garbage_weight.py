"""Garbage-time drive weight for model M, chosen on the 2022 validation season only (train 2020-2021).

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.garbage_weight --validate 2022

For each weight, K's config (split, form, conference, independents_separate, drives, phase, qb) with
normal-likelihood snapshots for 2020-2022 (hyperparameters 2019..S-1 refitted per weight). The outcome
layer is fitted on 2020-2021 with N_SEEDS draw subsets and seeds, and every 2022 score is averaged
over them. 2023+ is never scored.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import DRAWS_DIR, draws_name, snapshot_season
from gnomepy_research.models.cfb.features.drives import GARBAGE_WEIGHT
from gnomepy_research.models.cfb.features.qb_status import apply_overrides, historical_starters, load_qb_status, qb_deltas
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.sessions.cfb_win_probability.evaluation.ablation import LIKELIHOOD, N_SEEDS
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import (
    FIRST_OUTCOME_SEASON, MODEL_CONFIGS, OUTCOME_FLAGS, QB_STATUS_CSV, RESULTS, VALIDATION_SEASON)
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import N_FIT_DRAWS, fit, game_draws

WEIGHTS = (GARBAGE_WEIGHT, 0.5, 1.0)
BASE_MODEL = "K"
BIG_FAVORITE = 0.8
TOSS_UP = 0.6
M_GARBAGE = RESULTS / "M_garbage.json"
REPORT_TXT = RESULTS / f"garbage_weight_{VALIDATION_SEASON}.txt"


def pick_weight(scores: dict[float, float]) -> float:
    return min(scores, key=lambda w: scores[w])


def segment_log_loss(y: np.ndarray, p: np.ndarray, p_open: np.ndarray) -> dict:
    """Overall, big-favourite (open favourite >= BIG_FAVORITE) and toss-up (open favourite < TOSS_UP)
    log loss; games without an opening line count only overall."""
    ll = log_loss(np.asarray(y, float), np.asarray(p, float))
    fav = np.maximum(p_open, 1 - p_open)
    big, toss = fav >= BIG_FAVORITE, fav < TOSS_UP
    return {"overall": float(ll.mean()), "big_favorites": float(ll[big].mean()), "toss_ups": float(ll[toss].mean()),
            "n_overall": int(len(ll)), "n_big_favorites": int(big.sum()), "n_toss_ups": int(toss.sum())}


def ensure_weight(inputs: Inputs, cfg: dict, w: float) -> None:
    for season in range(FIRST_OUTCOME_SEASON, VALIDATION_SEASON + 1):
        name = draws_name(season, cfg["units"], cfg["conference"], drives=True,
                          independents_separate=cfg["independents_separate"], garbage_weight=w)
        if not (DRAWS_DIR / f"{name}.npz").exists():
            snapshot_season(inputs, season, LIKELIHOOD, units=cfg["units"], conference=cfg["conference"], drives=True,
                            independents_separate=cfg["independents_separate"], garbage_weight=w)


def score_weight(games: pd.DataFrame, qb: pd.DataFrame, cfg: dict, w: float, p_open: pd.Series) -> dict:
    flags = {k: cfg[k] for k in OUTCOME_FLAGS if k in cfg} | {"garbage_weight": w}
    kw = dict(units=cfg["units"], form=cfg["form"], qb=qb, **flags)
    train_seasons = list(range(FIRST_OUTCOME_SEASON, VALIDATION_SEASON))
    test = game_draws(games, [VALIDATION_SEASON], **kw)
    rows = test.games["home_win"].notna().to_numpy()
    y = test.games.loc[rows, "home_win"].to_numpy()
    po = test.games.loc[rows, "game_id"].map(p_open).to_numpy(float)
    per_seed = []
    for r in range(N_SEEDS):
        train = game_draws(games, train_seasons, n_draws=N_FIT_DRAWS, seed=r, **kw)
        p = fit(train, seed=r).predict(test.x[rows])["p_home"].to_numpy()
        per_seed.append(segment_log_loss(y, p, po))
    out = {k: float(np.mean([s[k] for s in per_seed])) for k in ("overall", "big_favorites", "toss_ups")}
    out |= {k: per_seed[0][k] for k in ("n_overall", "n_big_favorites", "n_toss_ups")}
    out["overall_by_seed"] = [s["overall"] for s in per_seed]
    return out


def format_table(scores: dict[float, dict]) -> str:
    t = pd.DataFrame({w: {k: s[k] for k in ("overall", "big_favorites", "toss_ups")} for w, s in scores.items()}).T
    t.index.name = "garbage_weight"
    first = next(iter(scores.values()))
    return (t.round(5).to_string()
            + f"\n(n: overall {first['n_overall']}, big favourites {first['n_big_favorites']}, "
              f"toss-ups {first['n_toss_ups']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, required=True)
    args = ap.parse_args()
    if args.validate != VALIDATION_SEASON:
        raise SystemExit(f"the garbage weight is only chosen on the {VALIDATION_SEASON} validation season")
    cfg = MODEL_CONFIGS[BASE_MODEL]
    ds = DatasetStore()
    games = ds.load("cfb_games")
    p_open = market_probs(ds.load("cfb_lines")).set_index("game_id")["p_open"]
    src = SourceData.load(ds)
    starters = apply_overrides(historical_starters(src), load_qb_status(QB_STATUS_CSV, src))
    qb = qb_deltas(src, games[games["season"].between(FIRST_OUTCOME_SEASON, VALIDATION_SEASON)], starters)
    inputs = Inputs.load(ds, with_form=cfg["form"], with_drives=True)
    scores = {}
    for w in WEIGHTS:
        print(f"weight {w}: ensuring snapshots", flush=True)
        ensure_weight(inputs, cfg, w)
        scores[w] = score_weight(games, qb, cfg, w, p_open)
        print(f"weight {w}: {json.dumps(scores[w])}", flush=True)
    best = pick_weight({w: s["overall"] for w, s in scores.items()})
    text = (f"garbage-time drive weight: {BASE_MODEL} config, train {FIRST_OUTCOME_SEASON}-{VALIDATION_SEASON - 1}, "
            f"score {VALIDATION_SEASON}; {N_SEEDS} outcome seeds averaged; big favourites = open favourite >= "
            f"{BIG_FAVORITE}, toss-ups = open favourite < {TOSS_UP}\n\n{VALIDATION_SEASON} log loss:\n"
            f"{format_table(scores)}\n\nchosen garbage_weight {best}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(text + "\n")
    M_GARBAGE.write_text(json.dumps({"garbage_weight": best, "scores": {str(w): s for w, s in scores.items()}},
                                    indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
