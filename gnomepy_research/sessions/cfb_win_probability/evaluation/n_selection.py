"""Model N extension ladder, chosen on the 2022 validation season only (train 2020-2021).

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.n_selection --validate 2022

Extensions are added in order (couple, fcs, finish) on top of K's config. Each is kept unless it worsens
the 2022 log loss by more than KEEP_TOLERANCE against the last kept config: they correct measured
misspecifications, so the rule guards against harm rather than searching for gain. Snapshots for
2020-2022 (hyperparameters 2019..S-1) are built per candidate; the outcome layer is fitted on 2020-2021
with N_SEEDS draw subsets and seeds. 2023+ is never scored.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.models.cfb.efficiency.inference import HYPER_DIR, Inputs
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import DRAWS_DIR, draws_name, hyper_name, snapshot_season
from gnomepy_research.models.cfb.features.qb_status import apply_overrides, historical_starters, load_qb_status, qb_deltas
from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.sessions.cfb_win_probability.evaluation.ablation import LIKELIHOOD, N_SEEDS
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import (
    FIRST_OUTCOME_SEASON, MODEL_CONFIGS, OUTCOME_FLAGS, QB_STATUS_CSV, RESULTS, VALIDATION_SEASON)
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.garbage_weight import segment_log_loss
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import N_FIT_DRAWS, fit, game_draws

BASE_MODEL = "K"
# The frozen Phase 13 ladder; later extensions (prior) are evaluated separately.
LADDER = ("couple", "fcs", "finish")
KEEP_TOLERANCE = 0.002
N_SELECTION = RESULTS / "N_selection.json"
REPORT_TXT = RESULTS / f"n_selection_{VALIDATION_SEASON}.txt"
HYPER_KEYS = ("tau_off", "tau_def", "s_off", "s_def", "sigma_game", "lam_ppd", "s_game_ppd", "s_drive_ppd", "kappa_ppd",
              "s_fcs", "s_fcs_ppd", "s_fin_off", "s_fin_def")


def candidate(extensions: tuple[str, ...]) -> dict:
    return {**MODEL_CONFIGS[BASE_MODEL], "extensions": extensions}


def ladder_step(kept: tuple[str, ...], ext: str) -> tuple[str, ...]:
    return tuple(e for e in LADDER if e in set(kept) | {ext})


def keep(score: float, kept_score: float, tolerance: float = KEEP_TOLERANCE) -> bool:
    return score - kept_score <= tolerance


def snapshot_kwargs(cfg: dict) -> dict:
    return dict(units=cfg["units"], conference=cfg["conference"], drives=cfg["drives"],
                independents_separate=cfg["independents_separate"], extensions=cfg.get("extensions", ()))


def ensure_snapshots(inputs: Inputs, cfg: dict) -> None:
    kw = snapshot_kwargs(cfg)
    for season in range(FIRST_OUTCOME_SEASON, VALIDATION_SEASON + 1):
        if not (DRAWS_DIR / f"{draws_name(season, **kw)}.npz").exists():
            snapshot_season(inputs, season, LIKELIHOOD, **kw)


def score(games: pd.DataFrame, qb: pd.DataFrame, cfg: dict, p_open: pd.Series, extra: dict | None = None) -> dict:
    flags = {k: cfg[k] for k in OUTCOME_FLAGS if k in cfg}
    kw = dict(units=cfg["units"], form=cfg["form"], qb=qb, **flags, **(extra or {}))
    test = game_draws(games, [VALIDATION_SEASON], **kw)
    rows = test.games["home_win"].notna().to_numpy()
    y = test.games.loc[rows, "home_win"].to_numpy()
    po = test.games.loc[rows, "game_id"].map(p_open).to_numpy(float)
    per_seed = []
    for r in range(N_SEEDS):
        train = game_draws(games, list(range(FIRST_OUTCOME_SEASON, VALIDATION_SEASON)), n_draws=N_FIT_DRAWS, seed=r, **kw)
        per_seed.append(segment_log_loss(y, fit(train, seed=r).predict(test.x[rows])["p_home"].to_numpy(), po))
    out = {k: float(np.mean([s[k] for s in per_seed])) for k in ("overall", "big_favorites", "toss_ups")}
    return out | {"overall_by_seed": [s["overall"] for s in per_seed], "n": per_seed[0]["n_overall"]}


def hyper_summary(cfg: dict) -> dict:
    name = hyper_name(LIKELIHOOD, VALIDATION_SEASON, **snapshot_kwargs(cfg))
    h = json.loads((HYPER_DIR / f"{name}.json").read_text())["hyperparameters"]
    return {k: h[k] for k in HYPER_KEYS if k in h}


def label(extensions: tuple[str, ...]) -> str:
    return BASE_MODEL + ("+" + "+".join(extensions) if extensions else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, required=True)
    args = ap.parse_args()
    if args.validate != VALIDATION_SEASON:
        raise SystemExit(f"model N is only selected on the {VALIDATION_SEASON} validation season")
    ds = DatasetStore()
    games = ds.load("cfb_games")
    p_open = market_probs(ds.load("cfb_lines")).set_index("game_id")["p_open"]
    src = SourceData.load(ds)
    starters = apply_overrides(historical_starters(src), load_qb_status(QB_STATUS_CSV, src))
    qb = qb_deltas(src, games[games["season"].between(FIRST_OUTCOME_SEASON, VALIDATION_SEASON)], starters)
    inputs = Inputs.load(ds, with_form=True, with_drives=True)

    results = {}

    def run(ext: tuple[str, ...]) -> dict:
        cfg = candidate(ext)
        print(f"{label(ext)}: ensuring snapshots", flush=True)
        ensure_snapshots(inputs, cfg)
        results[label(ext)] = score(games, qb, cfg, p_open) | {"hypers": hyper_summary(cfg)}
        print(f"{label(ext)}: {json.dumps(results[label(ext)])}", flush=True)
        return results[label(ext)]

    kept = ()
    kept_score = run(kept)["overall"]
    steps = []
    for ext in LADDER:
        cand = ladder_step(kept, ext)
        s = run(cand)["overall"]
        ok = keep(s, kept_score)
        steps.append({"candidate": label(cand), "log_loss": s, "vs_kept": s - kept_score, "kept": ok})
        if ok:
            kept, kept_score = cand, s
    table = pd.DataFrame({k: {m: v[m] for m in ("overall", "big_favorites", "toss_ups")} for k, v in results.items()}).T
    hyp = pd.DataFrame({k: {h: (np.round(v["hypers"][h], 4).tolist() if h in v["hypers"] else "") for h in HYPER_KEYS}
                        for k, v in results.items()}).T
    text = "\n\n".join([
        f"model N extension ladder on {BASE_MODEL}: train {FIRST_OUTCOME_SEASON}-{VALIDATION_SEASON - 1}, score "
        f"{VALIDATION_SEASON} ({next(iter(results.values()))['n']} games); {N_SEEDS} outcome seeds averaged; keep a step "
        f"unless it worsens log loss by more than {KEEP_TOLERANCE} vs the last kept config",
        f"{VALIDATION_SEASON} log loss:\n" + table.round(5).to_string(),
        "ladder:\n" + pd.DataFrame(steps).round(5).to_string(index=False),
        f"hyperparameters fitted on {hyper_name(LIKELIHOOD, VALIDATION_SEASON).split('_', 1)[1]} seasons:\n"
        + hyp.to_string(),
        f"chosen extensions: {list(kept)}"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(text + "\n")
    N_SELECTION.write_text(json.dumps({"extensions": list(kept), "steps": steps, "scores": results}, indent=2,
                                      default=float) + "\n")
    print(text)


if __name__ == "__main__":
    main()
